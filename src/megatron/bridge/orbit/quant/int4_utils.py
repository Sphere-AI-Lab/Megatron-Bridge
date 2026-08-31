# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""INT4 checkpoint-loading utilities for QOFT.

These helpers operate on the canonical Kimi-K2 native INT4 triplet format:
    weight_packed:  [out, in // 8]   int32
    weight_scale:   [out, in // 32]  float16
    weight_shape:   [2]              int64
"""

import re
from typing import Any, Dict

import torch


# Canonical INT4 checkpoint triplet suffixes -- the single source of truth every
# INT4 checkpoint reader/writer in orbit should use instead of hand-spelling
# these strings (see docs/orbit/MIGRATION_MEMORY.md for why: this exact class of
# drift risk already caused two real bugs in the FP8/NVFP4 equivalents).
INT4_PACKED_SUFFIX = "_packed"
INT4_SCALE_SUFFIX = "_scale"
INT4_SHAPE_SUFFIX = "_shape"

_EXPERT_WEIGHT_RE = re.compile(r"^(.*\.experts\.linear_fc[12])\.weight(\d+)$")
# Also matches the base "weight" key (no digit suffix) for expert grouped linears.
# This key is produced by sharded_state_dict() but has no corresponding checkpoint
# entry — the per-expert weight{N} entries handle all experts.
_EXPERT_BASE_WEIGHT_RE = re.compile(r"^.*\.experts\.linear_fc[12]\.weight$")


def _empty_storage_view(tensor: torch.Tensor) -> torch.Tensor:
    """Return a zero-length view that preserves the tensor's device/type."""
    if tensor.ndim == 0:
        return tensor.reshape(1)[:0]
    return tensor.narrow(0, 0, 0)


def _loaded_tensor_payload(value: Any) -> torch.Tensor:
    """Extract the real tensor payload from raw tensors or ShardedTensor entries."""

    if isinstance(value, torch.Tensor):
        return value

    data = getattr(value, "data", None)
    if isinstance(data, torch.Tensor):
        return data

    raise TypeError(f"Expected loaded tensor payload, got {type(value).__name__}")


def _canonicalize_expert_key_for_checkpoint(key: str) -> str:
    """Map runtime expert state-dict keys to the checkpoint's canonical key format."""
    return key.replace(".experts.experts.", ".experts.")


def transform_sharded_state_dict_for_int4(
    sharded_state_dict: Dict[str, Any],
    group_size: int = 32,
    scale_dtype: torch.dtype = torch.float16,
) -> Dict[str, Any]:
    """Replace expert BF16 weight entries with INT4 triplet entries.

    Given a sharded state dict produced by ``model.sharded_state_dict()``,
    finds every expert weight key (``*.experts.linear_fc{1,2}.weight{N}``)
    and replaces the single BF16 ``ShardedTensor`` with three INT4-format
    ``ShardedTensor`` entries (``*_packed``, ``*_scale``, ``*_shape``) whose
    shapes and keys match what the direct INT4 converter wrote.

    Non-expert keys are left unchanged (but their meta tensors are materialized
    on CPU so PyTorch's distributed checkpoint planner can handle them).
    """
    # Example: for linear_fc2.weight0 on EP rank 1 of 4 (96 local experts):
    #
    #   Original BF16 ShardedTensor from model.sharded_state_dict():
    #     key            = "...linear_fc2.0.weight"
    #     local_shape    = (7168, 2048)          # one expert's weight, 2D
    #     global_shape   = (384, 7168, 2048)     # all experts, 3D (EP prepended)
    #     global_offset  = (96, 0, 0)            # this rank starts at expert 96
    #     prepend_axis_num = 1                   # EP axis only in global, not local
    #
    #   The direct INT4 converter stores each global expert as its own checkpoint
    #   key (weight0, weight1, ..., weight383), with no prepended expert axis.
    #   So on EP rank 1 the local runtime key `weight0` must map to checkpoint key
    #   `weight96`, and the expected checkpoint tensor shapes remain 2D / 1D:
    #     weight96_packed: local (7168, 256), global (7168, 256), int32
    #     weight96_scale:  local (7168, 64),  global (7168, 64),  fp16
    #     weight96_shape:  local (2,),        global (2,),         int64
    #
    #   Where 256 = 2048/8 (8 INT4 values per int32), 64 = 2048/32 (one scale per group)
    from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory

    canonical_keys = {_canonicalize_expert_key_for_checkpoint(k) for k in sharded_state_dict.keys()}
    new_sd: Dict[str, Any] = {}
    for key, value in sharded_state_dict.items():
        # Materialize any meta-device ShardedTensors on CPU so PyTorch's
        # distributed checkpoint planner can handle them. The loader will
        # overwrite the data with actual values from the checkpoint.
        if isinstance(value, ShardedTensor) and value.data is not None and value.data.device.type == "meta":
            value.data = torch.empty(value.local_shape, dtype=value.dtype, device="cpu")

        # Skip the base "weight" key (no digit) for expert grouped linears.
        # The checkpoint doesn't have it — per-expert weight{N} keys cover all experts.
        canonical_key = _canonicalize_expert_key_for_checkpoint(key)

        if _EXPERT_BASE_WEIGHT_RE.match(canonical_key):
            continue

        m = _EXPERT_WEIGHT_RE.match(canonical_key)
        if m is None:
            # Non-expert keys (dense, norm, embedding, vision) pass through
            new_sd[key] = value
            continue

        # If the module already exposes persistent INT4 buffers for this expert
        # weight (e.g. after loading an INT4 checkpoint for QOFT), prefer those
        # real triplets and drop the emptied BF16 placeholder weight entry.
        if (
            f"{canonical_key}{INT4_PACKED_SUFFIX}" in canonical_keys
            and f"{canonical_key}{INT4_SCALE_SUFFIX}" in canonical_keys
            and f"{canonical_key}{INT4_SHAPE_SUFFIX}" in canonical_keys
        ):
            continue

        # Expert weight key matched. It can be either:
        # - ShardedTensor (fc2 weights, non-gated)
        # - ShardedTensorFactory (fc1 weights with SwiGLU gate/up splitting)
        # For factories, we call build() to get the actual ShardedTensors with
        # correct EP metadata, then use one of them for shape/offset info.
        # The checkpoint stores the fused gate+up weight, so we need the FUSED
        # shape — we get this from the factory's data tensor.
        if isinstance(value, ShardedTensorFactory):
            # build() splits the fused weight into gate/up halves as ShardedTensors.
            # We grab one to extract EP sharding metadata (prepend, offsets, etc.).
            built_sub = value.build()
            if isinstance(built_sub, list):
                sub_sh_ten = built_sub[0]  # gate half
                sub_sh_tens = built_sub
            else:
                sub_sh_ten = next(iter(built_sub.values()))
                sub_sh_tens = list(built_sub.values())

            # The factory sub-shards split the fused SwiGLU weight into gate/up halves.
            # For the INT4 checkpoint we need metadata for the original fused tensor, not
            # the split sub-shards. In the common non-singleton path, the sub-shards:
            # - halve the local out-dimension
            # - double the out-axis fragmentation
            # - use out offsets in the split-tensor grid
            #
            # So we restore the fused local out-dimension, fused out offset, and fused
            # out-axis fragmentation here before constructing INT4 triplets.
            sh_ten = sub_sh_ten
            fused_local_out = value.data.shape[0]  # fused gate+up dim
            split_local_out = sh_ten.local_shape[-2]
            if split_local_out == 0 or fused_local_out % split_local_out != 0:
                raise ValueError(
                    f"Unexpected SwiGLU factory shapes for {key}: "
                    f"fused local out={fused_local_out}, split local out={split_local_out}"
                )
            split_factor = fused_local_out // split_local_out
            out_axis = sh_ten.prepend_axis_num
            rank_offset = sh_ten.global_offset[out_axis] // split_local_out

            axis_fragmentations_override = list(sh_ten.axis_fragmentations)
            if all(sub.key == value.key for sub in sub_sh_tens):
                # singleton_local_shards=False: sub-shards tile the fused tensor grid.
                if axis_fragmentations_override[out_axis] % split_factor != 0:
                    raise ValueError(
                        f"Unexpected SwiGLU factory fragmentation for {key}: "
                        f"{axis_fragmentations_override[out_axis]} not divisible by {split_factor}"
                    )
                axis_fragmentations_override[out_axis] //= split_factor
                fused_global_out = sh_ten.global_shape[out_axis]
            else:
                # singleton_local_shards=True: keys are split, so rebuild the fused grid.
                fused_global_out = sh_ten.global_shape[out_axis] * split_factor

            local_out_override = fused_local_out
            global_out_override = fused_global_out
            out_offset_override = rank_offset * fused_local_out
            axis_fragmentations_override = tuple(axis_fragmentations_override)
            local_in = sh_ten.local_shape[-1]
        elif isinstance(value, ShardedTensor):
            sh_ten = value
            local_out_override = None
            global_out_override = None
            out_offset_override = None
            axis_fragmentations_override = sh_ten.axis_fragmentations
            local_in = sh_ten.local_shape[-1]
        else:
            new_sd[key] = value
            continue
        prepend = sh_ten.prepend_axis_num

        # The last two dims are the actual weight dimensions (out, in).
        # For SwiGLU factories, local_out is overridden to the fused size
        # (gate+up concatenated) since the checkpoint stores them fused.
        local_out = local_out_override if local_out_override is not None else sh_ten.local_shape[-2]
        global_out = global_out_override if global_out_override is not None else sh_ten.global_shape[-2]
        global_in = sh_ten.global_shape[-1]

        # INT4 packed: 8 INT4 values packed into one int32 along in-features dim
        packed_in = local_in // 8  # e.g. 2048 / 8 = 256
        global_packed_in = global_in // 8
        # Scale: one fp16 scale value per group of `group_size` elements
        num_groups = local_in // group_size  # e.g. 2048 / 32 = 64
        global_num_groups = global_in // group_size

        # Weight dim offsets (after any prepended axes). Usually 0 for TP=1.
        out_offset = out_offset_override if out_offset_override is not None else sh_ten.global_offset[prepend]
        in_offset = sh_ten.global_offset[prepend + 1] if len(sh_ten.global_offset) > prepend + 1 else 0

        # The checkpoint stores one key per global expert rather than a single tensor
        # with a prepended expert axis, so we:
        # 1. drop the prepended expert axis from shapes/fragmentations/offsets
        # 2. rewrite the checkpoint key to the global expert index on this rank
        weight_axis_fragmentations = axis_fragmentations_override[prepend:]
        expert_global_idx = sh_ten.global_offset[prepend - 1] if prepend > 0 else int(m.group(2))
        ckpt_key = re.sub(r"weight\d+$", f"weight{expert_global_idx}", canonical_key)

        # Build (suffix, local_shape, global_shape, global_offset, dtype) for each triplet.
        # local_shape is 2D (or 1D for shape), and global_shape matches the
        # checkpoint's per-expert tensor layout (no prepended expert axis).
        triplets = [
            # weight_packed: [out, in/8] int32 — the quantized weights
            (
                INT4_PACKED_SUFFIX,
                (local_out, packed_in),
                (global_out, global_packed_in),
                (out_offset, in_offset // 8),
                weight_axis_fragmentations,
                torch.int32,
            ),
            # weight_scale: [out, in/group_size] fp16 — per-group scale factors
            (
                INT4_SCALE_SUFFIX,
                (local_out, num_groups),
                (global_out, global_num_groups),
                (out_offset, in_offset // group_size),
                weight_axis_fragmentations,
                scale_dtype,
            ),
            # weight_shape: [2] int64 — original (out, in) dimensions
            (INT4_SHAPE_SUFFIX, (2,), (2,), (0,), (1,), torch.int64),
        ]

        for suffix, local_sh, global_sh, off, axis_frags, dtype in triplets:
            # Allocate a real CPU tensor (not meta) — the checkpoint loader will
            # overwrite it with actual data from disk. Meta tensors are not
            # supported by PyTorch's distributed checkpoint planner.
            data = torch.empty(local_sh, dtype=dtype, device="cpu")
            new_sh_ten = ShardedTensor(
                key=ckpt_key + suffix,
                data=data,
                dtype=dtype,
                local_shape=local_sh,
                global_shape=global_sh,
                global_offset=off,
                axis_fragmentations=axis_frags,
                replica_id=sh_ten.replica_id,
                prepend_axis_num=0,
                allow_shape_mismatch=True,  # shape tensor (2,) vs global (384, 2)
            )
            new_sd[canonical_key + suffix] = new_sh_ten

    return new_sd


def register_int4_buffers_after_load(
    model: "torch.nn.Module",
    loaded_state_dict: Dict[str, Any],
) -> None:
    """Register loaded INT4 triplets as buffers on expert modules.

    After ``dist_checkpointing.load()`` fills the sharded state dict with
    real tensors, this walks the dict, finds INT4 triplet entries, registers
    them as persistent buffers on the corresponding modules, and empties the
    BF16 ``.weight{N}`` parameters to free memory.
    """
    _TRIPLET_RE = re.compile(r"^(.*\.experts\.linear_fc[12])\.weight(\d+)_(packed|scale|shape)$")

    # Group triplets by (module_path, expert_idx)
    triplets: Dict[tuple, Dict[str, torch.Tensor]] = {}
    for key, value in loaded_state_dict.items():
        m = _TRIPLET_RE.match(key)
        if m is None:
            continue
        group_key = (m.group(1), m.group(2))
        if group_key not in triplets:
            triplets[group_key] = {}
        triplets[group_key][m.group(3)] = value

    registered = 0
    for (module_path, idx), parts in triplets.items():
        if len(parts) != 3:
            continue

        # Navigate to the module
        module = model
        for attr in module_path.split("."):
            module = getattr(module, attr)

        weight_name = f"weight{idx}"
        w = getattr(module, weight_name, None)
        # When loading into a meta-initialized model, expert parameters still live on
        # the meta device here. Registering INT4 buffers onto `meta` would discard the
        # loaded payload and later `to_empty_if_meta_device(...)` would materialize them
        # as empty CUDA tensors. Keep the real loaded tensors on CPU until the normal
        # post-load device materialization step moves them to the target device.
        payloads = {suffix: _loaded_tensor_payload(parts[suffix]) for suffix in ("packed", "scale", "shape")}
        target_device = payloads["packed"].device
        if w is not None and w.device.type != "meta":
            target_device = w.device
        for suffix in ("packed", "scale", "shape"):
            module.register_buffer(
                f"{weight_name}_{suffix}",
                payloads[suffix].to(target_device),
                persistent=True,
            )

        # Empty the BF16 weight parameter to free memory
        if w is not None:
            w.data = _empty_storage_view(w.data)
        registered += 1

    # Also empty the base concatenated `weight` tensor on each expert module.
    # TE's GroupedLinear stores all experts in one big [num_experts * out, in]
    # tensor — useless memory now that we have INT4 buffers per expert.
    freed_modules = set()
    for (module_path, _), _ in triplets.items():
        if module_path in freed_modules:
            continue
        freed_modules.add(module_path)
        module = model
        for attr in module_path.split("."):
            module = getattr(module, attr)
        base_w = getattr(module, "weight", None)
        if base_w is not None and hasattr(base_w, "data"):
            base_w.data = _empty_storage_view(base_w.data)

    print(f"Registered INT4 buffers for {registered} expert weights, freed {len(freed_modules)} base weight tensors")
