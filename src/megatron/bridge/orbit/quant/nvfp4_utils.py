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

"""NVFP4 checkpoint-loading utilities for QOFT (mirrors peft/int4_utils.py).

The dense-linear half of the pipeline lives in
``models.conversion.low_precision.nvfp4`` and is re-exported here for
back-compat. This module adds the grouped-MoE expert pre-load shape transform
``transform_sharded_state_dict_for_nvfp4`` that matches
``*.experts.linear_fc{1,2}.weight{N}`` runtime keys and emits the per-expert
NVFP4 4-tuple entries the on-disk checkpoint stores.
"""

from __future__ import annotations

import re
from typing import Any, Dict

import torch

# Reuse INT4's shared helpers: the replica-id helper marks TP-replicated scalar
# entries so the dist-ckpt loader sees one primary writer per TP rank rather
# than every rank claiming to be the same primary; the payload + storage-view
# helpers are byte-identical between INT4 and NVFP4 paths.
from megatron.bridge.orbit.low_precision.int4 import (
    _empty_storage_view,
    _loaded_tensor_payload,
    _replica_id_with_current_tp_rank,
)

# Re-exports for back-compat with code that imports from peft.nvfp4_utils.
from megatron.bridge.orbit.low_precision.nvfp4 import (
    NVFP4_GROUP_SIZE,
    apply_modelopt_nvfp4_to_meta_model,
    build_fused_nvfp4_weight_entries,
    build_megatron_nvfp4_weight_entries,
    collect_nvfp4_target_module_names,
    dequantize_nvfp4,
    extract_nvfp4_weight_bundle,
    is_nvfp4_source,
    is_nvfp4_weight_mapping,
    register_nvfp4_buffers_after_load_dense,
    scale_to_amax,
    transform_sharded_state_dict_for_nvfp4_dense,
)


__all__ = [
    "apply_modelopt_nvfp4_to_meta_model",
    "build_fused_nvfp4_weight_entries",
    "build_megatron_nvfp4_weight_entries",
    "collect_nvfp4_target_module_names",
    "dequantize_nvfp4",
    "extract_nvfp4_weight_bundle",
    "is_nvfp4_source",
    "is_nvfp4_weight_mapping",
    "register_nvfp4_buffers_after_load",
    "register_nvfp4_buffers_after_load_dense",
    "scale_to_amax",
    "transform_sharded_state_dict_for_nvfp4",
    "transform_sharded_state_dict_for_nvfp4_dense",
]


# Matches grouped-MoE per-expert weight keys: ``*.experts.linear_fc1.weight0``
# (the digit suffix is the runtime local-expert index — for EP>1 it is rewritten
# below to the global expert index).
_EXPERT_WEIGHT_RE = re.compile(r"^(.*\.experts\.linear_fc[12])\.weight(\d+)$")
# Matches the base "weight" key (no digit) emitted by ``sharded_state_dict()``
# for grouped-linear modules. The checkpoint does not store this key — the
# per-expert ``weight{N}`` entries cover all experts — so we drop it from the
# loaded plan to avoid a "missing in checkpoint" error.
_EXPERT_BASE_WEIGHT_RE = re.compile(r"^.*\.experts\.linear_fc[12]\.weight$")
# Distinguishes SwiGLU fused-gate+up fc1 from the non-fused fc2 entries.
_FC1_RE = re.compile(r"\.linear_fc1\.weight\d+$")


def _canonicalize_expert_key_for_checkpoint(key: str) -> str:
    """Map runtime expert state-dict keys to the checkpoint's canonical key format."""
    return key.replace(".experts.experts.", ".experts.")


def transform_sharded_state_dict_for_nvfp4(
    sharded_state_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Replace grouped-MoE expert weight ShardedTensors with NVFP4 4-tuple entries.

    For ``*.experts.linear_fc1.weight{N}`` (SwiGLU fused gate+up), emit
    ``weight{N}_w`` (gate) + ``weight{N}_v`` (up) halves at the packed shape,
    plus a shared ``weight_quantizer._scale{N}`` (fp8_e4m3fn, ``[out, in/16]``)
    and two ``weight_quantizer._double_scale{N}`` / ``_amax{N}`` fp32 scalars.

    For ``*.experts.linear_fc2.weight{N}`` (non-fused), emit ``weight{N}`` at
    the packed shape plus the same three quantizer entries.

    Non-expert keys pass through unchanged. The base ``*.experts.linear_fc{1,2}
    .weight`` key (no digit) emitted by ``sharded_state_dict()`` is dropped
    because the checkpoint stores per-expert keys only.

    Meta-device ``ShardedTensor`` entries are materialized on CPU so PyTorch's
    distributed checkpoint planner can handle them; the loader overwrites the
    data with the actual values from disk.
    """
    from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory

    new_sd: Dict[str, Any] = {}

    for key, value in sharded_state_dict.items():
        skey = str(key)

        # Materialize meta-device tensors on CPU so the DCP planner can handle them.
        if isinstance(value, ShardedTensor) and value.data is not None and value.data.device.type == "meta":
            value.data = torch.empty(value.local_shape, dtype=value.dtype, device="cpu")

        canonical = _canonicalize_expert_key_for_checkpoint(skey)

        # The grouped base "weight" key (no digit) has no checkpoint entry — drop it.
        if _EXPERT_BASE_WEIGHT_RE.match(canonical):
            continue

        m = _EXPERT_WEIGHT_RE.match(canonical)
        if m is None:
            new_sd[key] = value
            continue

        is_fc1 = _FC1_RE.search(canonical) is not None

        # The expert weight can be either:
        #   - ShardedTensorFactory (fc1 SwiGLU: factory.build() splits gate/up halves)
        #   - ShardedTensor (fc2: simple per-expert weight)
        # For factories, the checkpoint stores the fused gate+up weight, so we
        # restore the fused local out-dim, fused out offset, and fused out-axis
        # fragmentation before constructing the NVFP4 entries.
        if isinstance(value, ShardedTensorFactory):
            built_sub = value.build()
            if isinstance(built_sub, list):
                sub_sh_ten = built_sub[0]
                sub_sh_tens = built_sub
            else:
                sub_sh_ten = next(iter(built_sub.values()))
                sub_sh_tens = list(built_sub.values())
            sh_ten = sub_sh_ten

            fused_local_out = value.data.shape[0]
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
                # singleton_local_shards=False: sub-shards tile the fused grid.
                if axis_fragmentations_override[out_axis] % split_factor != 0:
                    raise ValueError(
                        f"Unexpected SwiGLU factory fragmentation for {key}: "
                        f"{axis_fragmentations_override[out_axis]} not divisible by {split_factor}"
                    )
                axis_fragmentations_override[out_axis] //= split_factor
                fused_global_out = sh_ten.global_shape[out_axis]
            else:
                # singleton_local_shards=True: split keys -> rebuild fused grid.
                fused_global_out = sh_ten.global_shape[out_axis] * split_factor

            local_out = fused_local_out
            global_out = fused_global_out
            out_offset = rank_offset * fused_local_out
            axis_fragmentations = tuple(axis_fragmentations_override)
        elif isinstance(value, ShardedTensor):
            sh_ten = value
            local_out = sh_ten.local_shape[-2]
            global_out = sh_ten.global_shape[-2]
            out_offset = sh_ten.global_offset[sh_ten.prepend_axis_num]
            axis_fragmentations = sh_ten.axis_fragmentations
        else:
            new_sd[key] = value
            continue

        local_in = sh_ten.local_shape[-1]
        global_in = sh_ten.global_shape[-1]
        in_offset = (
            sh_ten.global_offset[sh_ten.prepend_axis_num + 1]
            if len(sh_ten.global_offset) > sh_ten.prepend_axis_num + 1
            else 0
        )
        prepend = sh_ten.prepend_axis_num
        # Strip the prepended expert/layer axis from the weight shape tuple.
        weight_axis_fragmentations = (
            tuple(axis_fragmentations[prepend:]) if prepend > 0 else tuple(axis_fragmentations)
        )
        # For grouped MoE with EP, the local runtime index N differs from the
        # global expert index (e.g. EP rank 1 of 4 with 96 local experts maps
        # local weight0 -> global weight96). Use the prepend axis's global_offset
        # entry when present, otherwise the literal digit from the runtime key.
        expert_global_idx = sh_ten.global_offset[prepend - 1] if prepend > 0 else int(m.group(2))

        # ``prefix`` is extracted from the canonical-matched regex, so it is
        # already canonicalized. Use it uniformly for both the runtime dict
        # keys and the on-disk ckpt key fields so the post-load register step
        # can find packed/scale/scalar entries under one consistent path
        # (matches int4_utils.py and fp8_utils.py).
        prefix = m.group(1)

        if is_fc1:
            # SwiGLU: gate (_w) and up (_v) halves at half-out, half-in (FP4 packed).
            half_in = local_in // 2
            global_half_in = global_in // 2
            for half_suffix in ("_w", "_v"):
                ckpt_key = f"{prefix}.weight{expert_global_idx}{half_suffix}"
                new_st = ShardedTensor(
                    key=ckpt_key,
                    data=torch.empty((local_out // 2, half_in), dtype=torch.uint8, device="cpu"),
                    dtype=torch.uint8,
                    local_shape=(local_out // 2, half_in),
                    global_shape=(global_out // 2, global_half_in),
                    global_offset=(out_offset // 2, in_offset // 2),
                    axis_fragmentations=weight_axis_fragmentations,
                    replica_id=sh_ten.replica_id,
                    prepend_axis_num=0,
                    allow_shape_mismatch=True,
                )
                new_sd[f"{canonical}{half_suffix}"] = new_st
        else:
            # fc2: non-fused, single packed weight at full-out, half-in.
            half_in = local_in // 2
            global_half_in = global_in // 2
            ckpt_key = f"{prefix}.weight{expert_global_idx}"
            new_st = ShardedTensor(
                key=ckpt_key,
                data=torch.empty((local_out, half_in), dtype=torch.uint8, device="cpu"),
                dtype=torch.uint8,
                local_shape=(local_out, half_in),
                global_shape=(global_out, global_half_in),
                global_offset=(out_offset, in_offset // 2),
                axis_fragmentations=weight_axis_fragmentations,
                replica_id=sh_ten.replica_id,
                prepend_axis_num=0,
                allow_shape_mismatch=True,
            )
            new_sd[canonical] = new_st

        # Per-expert quantizer entries: one set per expert. For SwiGLU the scale
        # spans the fused full intermediate (not halved) and is shared between
        # the _w and _v halves at runtime via the same ``weight_quantizer._scale{N}``.
        scale_cols = local_in // NVFP4_GROUP_SIZE
        global_scale_cols = global_in // NVFP4_GROUP_SIZE
        scale_st = ShardedTensor(
            key=f"{prefix}.weight_quantizer._scale{expert_global_idx}",
            data=torch.empty((local_out, scale_cols), dtype=torch.float8_e4m3fn, device="cpu"),
            dtype=torch.float8_e4m3fn,
            local_shape=(local_out, scale_cols),
            global_shape=(global_out, global_scale_cols),
            global_offset=(out_offset, in_offset // NVFP4_GROUP_SIZE),
            axis_fragmentations=weight_axis_fragmentations,
            replica_id=sh_ten.replica_id,
            prepend_axis_num=0,
            allow_shape_mismatch=True,
        )
        new_sd[f"{prefix}.weight_quantizer._scale{expert_global_idx}"] = scale_st

        for scalar_suffix in ("_double_scale", "_amax"):
            scalar_st = ShardedTensor(
                key=f"{prefix}.weight_quantizer.{scalar_suffix}{expert_global_idx}",
                data=torch.empty((), dtype=torch.float32, device="cpu"),
                dtype=torch.float32,
                local_shape=(),
                global_shape=(),
                global_offset=(),
                axis_fragmentations=(),
                # TP-replicated scalar — wrap replica_id with the current TP rank
                # so the dist-ckpt loader sees one primary writer per TP rank.
                replica_id=_replica_id_with_current_tp_rank(sh_ten.replica_id),
                prepend_axis_num=0,
                allow_shape_mismatch=True,
            )
            new_sd[f"{prefix}.weight_quantizer.{scalar_suffix}{expert_global_idx}"] = scalar_st

    return new_sd


# Matches per-expert NVFP4 entries in the loaded state dict. The regex has two
# alternatives:
#   weight{N}            -> group(2)=N, group(3)=None    (fc2 packed)
#   weight{N}_w / _v     -> group(2)=N, group(3)="_w"/"_v" (fc1 SwiGLU halves)
#   weight_quantizer._scale{N} / _double_scale{N} / _amax{N}
#                        -> group(4)=name, group(5)=N
_TRIPLET_OR_QUAD_RE = re.compile(
    r"^(.*\.experts\.linear_fc[12])\."
    r"(?:weight(\d+)(_w|_v)?|weight_quantizer\._(scale|double_scale|amax)(\d+))$"
)


def register_nvfp4_buffers_after_load(
    model,
    loaded_state_dict: Dict[str, Any],
) -> None:
    """Register loaded NVFP4 expert buffers on modules; empty bf16 placeholders.

    Walks ``loaded_state_dict``, groups entries by ``(module_path, expert_idx)``,
    and on each grouped MoE module registers per-expert NVFP4 buffers:

      - ``weight{N}_packed`` (fc2) or ``weight{N}_w_packed`` + ``weight{N}_v_packed``
        (fc1 SwiGLU)
      - ``weight_scale{N}``, ``weight_double_scale{N}``, ``weight_amax{N}``

    Then empties the bf16 ``weight{N}`` Parameter (for fc1, both halves were
    merged into one Parameter pre-``mtq.compress``; post-load we don't need it).
    Also empties the base concat ``weight`` (mirrors INT4 expert register).
    """
    groups: Dict[tuple, Dict[str, torch.Tensor]] = {}
    for key, value in loaded_state_dict.items():
        m = _TRIPLET_OR_QUAD_RE.match(str(key))
        if m is None:
            continue
        module_path = m.group(1)
        # weight{N}(_w|_v)? -> group(2) holds N
        # weight_quantizer._{scale|double_scale|amax}{N} -> group(5) holds N
        expert_idx = m.group(2) if m.group(2) is not None else m.group(5)
        if expert_idx is None:
            continue
        gk = (module_path, expert_idx)
        groups.setdefault(gk, {})
        if m.group(2) is not None:
            # weight{N}, weight{N}_w, weight{N}_v
            half = m.group(3) or ""  # "", "_w", "_v"
            groups[gk][f"weight{half}"] = value
        else:
            buffer = m.group(4)  # scale / double_scale / amax
            groups[gk][buffer] = value

    registered = 0
    for (module_path, idx), parts in groups.items():
        # Navigate to the module.
        module = model
        for attr in module_path.split("."):
            module = getattr(module, attr)

        for src_name, dest_name in (
            ("weight", f"weight{idx}_packed"),
            ("weight_w", f"weight{idx}_w_packed"),
            ("weight_v", f"weight{idx}_v_packed"),
            ("scale", f"weight_scale{idx}"),
            ("double_scale", f"weight_double_scale{idx}"),
            ("amax", f"weight_amax{idx}"),
        ):
            if src_name in parts:
                module.register_buffer(dest_name, _loaded_tensor_payload(parts[src_name]), persistent=True)

        # Empty the bf16 weight{idx} Parameter (placeholder; runtime uses buffers).
        weight_name = f"weight{idx}"
        w = getattr(module, weight_name, None)
        if w is not None and hasattr(w, "data"):
            w.data = _empty_storage_view(w.data)
        registered += 1

    # Also empty the base concat `weight` per touched module (mirrors INT4 behavior).
    seen_modules = set()
    for (module_path, _), _ in groups.items():
        if module_path in seen_modules:
            continue
        seen_modules.add(module_path)
        module = model
        for attr in module_path.split("."):
            module = getattr(module, attr)
        base_w = getattr(module, "weight", None)
        if base_w is not None and hasattr(base_w, "data"):
            base_w.data = _empty_storage_view(base_w.data)

    print(f"[NVFP4 expert register] Registered NVFP4 buffers for {registered} expert weights")
