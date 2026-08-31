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

"""FP8 utilities for QOFT (Quantized OFT).

Block-wise FP8 dequantization, per-tensor quantization, and scale-merge
helpers for QKV / GatedMLP fusion during checkpoint conversion.
"""

import math
import re
from typing import Any, Dict, Tuple

import torch


FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
FP8_WEIGHT_BLOCK_SIZE = 128

# --------------------------------------------------------------------------- #
# The FP8 checkpoint key format
# --------------------------------------------------------------------------- #

# FP8 stores each weight next to a block-scale sibling, and a SwiGLU linear_fc1
# additionally stores its gate and up halves separately. Both the transform that
# requests these entries and the post-load step that consumes them must agree on
# the spelling; defining the suffixes once is what keeps them from drifting.
FP8_SCALE_INV_SUFFIX = "_scale_inv"
FP8_SWIGLU_HALF_SUFFIXES = ("_w", "_v")


def fp8_entry_names(weight_key: str, *, swiglu: bool = False) -> dict[str, str]:
    """Canonical on-disk names for one FP8 weight and its block-scale sibling.

    This is the single definition of the FP8 checkpoint key format. Anything
    that spells these keys out by hand is a drift waiting to happen: a
    disagreement between the names stored and the names requested surfaces as an
    opaque distributed-checkpoint KeyError that reads like a corrupt file.

    Args:
        weight_key: Fully qualified weight key, e.g.
            ``decoder.layers.0.mlp.experts.linear_fc1.weight3``.
        swiglu: Whether the weight is a fused SwiGLU gate+up projection, whose
            halves are stored separately.

    Returns:
        Mapping of role (``weight`` or ``weight_w``/``weight_v``, plus
        ``scale_inv``) to its canonical checkpoint key.
    """
    names = {"scale_inv": f"{weight_key}{FP8_SCALE_INV_SUFFIX}"}
    if swiglu:
        names["weight_w"] = f"{weight_key}{FP8_SWIGLU_HALF_SUFFIXES[0]}"
        names["weight_v"] = f"{weight_key}{FP8_SWIGLU_HALF_SUFFIXES[1]}"
    else:
        names["weight"] = weight_key
    return names


_GROUPED_EXPERT_WEIGHT_RE = re.compile(r"^(.*\.experts\.linear_fc[12])\.weight(\d+)$")
_GROUPED_EXPERT_BASE_WEIGHT_RE = re.compile(r"^.*\.experts\.linear_fc[12]\.weight$")
_GROUPED_EXPERT_SCALE_INV_RE = re.compile(rf"^(.*\.experts\.linear_fc[12])\.weight(\d+){FP8_SCALE_INV_SUFFIX}$")
_GROUPED_EXPERT_SPLIT_WEIGHT_RE = re.compile(r"^(.*\.experts\.linear_fc1)\.weight(\d+)_([wv])$")
_DIRECT_FP8_LINEAR_WEIGHT_RE = re.compile(r"^(.*\.(?:linear_qkv|linear_proj|linear_fc1|linear_fc2))\.weight$")
_DIRECT_FP8_SCALE_INV_RE = re.compile(
    rf"^(.*\.(?:linear_qkv|linear_proj|linear_fc1|linear_fc2))\.weight{FP8_SCALE_INV_SUFFIX}$"
)
_DIRECT_FP8_SPLIT_WEIGHT_RE = re.compile(r"^(.*\.linear_fc1)\.weight_([wv])$")


def _canonicalize_expert_key_for_checkpoint(key: str) -> str:
    return key.replace(".experts.experts.", ".experts.")


def _is_extra_state_entry(key: str, value: Any) -> bool:
    candidates = [key, getattr(value, "key", None), getattr(value, "unique_key", None)]
    return any(
        isinstance(candidate, str) and "_extra_state" in _canonicalize_expert_key_for_checkpoint(candidate)
        for candidate in candidates
    )


def _loaded_tensor_payload(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value

    data = getattr(value, "data", None)
    if isinstance(data, torch.Tensor):
        return data

    raise TypeError(f"Expected loaded tensor payload, got {type(value).__name__}")


def _fp8_scale_shape_for_weight(weight_shape: Tuple[int, ...], block_size: int) -> Tuple[int, ...]:
    if len(weight_shape) == 1:
        return (max(1, math.ceil(weight_shape[0] / block_size)),)
    if len(weight_shape) >= 2:
        return tuple(weight_shape[:-2]) + (
            max(1, math.ceil(weight_shape[-2] / block_size)),
            max(1, math.ceil(weight_shape[-1] / block_size)),
        )
    raise ValueError(f"Unsupported FP8 weight shape: {weight_shape}")


def transform_sharded_state_dict_for_fp8(
    sharded_state_dict: Dict[str, Any],
    block_size: int = FP8_WEIGHT_BLOCK_SIZE,
) -> Dict[str, Any]:
    """Rewrite FP8 weights to the direct-checkpoint schema.

    Grouped expert linears expose local runtime keys like ``weight0`` while the
    underlying sharded tensor metadata still points at the canonical base
    ``weight`` tensor. The direct FP8 checkpoint stores one key per global
    expert plus a sibling ``*_scale_inv`` tensor. This transform keeps the
    runtime-facing dict keys local (for ``load_state_dict``) but rewrites the
    underlying checkpoint keys to the global expert indices expected on disk.

    Dense direct-FP8 checkpoints written from ModelOpt-shaped meta models store
    linear weights as ``*.weight_w`` plus ``*.weight_scale_inv``.  Runtime
    Megatron modules still request ``*.weight``, so their sharded tensor entries
    are pointed at the on-disk ``*_w`` key and a scale entry is added for the
    post-load hook to register.
    """

    from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory

    def _split_factory(value: ShardedTensorFactory):
        built_sub = value.build()
        if isinstance(built_sub, list):
            return built_sub[0], built_sub
        sub_sh_ten = next(iter(built_sub.values()))
        return sub_sh_ten, list(built_sub.values())

    def _add_scale_entry(
        out: Dict[str, Any],
        scale_key: str,
        sh_ten: ShardedTensor,
        weight_local_shape: Tuple[int, ...],
        weight_global_shape: Tuple[int, ...],
        weight_global_offset: Tuple[int, ...],
        axis_fragmentations: Tuple[int, ...],
        replica_id=None,
        checkpoint_key: str | None = None,
    ) -> None:
        if replica_id is None:
            replica_id = sh_ten.replica_id
        scale_local_shape = _fp8_scale_shape_for_weight(weight_local_shape, block_size)
        scale_global_shape = _fp8_scale_shape_for_weight(weight_global_shape, block_size)
        scale_global_offset = tuple(offset // block_size for offset in weight_global_offset)
        out[scale_key] = ShardedTensor(
            key=checkpoint_key or scale_key,
            data=torch.empty(scale_local_shape, dtype=torch.float32, device="cpu"),
            dtype=torch.float32,
            local_shape=scale_local_shape,
            global_shape=scale_global_shape,
            global_offset=scale_global_offset,
            axis_fragmentations=axis_fragmentations,
            replica_id=replica_id,
            prepend_axis_num=0,
        )

    def _direct_split_sharded_tensor(
        split_key: str,
        sub_sh_ten: ShardedTensor,
        split_factor: int,
        same_key_splits: bool,
    ) -> ShardedTensor:
        prepend = sub_sh_ten.prepend_axis_num
        local_shape = tuple(sub_sh_ten.local_shape)
        global_shape = list(sub_sh_ten.global_shape[prepend:])
        global_offset = list(sub_sh_ten.global_offset[prepend:])
        axis_fragmentations = list(sub_sh_ten.axis_fragmentations[prepend:])

        if same_key_splits:
            if global_shape[0] % split_factor != 0:
                raise ValueError(
                    f"Unexpected SwiGLU split shape for {split_key}: {global_shape[0]} not divisible by {split_factor}"
                )
            if axis_fragmentations[0] % split_factor != 0:
                raise ValueError(
                    f"Unexpected SwiGLU split fragmentation for {split_key}: "
                    f"{axis_fragmentations[0]} not divisible by {split_factor}"
                )
            split_global_out = global_shape[0] // split_factor
            global_shape[0] = split_global_out
            global_offset[0] %= split_global_out
            axis_fragmentations[0] //= split_factor

        return ShardedTensor(
            key=split_key,
            data=torch.empty(local_shape, dtype=sub_sh_ten.dtype, device="cpu"),
            dtype=sub_sh_ten.dtype,
            local_shape=local_shape,
            global_shape=tuple(global_shape),
            global_offset=tuple(global_offset),
            axis_fragmentations=tuple(axis_fragmentations),
            replica_id=sub_sh_ten.replica_id,
            prepend_axis_num=0,
        )

    new_sd: Dict[str, Any] = {}
    processed_keys: set[str] = set()
    for key, value in sharded_state_dict.items():
        canonical_key = _canonicalize_expert_key_for_checkpoint(key)

        if _GROUPED_EXPERT_BASE_WEIGHT_RE.match(canonical_key):
            processed_keys.add(canonical_key)
            continue

        if _is_extra_state_entry(key, value):
            processed_keys.add(canonical_key)
            continue

        match = _GROUPED_EXPERT_WEIGHT_RE.match(canonical_key)
        if match is None:
            if _DIRECT_FP8_LINEAR_WEIGHT_RE.match(canonical_key) and isinstance(
                value, (ShardedTensor, ShardedTensorFactory)
            ):
                continue
            new_sd[canonical_key] = value
            processed_keys.add(canonical_key)
            continue

        if isinstance(value, ShardedTensorFactory):
            sh_ten, sub_sh_tens = _split_factory(value)
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
                if axis_fragmentations_override[out_axis] % split_factor != 0:
                    raise ValueError(
                        f"Unexpected SwiGLU factory fragmentation for {key}: "
                        f"{axis_fragmentations_override[out_axis]} not divisible by {split_factor}"
                    )
                axis_fragmentations_override[out_axis] //= split_factor
                fused_global_out = sh_ten.global_shape[out_axis]
            else:
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
            new_sd[canonical_key] = value
            processed_keys.add(canonical_key)
            continue

        prepend = sh_ten.prepend_axis_num
        local_out = local_out_override if local_out_override is not None else sh_ten.local_shape[-2]
        global_out = global_out_override if global_out_override is not None else sh_ten.global_shape[-2]
        global_in = sh_ten.global_shape[-1]
        out_offset = out_offset_override if out_offset_override is not None else sh_ten.global_offset[prepend]
        in_offset = sh_ten.global_offset[prepend + 1] if len(sh_ten.global_offset) > prepend + 1 else 0

        weight_axis_fragmentations = axis_fragmentations_override[prepend:]
        expert_global_idx = sh_ten.global_offset[prepend - 1] if prepend > 0 else int(match.group(2))
        ckpt_key = re.sub(r"weight\d+$", f"weight{expert_global_idx}", canonical_key)

        weight_local_shape = (local_out, local_in)
        weight_global_shape = (global_out, global_in)
        weight_global_offset = (out_offset, in_offset)

        if isinstance(value, ShardedTensorFactory) and ".linear_fc1." in canonical_key:
            same_key_splits = all(sub.key == value.key for sub in sub_sh_tens[:2])
            for suffix, sub_sh_ten in zip(("w", "v"), sub_sh_tens[:2]):
                split_key = f"{canonical_key}_{suffix}"  # suffix from FP8_SWIGLU_HALF_SUFFIXES
                new_sd[split_key] = _direct_split_sharded_tensor(
                    f"{ckpt_key}_{suffix}",
                    sub_sh_ten,
                    split_factor,
                    same_key_splits,
                )
            _add_scale_entry(
                new_sd,
                fp8_entry_names(canonical_key)["scale_inv"],
                sh_ten,
                weight_local_shape,
                weight_global_shape,
                weight_global_offset,
                weight_axis_fragmentations,
                checkpoint_key=fp8_entry_names(ckpt_key)["scale_inv"],
            )
            processed_keys.add(canonical_key)
            continue

        new_sd[canonical_key] = ShardedTensor(
            key=ckpt_key,
            data=torch.empty(weight_local_shape, dtype=sh_ten.dtype, device="cpu"),
            dtype=sh_ten.dtype,
            local_shape=weight_local_shape,
            global_shape=weight_global_shape,
            global_offset=weight_global_offset,
            axis_fragmentations=weight_axis_fragmentations,
            replica_id=sh_ten.replica_id,
            prepend_axis_num=0,
        )

        _add_scale_entry(
            new_sd,
            fp8_entry_names(canonical_key)["scale_inv"],
            sh_ten,
            weight_local_shape,
            weight_global_shape,
            weight_global_offset,
            weight_axis_fragmentations,
            checkpoint_key=fp8_entry_names(ckpt_key)["scale_inv"],
        )
        processed_keys.add(canonical_key)

    for key, value in sharded_state_dict.items():
        canonical_key = _canonicalize_expert_key_for_checkpoint(key)
        if canonical_key in processed_keys:
            continue

        match = _DIRECT_FP8_LINEAR_WEIGHT_RE.match(canonical_key)
        if match is None or not isinstance(value, (ShardedTensor, ShardedTensorFactory)):
            continue

        if isinstance(value, ShardedTensorFactory) and canonical_key.endswith(".linear_fc1.weight"):
            sh_ten, sub_sh_tens = _split_factory(value)
            if len(sub_sh_tens) >= 2:
                split_local_out = sh_ten.local_shape[-2]
                fused_local_out = value.data.shape[0]
                if split_local_out == 0 or fused_local_out % split_local_out != 0:
                    raise ValueError(
                        f"Unexpected SwiGLU factory shapes for {key}: "
                        f"fused local out={fused_local_out}, split local out={split_local_out}"
                    )
                split_factor = fused_local_out // split_local_out
                same_key_splits = all(sub.key == value.key for sub in sub_sh_tens[:2])
                for suffix, sub_sh_ten in zip(("w", "v"), sub_sh_tens[:2]):
                    split_key = f"{canonical_key}_{suffix}"  # suffix from FP8_SWIGLU_HALF_SUFFIXES
                    new_sd[split_key] = _direct_split_sharded_tensor(
                        split_key,
                        sub_sh_ten,
                        split_factor,
                        same_key_splits,
                    )
                fused_local_shape = tuple(value.data.shape)
                prepend = sh_ten.prepend_axis_num
                rank_offset = sh_ten.global_offset[-2] // split_local_out
                fused_out_offset = rank_offset * fused_local_out
                fused_axis_fragmentations = list(sh_ten.axis_fragmentations[prepend:])
                if same_key_splits:
                    fused_global_out = sh_ten.global_shape[-2]
                    if fused_axis_fragmentations[-2] % split_factor != 0:
                        raise ValueError(
                            f"Unexpected SwiGLU factory fragmentation for {key}: "
                            f"{fused_axis_fragmentations[-2]} not divisible by {split_factor}"
                        )
                    fused_axis_fragmentations[-2] //= split_factor
                else:
                    fused_global_out = sh_ten.global_shape[-2] * split_factor
                fused_axis_fragmentations = tuple(fused_axis_fragmentations)
                fused_global_shape = tuple(sh_ten.global_shape[prepend:-2]) + (
                    fused_global_out,
                    sh_ten.global_shape[-1],
                )
                fused_global_offset = tuple(sh_ten.global_offset[prepend:-2]) + (
                    fused_out_offset,
                    sh_ten.global_offset[-1],
                )
                _add_scale_entry(
                    new_sd,
                    fp8_entry_names(canonical_key)["scale_inv"],
                    sh_ten,
                    fused_local_shape,
                    fused_global_shape,
                    fused_global_offset,
                    fused_axis_fragmentations,
                )
                processed_keys.add(canonical_key)
                continue

        sh_ten = value if isinstance(value, ShardedTensor) else _split_factory(value)[0]
        prepend = sh_ten.prepend_axis_num
        weight_axis_fragmentations = sh_ten.axis_fragmentations[prepend:]
        weight_local_shape = tuple(sh_ten.local_shape)
        weight_global_shape = tuple(sh_ten.global_shape[prepend:])
        weight_global_offset = tuple(sh_ten.global_offset[prepend:])
        new_sd[canonical_key] = ShardedTensor(
            key=fp8_entry_names(canonical_key, swiglu=True)["weight_w"],
            data=torch.empty(weight_local_shape, dtype=sh_ten.dtype, device="cpu"),
            dtype=sh_ten.dtype,
            local_shape=weight_local_shape,
            global_shape=weight_global_shape,
            global_offset=weight_global_offset,
            axis_fragmentations=weight_axis_fragmentations,
            replica_id=sh_ten.replica_id,
            prepend_axis_num=0,
        )
        _add_scale_entry(
            new_sd,
            fp8_entry_names(canonical_key)["scale_inv"],
            sh_ten,
            weight_local_shape,
            weight_global_shape,
            weight_global_offset,
            weight_axis_fragmentations,
        )
        processed_keys.add(canonical_key)

    for key, value in sharded_state_dict.items():
        canonical_key = _canonicalize_expert_key_for_checkpoint(key)
        if canonical_key in processed_keys or canonical_key in new_sd:
            continue
        new_sd[canonical_key] = value

    return new_sd


def register_fp8_scale_inv_buffers_after_load(
    model: torch.nn.Module,
    loaded_state_dict: Dict[str, Any],
) -> int:
    """Prepare a direct FP8 state dict before ``load_state_dict``.

    Direct FP8 checkpoints carry explicit ``*_scale_inv`` tensors, so TE
    ``_extra_state`` payloads are stale metadata for this path and are dropped.
    """

    registered = 0
    dense_split_weights: Dict[str, Dict[str, torch.Tensor]] = {}
    grouped_split_weights: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = {}

    for key in list(loaded_state_dict.keys()):
        if isinstance(key, str) and "_extra_state" in key:
            loaded_state_dict.pop(key, None)

    for key, value in loaded_state_dict.items():
        match = _GROUPED_EXPERT_SCALE_INV_RE.match(key)
        dense_match = _DIRECT_FP8_SCALE_INV_RE.match(key)
        grouped_split_match = _GROUPED_EXPERT_SPLIT_WEIGHT_RE.match(key)
        split_match = _DIRECT_FP8_SPLIT_WEIGHT_RE.match(key)

        if grouped_split_match is not None:
            grouped_split_weights.setdefault(
                (grouped_split_match.group(1), grouped_split_match.group(2)),
                {},
            )[grouped_split_match.group(3)] = _loaded_tensor_payload(value)
            continue

        if split_match is not None:
            dense_split_weights.setdefault(split_match.group(1), {})[split_match.group(2)] = _loaded_tensor_payload(
                value
            )
            continue

        if match is None and dense_match is None:
            continue

        module = model
        module_path = match.group(1) if match is not None else dense_match.group(1)
        for attr in module_path.split("."):
            module = getattr(module, attr)

        buffer_name = (
            fp8_entry_names(f"weight{match.group(2)}")["scale_inv"]
            if match is not None
            else fp8_entry_names("weight")["scale_inv"]
        )
        payload = _loaded_tensor_payload(value).to(dtype=torch.float32)
        weight_name = f"weight{match.group(2)}" if match is not None else "weight"
        weight = getattr(module, weight_name, None)
        target_device = payload.device
        if isinstance(weight, torch.Tensor) and weight.device.type != "meta":
            target_device = weight.device
        payload = payload.to(target_device)

        if buffer_name in module._buffers:
            module._buffers[buffer_name] = payload
        else:
            module.register_buffer(buffer_name, payload, persistent=True)
        registered += 1

    for (module_path, weight_idx), parts in grouped_split_weights.items():
        if not {"w", "v"}.issubset(parts):
            continue
        loaded_state_dict[f"{module_path}.weight{weight_idx}"] = torch.cat([parts["w"], parts["v"]], dim=0)
        loaded_state_dict.pop(fp8_entry_names(f"{module_path}.weight{weight_idx}", swiglu=True)["weight_w"], None)
        loaded_state_dict.pop(fp8_entry_names(f"{module_path}.weight{weight_idx}", swiglu=True)["weight_v"], None)

    for module_path, parts in dense_split_weights.items():
        if not {"w", "v"}.issubset(parts):
            continue
        loaded_state_dict[f"{module_path}.weight"] = torch.cat([parts["w"], parts["v"]], dim=0)
        loaded_state_dict.pop(fp8_entry_names(f"{module_path}.weight", swiglu=True)["weight_w"], None)
        loaded_state_dict.pop(fp8_entry_names(f"{module_path}.weight", swiglu=True)["weight_v"], None)

    return registered


# ------------------------------------------------------------------ #
# Dequantization
# ------------------------------------------------------------------ #


def dequant_fp8(
    w_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 weight: ``w = w_fp8 * scale_inv`` (block-wise or per-tensor).

    On CUDA with block-wise scales, dispatches to a fused triton kernel
    (``triton_oft.dequant_fp8.dequant_fp8_block_triton``) that avoids the
    full fp32 intermediate. Per-tensor (scalar scale) stays on the PyTorch
    path — it's already a single-pass op.

    Args:
        w_fp8: ``[out, in]`` in ``float8_e4m3fn``.
        scale_inv: ``[1]`` (per-tensor) or ``[out//B, in//B]`` (block-wise).
        out_dtype: Target dtype.
    """
    if scale_inv.numel() == 1:
        return (w_fp8.float() * scale_inv.float().item()).to(out_dtype)

    # Fused triton fast path (2-D block-wise FP8 on CUDA).
    if w_fp8.is_cuda and w_fp8.dim() == 2 and scale_inv.dim() == 2:
        try:
            from megatron.bridge.orbit.oft.triton_oft.dequant_fp8 import (
                dequant_fp8_block_triton,
            )

            return dequant_fp8_block_triton(w_fp8, scale_inv.to(torch.float32), out_dtype)
        except Exception:
            # Fall through to PyTorch reference on any kernel failure.
            pass

    out_feat, in_feat = w_fp8.shape
    sr, sc = scale_inv.shape
    bh, bw = out_feat // sr, in_feat // sc
    w = w_fp8.float().reshape(sr, bh, sc, bw)
    w = w * scale_inv.float().unsqueeze(1).unsqueeze(3)
    return w.reshape(out_feat, in_feat).to(out_dtype)


# ------------------------------------------------------------------ #
# Scale-inv merging (mirrors weight merge during checkpoint conversion)
# ------------------------------------------------------------------ #


def merge_qkv_scale_inv(config, q_s, k_s, v_s):
    """Merge Q/K/V ``weight_scale_inv`` following the Megatron QKV interleave.

    Same interleaving as ``merge_qkv_weights`` but on the reduced-resolution
    scale tensors.  Works because ``block_size`` evenly divides ``head_dim``.

    Args:
        config: ``TransformerConfig``-like with ``num_attention_heads``,
            ``num_query_groups``, ``kv_channels``, ``hidden_size``.
        q_s, k_s, v_s: Scale-inv tensors, shapes
            ``[q_heads * head_blocks, in_blocks]``, etc.

    Returns:
        Merged scale-inv of shape ``[merged_out_blocks, in_blocks]``.
    """
    head_num = config.num_attention_heads
    num_query_groups = config.num_query_groups
    heads_per_group = head_num // num_query_groups

    in_blocks = q_s.shape[-1]
    # Each head occupies head_size / block_size rows in the scale tensor.
    # Infer block_size from the scale shape: q has head_num * (head_size/B) rows.
    scale_rows_per_q_head = q_s.shape[0] // head_num  # head_size / block_size
    scale_rows_per_kv_head = k_s.shape[0] // num_query_groups

    q_r = q_s.reshape(head_num, scale_rows_per_q_head, in_blocks)
    k_r = k_s.reshape(num_query_groups, scale_rows_per_kv_head, in_blocks)
    v_r = v_s.reshape(num_query_groups, scale_rows_per_kv_head, in_blocks)

    parts = []
    for i in range(num_query_groups):
        parts.append(q_r[i * heads_per_group : (i + 1) * heads_per_group])
        parts.append(k_r[i : i + 1])
        parts.append(v_r[i : i + 1])
    return torch.cat(parts, dim=0).reshape(-1, in_blocks)


def merge_gated_mlp_scale_inv(gate_s, up_s):
    """Merge gate + up ``weight_scale_inv`` (simple cat along dim 0)."""
    return torch.cat([gate_s, up_s], dim=0)
