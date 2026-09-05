# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""INT4 conversion-general helpers."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import torch

from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    DirectMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
    RowParallelMapping,
    merge_qkv_biases,
    merge_qkv_weights,
)
from megatron.bridge.orbit.low_precision.common import (
    TensorSpillManager,
    _validate_single_rank_direct_conversion_tasks,
    add_tensor_entry,
    prepare_empty_model_state,
)
from megatron.bridge.orbit.quant.int4_utils import (
    INT4_PACKED_SUFFIX,
    INT4_SCALE_SUFFIX,
    INT4_SHAPE_SUFFIX,
)
from megatron.bridge.orbit.quantized_geometry import (
    reconstruct_swiglu_factory_geometry,
    resolve_dense_layer_index,
    rewrite_dense_layer_key,
    validate_quantized_shard_geometry,
)


logger = logging.getLogger(__name__)

__all__ = [
    "build_int4_direct_model_state_dict",
    "convert_hf_weight_for_direct_save",
    "dequantize_int4",
    "hf_param_uses_int4",
    "quantize_to_int4",
    "requantize_int4_with_scales",
    "register_int4_buffers_after_load_dense",
    "transform_sharded_state_dict_for_int4_dense",
]


@dataclass(frozen=True)
class _Int4Triplet:
    packed: torch.Tensor
    scale: torch.Tensor
    shape: torch.Tensor


def _validate_int4_triplet(triplet: _Int4Triplet, *, group_size: int, key: str) -> None:
    """Validate one packed INT4 tensor family against its declared logical grid."""

    def fail(detail: str) -> None:
        raise ValueError(f"Invalid INT4 triplet for {key!r}: {detail}")

    if group_size <= 0:
        fail(f"group_size must be positive, got group_size={group_size}")
    if triplet.packed.dtype != torch.int32:
        fail(f"weight_packed dtype must be torch.int32, got {triplet.packed.dtype}")
    if not triplet.scale.dtype.is_floating_point:
        fail(f"weight_scale must be floating, got {triplet.scale.dtype}")
    if not bool(torch.all(torch.isfinite(triplet.scale) & (triplet.scale > 0)).item()):
        fail("weight_scale must be finite and positive")
    if triplet.shape.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        fail(f"weight_shape dtype must be integral, got {triplet.shape.dtype}")
    if triplet.shape.dim() != 1 or triplet.shape.numel() != 2:
        fail(f"weight_shape must be a rank-1 tensor with two values, got {tuple(triplet.shape.shape)}")

    logical_dimensions: list[int] = []
    for name, raw_value in zip(("output rows", "input width"), triplet.shape.tolist()):
        try:
            value = int(raw_value)
        except (TypeError, ValueError, OverflowError):
            fail(f"weight_shape {name} must be a positive integer, got {raw_value!r}")
        if raw_value != value or value <= 0:
            fail(f"weight_shape {name} must be a positive integer, got {raw_value!r}")
        logical_dimensions.append(value)
    logical_rows, logical_width = logical_dimensions

    if triplet.packed.dim() != 2:
        fail(f"weight_packed must be rank 2, got shape {tuple(triplet.packed.shape)}")
    if triplet.scale.dim() != 2:
        fail(f"weight_scale must be rank 2, got shape {tuple(triplet.scale.shape)}")
    if logical_width % 8 != 0:
        fail(f"logical input width {logical_width} must be divisible by 8 for INT4 packing")
    if logical_width % group_size != 0:
        fail(f"logical input width {logical_width} must be divisible by group_size={group_size}")

    expected_packed_shape = (logical_rows, logical_width // 8)
    if tuple(triplet.packed.shape) != expected_packed_shape:
        fail(
            f"weight_packed shape {tuple(triplet.packed.shape)} does not match "
            f"logical weight_shape={(logical_rows, logical_width)}; expected {expected_packed_shape}"
        )

    expected_scale_shape = (logical_rows, logical_width // group_size)
    if tuple(triplet.scale.shape) != expected_scale_shape:
        fail(
            f"weight_scale shape {tuple(triplet.scale.shape)} does not match "
            f"logical weight_shape={(logical_rows, logical_width)} and group_size={group_size}; "
            f"expected {expected_scale_shape}"
        )


def _canonicalize_int4_triplet(
    triplet: _Int4Triplet,
    *,
    group_size: int,
    scale_dtype: torch.dtype,
    key: str,
) -> _Int4Triplet:
    """Return a triplet matching the strict DCP load schema exactly."""
    if scale_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"Invalid INT4 output scale dtype for {key!r}: expected torch.float16 or torch.bfloat16, got {scale_dtype}"
        )
    _validate_int4_triplet(triplet, group_size=group_size, key=key)

    logical_dimensions = [int(value) for value in triplet.shape.tolist()]
    int32_max = torch.iinfo(torch.int32).max
    if any(value > int32_max for value in logical_dimensions):
        raise ValueError(
            f"Invalid INT4 triplet for {key!r}: weight_shape values must fit torch.int32, got {logical_dimensions}"
        )

    canonical_scale = triplet.scale.to(dtype=scale_dtype)
    if not bool(torch.all(torch.isfinite(canonical_scale) & (canonical_scale > 0)).item()):
        raise ValueError(
            f"Invalid INT4 triplet for {key!r}: weight_scale cannot be represented as "
            f"finite positive {scale_dtype} values"
        )
    return _Int4Triplet(
        packed=triplet.packed.contiguous(),
        scale=canonical_scale.contiguous(),
        shape=triplet.shape.to(dtype=torch.int32).contiguous(),
    )


def _validate_hf_int4_triplets(
    hf_param: Any,
    hf_triplets: Any,
    *,
    group_size: int,
    fallback_key: str,
) -> None:
    """Validate source triplets individually, including QKV and gated inputs."""
    if isinstance(hf_triplets, _Int4Triplet):
        key = hf_param if isinstance(hf_param, str) else fallback_key
        _validate_int4_triplet(hf_triplets, group_size=group_size, key=key)
        return
    if isinstance(hf_triplets, dict):
        for role, triplet in hf_triplets.items():
            source_param = hf_param.get(role) if isinstance(hf_param, dict) else None
            source_key = source_param if isinstance(source_param, str) else f"{fallback_key}[{role}]"
            _validate_hf_int4_triplets(
                source_param,
                triplet,
                group_size=group_size,
                fallback_key=source_key,
            )


def dequantize_int4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_shape: torch.Tensor,
    group_size: int | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Dequantize INT4 packed weights to bfloat16 with the Triton CUDA path."""
    target_device = weight_packed.device if device is None else torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("dequantize_int4 requires a CUDA target device")
    if group_size is None:
        group_size = _infer_int4_group_size(weight_packed, weight_scale)

    from megatron.bridge.orbit.oft.triton_oft.int4_dequant import dequantize_int4_triton

    return dequantize_int4_triton(
        weight_packed.to(target_device),
        weight_scale.to(target_device),
        weight_shape,
        group_size=group_size,
        out_dtype=torch.bfloat16,
    )


def _infer_int4_group_size(weight_packed: torch.Tensor, weight_scale: torch.Tensor) -> int:
    """Infer per-row INT4 group size from packed width and scale groups."""
    if weight_packed.dim() != 2:
        raise ValueError(f"expected 2-D weight_packed, got {tuple(weight_packed.shape)}")
    out_features, packed_in = weight_packed.shape
    in_features = packed_in * 8
    if out_features <= 0:
        raise ValueError(f"out_features must be positive, got {out_features}")
    if weight_scale.numel() % out_features != 0:
        raise ValueError(
            f"weight_scale with {weight_scale.numel()} values cannot be grouped by out_features={out_features}"
        )
    scale_groups = weight_scale.numel() // out_features
    if scale_groups <= 0:
        raise ValueError(f"scale_groups must be positive, got {scale_groups}")
    if in_features % scale_groups != 0:
        raise ValueError(f"in_features={in_features} is not divisible by scale_groups={scale_groups}")
    return in_features // scale_groups


def quantize_to_int4(
    weight: torch.Tensor,
    group_size: int = 32,
    scale_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize bfloat16/float16 weights to INT4 packed format."""
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2-D, got shape {tuple(weight.shape)}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(f"in_features={in_features} must be divisible by group_size={group_size}")
    if in_features % 8 != 0:
        raise ValueError(f"in_features must be divisible by 8, got {in_features}")
    weight_shape = torch.tensor([out_features, in_features], dtype=torch.int32)

    w = weight.float()
    num_groups = in_features // group_size
    w_grouped = w.view(out_features, num_groups, -1)

    group_max = w_grouped.abs().amax(dim=-1)
    scale = group_max / 7.0
    scale = scale.clamp(min=1e-10)

    scale_expanded = scale.unsqueeze(-1).expand_as(w_grouped)
    w_q = (w_grouped / scale_expanded).round().clamp(-8, 7)
    w_q = w_q.view(out_features, -1)[:, :in_features]
    w_q = (w_q + 8).to(torch.uint8)

    w_q_grouped = w_q.view(out_features, in_features // 8, 8).to(torch.int32)
    packed = torch.zeros(
        out_features,
        in_features // 8,
        dtype=torch.int32,
        device=weight.device,
    )
    for i in range(8):
        packed |= (w_q_grouped[:, :, i] & 0xF) << (i * 4)

    return packed, scale.to(scale_dtype), weight_shape


def requantize_int4_with_scales(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Re-quantize a dequantized weight onto its ORIGINAL per-group grid.

    ``quantize_to_int4`` derives fresh scales as ``amax / 7`` and therefore can
    never emit ``-8`` — a valid INT4 value that real compressed-tensors
    checkpoints use (30% of groups in the nm-testing Llama W4A16 model carry a
    ``-8`` group minimum). Re-quantizing such data with recomputed scales
    re-grids those groups. This variant reuses the source ``scale`` tensor, so
    values that came from ``dequantize_int4(packed, scale, ...)`` round back to
    their original integers bitwise (the BF16 transit error is far below half
    a grid step).

    Args:
        weight: Dequantized values, ``[out_features, in_features]``.
        scale: The ORIGINAL per-group scales, ``[out_features, num_groups]``
            (rows aligned with ``weight`` rows).

    Returns:
        ``(weight_packed, weight_scale, weight_shape)`` in the same packed
        layout ``quantize_to_int4`` produces; ``weight_scale`` is the input
        ``scale`` unchanged.
    """
    out_features, in_features = weight.shape
    if scale.dim() != 2 or scale.shape[0] != out_features or in_features % scale.shape[1] != 0:
        raise ValueError(f"scale shape {tuple(scale.shape)} does not tile weight shape {tuple(weight.shape)}")
    if in_features % 8 != 0:
        raise ValueError(f"in_features must be divisible by 8, got {in_features}")
    num_groups = scale.shape[1]
    group_size = in_features // num_groups
    weight_shape = torch.tensor([out_features, in_features], dtype=torch.int32)

    w_grouped = weight.float().view(out_features, num_groups, group_size)
    s = scale.float().clamp(min=1e-10).unsqueeze(-1)
    w_q = torch.round(w_grouped / s).clamp(-8, 7).view(out_features, -1)
    w_q = (w_q + 8).to(torch.uint8)

    w_q_grouped = w_q.view(out_features, in_features // 8, 8).to(torch.int32)
    packed = torch.zeros(out_features, in_features // 8, dtype=torch.int32, device=weight.device)
    for i in range(8):
        packed |= (w_q_grouped[:, :, i] & 0xF) << (i * 4)

    return packed, scale, weight_shape


def hf_weight_has_int4_triplet(weight_key: str, hf_state_dict: Mapping[str, torch.Tensor]) -> bool:
    if not weight_key.endswith(".weight"):
        return False
    return any(
        f"{weight_key}{suffix}" in hf_state_dict
        for suffix in (INT4_PACKED_SUFFIX, INT4_SCALE_SUFFIX, INT4_SHAPE_SUFFIX)
    )


def hf_param_uses_int4(hf_param: Any, hf_state_dict: Mapping[str, torch.Tensor]) -> bool:
    """Return True if an HF param name, or any name nested in a dict, is INT4-packed."""
    if isinstance(hf_param, str):
        return hf_weight_has_int4_triplet(hf_param, hf_state_dict)
    if isinstance(hf_param, dict):
        return any(hf_param_uses_int4(value, hf_state_dict) for value in hf_param.values())
    return False


def convert_hf_weight_for_direct_save(task: Any, hf_weights: Any) -> torch.Tensor:
    """HF -> Megatron conversion for the TP=1 direct-save path."""
    mapping = task.mapping

    if getattr(mapping, "tp_size", 1) != 1:
        raise ValueError("Direct INT4 converter currently supports only single-rank TP=1 checkpoint writes.")

    if isinstance(mapping, AutoMapping):
        converted = hf_weights
        if mapping.permute_dims is not None:
            converted = torch.permute(converted, mapping.permute_dims).contiguous()
        return converted

    if isinstance(
        mapping,
        (DirectMapping, ColumnParallelMapping, RowParallelMapping, ReplicatedMapping),
    ):
        return hf_weights

    if isinstance(mapping, QKVMapping):
        config = mapping._get_config(task.megatron_module)
        if hf_weights["q"].ndim == 1:
            return merge_qkv_biases(
                config,
                hf_weights["q"],
                hf_weights["k"],
                hf_weights["v"],
            )
        return merge_qkv_weights(
            config,
            hf_weights["q"],
            hf_weights["k"],
            hf_weights["v"],
        )

    if isinstance(mapping, GatedMLPMapping):
        return torch.cat([hf_weights["gate"], hf_weights["up"]], dim=0)

    converted = mapping.hf_to_megatron(hf_weights, task.megatron_module)
    if converted is None:
        raise RuntimeError(f"Conversion returned None for {task.param_name}")
    if converted.device.type == "meta":
        raise RuntimeError(
            f"Conversion unexpectedly produced a meta tensor for {task.param_name}. "
            "Add an explicit direct-save handler for this mapping type."
        )
    return converted


def _load_hf_int4_triplet(weight_key: str, hf_state_dict: Mapping[str, torch.Tensor]) -> _Int4Triplet | None:
    if not weight_key.endswith(".weight"):
        return None
    base = weight_key[: -len(".weight")]
    packed_key = base + ".weight_packed"
    scale_key = base + ".weight_scale"
    shape_key = base + ".weight_shape"
    family_keys = (packed_key, scale_key, shape_key)
    present = [key in hf_state_dict for key in family_keys]
    if not any(present):
        return None
    if not all(present):
        missing = ", ".join(key for key, is_present in zip(family_keys, present) if not is_present)
        raise ValueError(f"Incomplete INT4 triplet for {weight_key!r}; missing: {missing}")
    return _Int4Triplet(
        packed=hf_state_dict[packed_key],
        scale=hf_state_dict[scale_key],
        shape=hf_state_dict[shape_key],
    )


def _load_hf_int4_triplets(
    hf_param: Any,
    hf_state_dict: Mapping[str, torch.Tensor],
) -> _Int4Triplet | dict[str, _Int4Triplet] | None:
    if isinstance(hf_param, str):
        return _load_hf_int4_triplet(hf_param, hf_state_dict)
    if isinstance(hf_param, dict):
        triplets = {role: _load_hf_int4_triplets(key, hf_state_dict) for role, key in hf_param.items()}
        if all(isinstance(value, _Int4Triplet) for value in triplets.values()):
            return triplets  # type: ignore[return-value]
    return None


def _make_int4_shape_like(template: torch.Tensor, out_features: int) -> torch.Tensor:
    return torch.tensor(
        [out_features, int(template[1].item())],
        dtype=template.dtype,
        device=template.device,
    )


def _merge_qkv_rows_for_int4(
    provider: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Merge row-aligned INT4 packed/scale tensors using Megatron's QKV order."""
    feature_dim = q.shape[1]
    head_num = provider.num_attention_heads
    num_query_groups = provider.num_query_groups
    heads_per_group = head_num // num_query_groups
    head_size = provider.kv_channels or (provider.hidden_size // head_num)
    q_head_size = head_size * 2 if getattr(provider, "attention_output_gate", False) else head_size

    q_reshaped = q.view(head_num, q_head_size, feature_dim)
    k_reshaped = k.view(num_query_groups, head_size, feature_dim)
    v_reshaped = v.view(num_query_groups, head_size, feature_dim)
    if getattr(provider, "attention_output_gate", False):
        q_reshaped, z_reshaped = torch.chunk(q_reshaped, 2, dim=1)

    qkv_rows = []
    for i in range(num_query_groups):
        q_group = q_reshaped[i * heads_per_group : (i + 1) * heads_per_group]
        k_group = k_reshaped[i : i + 1]
        v_group = v_reshaped[i : i + 1]
        if getattr(provider, "attention_output_gate", False):
            z_group = z_reshaped[i * heads_per_group : (i + 1) * heads_per_group]
            qkv_rows.extend([q_group, z_group, k_group, v_group])
        else:
            qkv_rows.extend([q_group, k_group, v_group])

    return torch.cat(qkv_rows, dim=0).reshape(-1, feature_dim)


def _convert_hf_int4_triplet_for_direct_save(task: Any, hf_triplets: Any) -> _Int4Triplet | None:
    """Convert HF INT4 triplets without dequantizing/re-quantizing."""
    mapping = task.mapping

    if getattr(mapping, "tp_size", 1) != 1:
        raise ValueError("Direct INT4 converter currently supports only single-rank TP=1 checkpoint writes.")

    if isinstance(
        mapping,
        (AutoMapping, DirectMapping, ColumnParallelMapping, RowParallelMapping, ReplicatedMapping),
    ):
        if isinstance(mapping, AutoMapping) and getattr(mapping, "permute_dims", None) is not None:
            return None
        if isinstance(hf_triplets, _Int4Triplet):
            return hf_triplets
        return None

    if isinstance(mapping, QKVMapping):
        if not isinstance(hf_triplets, dict):
            return None
        q = hf_triplets.get("q")
        k = hf_triplets.get("k")
        v = hf_triplets.get("v")
        if not all(isinstance(value, _Int4Triplet) for value in (q, k, v)):
            return None
        config = mapping._get_config(task.megatron_module)
        packed = _merge_qkv_rows_for_int4(config, q.packed, k.packed, v.packed)
        scale = _merge_qkv_rows_for_int4(config, q.scale, k.scale, v.scale)
        return _Int4Triplet(
            packed=packed.contiguous(),
            scale=scale.contiguous(),
            shape=_make_int4_shape_like(q.shape, packed.shape[0]),
        )

    if isinstance(mapping, GatedMLPMapping):
        if not isinstance(hf_triplets, dict):
            return None
        gate = hf_triplets.get("gate")
        up = hf_triplets.get("up")
        if not isinstance(gate, _Int4Triplet) or not isinstance(up, _Int4Triplet):
            return None
        packed = torch.cat([gate.packed, up.packed], dim=0).contiguous()
        scale = torch.cat([gate.scale, up.scale], dim=0).contiguous()
        return _Int4Triplet(
            packed=packed,
            scale=scale,
            shape=_make_int4_shape_like(gate.shape, packed.shape[0]),
        )

    return None


def build_int4_direct_model_state_dict(
    int4_bridge: Any,
    hf_pretrained: Any,
    meta_model: list[Any],
    model_template: dict[str, Any],
    *,
    group_size: int,
    scale_dtype: torch.dtype,
    spill_manager: TensorSpillManager | None = None,
) -> dict[str, Any]:
    """Create the prebuilt ``state_dict['model']`` for direct INT4 checkpoint save."""
    if len(meta_model) != 1:
        raise ValueError("Direct INT4 converter currently supports a single Megatron model chunk (no VP stages).")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got group_size={group_size}")

    conversion_tasks = int4_bridge.build_conversion_tasks(hf_pretrained, meta_model)
    _validate_single_rank_direct_conversion_tasks(conversion_tasks, format_name="INT4")

    model_state = prepare_empty_model_state(model_template)
    hf_state_dict: Mapping[str, torch.Tensor] = hf_pretrained.state

    num_regular = 0
    num_int4 = 0
    total_tasks = len(conversion_tasks)
    log_interval = max(1, total_tasks // 100)
    t_start = time.monotonic()

    for i, task in enumerate(conversion_tasks):
        if hf_param_uses_int4(task.mapping.hf_param, hf_state_dict):
            hf_triplets = _load_hf_int4_triplets(task.mapping.hf_param, hf_state_dict)
            _validate_hf_int4_triplets(
                task.mapping.hf_param,
                hf_triplets,
                group_size=group_size,
                fallback_key=task.param_name,
            )
            converted_triplet = _convert_hf_int4_triplet_for_direct_save(task, hf_triplets)
            if converted_triplet is None:
                raise RuntimeError(
                    "Cannot preserve INT4 triplets without dequantizing: "
                    f"param={task.param_name} "
                    f"mapping={type(task.mapping).__name__} "
                    f"hf_param={task.mapping.hf_param!r}. "
                    "Add a packed/scale/shape-preserving handler for this mapping."
                )
            converted_triplet = _canonicalize_int4_triplet(
                converted_triplet,
                group_size=group_size,
                scale_dtype=scale_dtype,
                key=task.param_name,
            )

            packed = converted_triplet.packed
            scale = converted_triplet.scale
            shape = converted_triplet.shape

            add_tensor_entry(
                model_state, f"{task.param_name}{INT4_PACKED_SUFFIX}", packed, spill_manager=spill_manager
            )
            add_tensor_entry(model_state, f"{task.param_name}{INT4_SCALE_SUFFIX}", scale, spill_manager=spill_manager)
            add_tensor_entry(model_state, f"{task.param_name}{INT4_SHAPE_SUFFIX}", shape, spill_manager=spill_manager)
            num_int4 += 1
        else:
            hf_weights = int4_bridge.maybe_modify_loaded_hf_weight(
                task.mapping.hf_param,
                hf_state_dict,
            )
            converted = convert_hf_weight_for_direct_save(task, hf_weights)
            add_tensor_entry(model_state, task.param_name, converted, spill_manager=spill_manager)
            num_regular += 1

        if (i + 1) % log_interval == 0 or (i + 1) == total_tasks:
            elapsed = time.monotonic() - t_start
            eta = elapsed / (i + 1) * (total_tasks - i - 1) if i + 1 < total_tasks else 0.0
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
            logger.info(
                "  [%s/%s] %s regular, %s INT4 | elapsed %s ETA %s | %s",
                i + 1,
                total_tasks,
                num_regular,
                num_int4,
                elapsed_str,
                eta_str,
                task.param_name,
            )

    logger.info(
        "Prepared direct checkpoint state dict: %s regular tensors, %s INT4 expert tensors",
        num_regular,
        num_int4,
    )
    if num_int4 == 0:
        raise RuntimeError("Direct INT4 conversion found no complete INT4 mappings in the source checkpoint")
    return model_state


# -------------------------------------------------------------------------- #
# Dense-model dist-checkpoint load helpers
# (Llama INT4 compressed-tensors / pack-quantized)
# -------------------------------------------------------------------------- #

# Dense Megatron linear weight keys: attention QKV / output proj + gated MLP
# fc1 / fc2. Matches the names produced by the Llama bridge's mapping registry
# for GPTModel (`decoder.layers.{i}.self_attention.linear_qkv.weight`, etc.).
_DENSE_LINEAR_WEIGHT_RE = re.compile(r"^(.*\.(?:linear_qkv|linear_proj|linear_fc1|linear_fc2|router))\.weight$")
_DENSE_LINEAR_TRIPLET_RE = re.compile(
    r"^(.*\.(?:linear_qkv|linear_proj|linear_fc1|linear_fc2|router))\.weight_(packed|scale|shape)$"
)
_EXPLICIT_LAYER_KEY_RE = re.compile(r"(^|\.)layers\.\d+\.")


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


def _replica_id_with_current_tp_rank(replica_id: Any) -> Any:
    """Mark replicated metadata tensors as TP replicas during distributed load."""

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return replica_id

    try:
        from megatron.core import parallel_state

        tp_rank = parallel_state.get_tensor_model_parallel_rank()
    except Exception:
        return replica_id

    if isinstance(replica_id, tuple) and len(replica_id) == 3:
        return (replica_id[0], tp_rank, replica_id[2])
    return replica_id


def transform_sharded_state_dict_for_int4_dense(
    sharded_state_dict: Dict[str, Any],
    group_size: int = 128,
    scale_dtype: "torch.dtype | None" = None,
) -> Dict[str, Any]:
    """Rewrite dense-linear BF16 weight entries as INT4 triplet entries.

    Parallels the expert-only
    ``megatron.bridge.orbit.quant.int4_utils.transform_sharded_state_dict_for_int4``
    but matches dense Megatron linear weights (``*.self_attention.linear_qkv.weight``,
    ``*.self_attention.linear_proj.weight``, ``*.mlp.linear_fc1.weight``,
    ``*.mlp.linear_fc2.weight``).

    The dist checkpoint produced by ``convert_int4_checkpoint_direct.py`` for
    compressed-tensors Llama stores every linear as ``{name}_packed`` +
    ``_scale`` + ``_shape``; the runtime model declares each linear as a BF16
    ``weight`` Parameter. This swaps the BF16 ``ShardedTensor`` for the three
    INT4 ``ShardedTensor`` entries so ``dist_checkpointing.load`` populates the
    triplets with real data.

    ``scale_dtype`` defaults to ``torch.bfloat16`` (matches the Red Hat / Neural
    Magic compressed-tensors recipe). Pass ``torch.float16`` for Kimi-style
    checkpoints.
    """
    from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory

    if scale_dtype is None:
        scale_dtype = torch.bfloat16
    if group_size <= 0:
        raise ValueError(f"INT4 group_size must be positive, got {group_size}")

    def _numel(shape: tuple[int, ...]) -> int:
        total = 1
        for dim in shape:
            total *= dim
        return total

    new_sd: Dict[str, Any] = {}
    replaced = 0
    cumulative_allocated_bytes = 0
    for key, value in sharded_state_dict.items():
        is_dense_linear = _DENSE_LINEAR_WEIGHT_RE.match(key) is not None and isinstance(
            value, (ShardedTensor, ShardedTensorFactory)
        )

        # Materialize meta tensors only for pass-through keys — for keys we're
        # about to replace with INT4 triplets, the BF16 payload is wasted RAM
        # (on 8B Llama that's ~13 GiB per rank of allocations we never use).
        if (
            not is_dense_linear
            and isinstance(value, ShardedTensor)
            and value.data is not None
            and value.data.device.type == "meta"
        ):
            value.data = torch.empty(value.local_shape, dtype=value.dtype, device="cpu")

        if not is_dense_linear:
            new_sd[key] = value
            continue

        if isinstance(value, ShardedTensorFactory):
            geometry = reconstruct_swiglu_factory_geometry(value, key=key)
            sh_ten = geometry.sharded_tensor
            local_out_override = geometry.local_out
            global_out_override = geometry.global_out
            out_offset_override = geometry.out_offset
            axis_fragmentations_override = geometry.axis_fragmentations
        else:
            sh_ten = value
            local_out_override = None
            global_out_override = None
            out_offset_override = None
            axis_fragmentations_override = sh_ten.axis_fragmentations
        # Homogeneous TransformerBlock checkpointing keeps one local 2-D
        # tensor per layer but prepends its global layer coordinate to DCP
        # metadata. The converter writes per-layer triplets, so that global
        # coordinate must replace the enclosing module's local layer index.
        prepend = sh_ten.prepend_axis_num
        global_layer_idx = resolve_dense_layer_index(sh_ten, key=key)
        local_out = local_out_override if local_out_override is not None else sh_ten.local_shape[-2]
        local_in = sh_ten.local_shape[-1]
        global_out = global_out_override if global_out_override is not None else sh_ten.global_shape[-2]
        global_in = sh_ten.global_shape[-1]

        weight_axis_fragmentations = tuple(axis_fragmentations_override[prepend:])
        out_offset_full = (
            out_offset_override
            if out_offset_override is not None
            else (sh_ten.global_offset[prepend] if len(sh_ten.global_offset) > prepend else 0)
        )
        in_offset_full = sh_ten.global_offset[prepend + 1] if len(sh_ten.global_offset) > prepend + 1 else 0
        validate_quantized_shard_geometry(
            key=key,
            local_shape=(local_out, local_in),
            global_shape=(global_out, global_in),
            global_offset=(out_offset_full, in_offset_full),
            axis_fragmentations=weight_axis_fragmentations,
            packing_factor=8,
            group_size=group_size,
        )

        packed_in = local_in // 8
        global_packed_in = global_in // 8
        num_groups = local_in // group_size
        global_num_groups = global_in // group_size

        ckpt_key = rewrite_dense_layer_key(key, global_layer_idx)
        triplets = [
            (
                INT4_PACKED_SUFFIX,
                (local_out, packed_in),
                (global_out, global_packed_in),
                (out_offset_full, in_offset_full // 8),
                torch.int32,
            ),
            (
                INT4_SCALE_SUFFIX,
                (local_out, num_groups),
                (global_out, global_num_groups),
                (out_offset_full, in_offset_full // group_size),
                scale_dtype,
            ),
            (INT4_SHAPE_SUFFIX, (2,), (2,), (0,), torch.int32),
        ]

        for suffix, local_sh, global_sh, off, dtype in triplets:
            alloc_bytes = _numel(local_sh) * torch.tensor(0, dtype=dtype).element_size()
            try:
                data = torch.empty(local_sh, dtype=dtype, device="cpu")
            except RuntimeError as exc:
                original_device = getattr(getattr(sh_ten, "data", None), "device", None)
                original_device = getattr(original_device, "type", original_device)
                raise RuntimeError(
                    "INT4 dense transform allocation failed: "
                    f"key={ckpt_key + suffix} "
                    f"original_key={key} "
                    f"local_shape={local_sh} "
                    f"global_shape={global_sh} "
                    f"dtype={dtype} "
                    f"alloc_bytes={alloc_bytes} "
                    f"cumulative_bytes_before={cumulative_allocated_bytes} "
                    f"cumulative_bytes_with_current={cumulative_allocated_bytes + alloc_bytes} "
                    f"original_dense_data_device={original_device}"
                ) from exc
            cumulative_allocated_bytes += alloc_bytes
            axis_frags = weight_axis_fragmentations if suffix != INT4_SHAPE_SUFFIX else (1,)
            new_sd[key + suffix] = ShardedTensor(
                key=ckpt_key + suffix,
                data=data,
                dtype=dtype,
                local_shape=local_sh,
                global_shape=global_sh,
                global_offset=off,
                axis_fragmentations=axis_frags,
                replica_id=(
                    _replica_id_with_current_tp_rank(sh_ten.replica_id)
                    if suffix == INT4_SHAPE_SUFFIX
                    else sh_ten.replica_id
                ),
                prepend_axis_num=0,
            )
        replaced += 1

    logger.info(
        "[INT4 dense transform] Replaced %s dense linear weight entries with INT4 triplets "
        "(group_size=%s, scale_dtype=%s)",
        replaced,
        group_size,
        scale_dtype,
    )
    return new_sd


def register_int4_buffers_after_load_dense(
    model: "torch.nn.Module",
    loaded_state_dict: Dict[str, Any],
) -> int:
    """Register loaded dense-linear INT4 triplets as buffers on modules.

    After ``dist_checkpointing.load`` fills the sharded state dict, this walks
    the triplet entries, registers them as persistent buffers on the matching
    linear modules, and empties the placeholder BF16 ``.weight`` Parameter.
    Caller should strip triplet entries from the state dict before calling
    ``model.load_state_dict``.
    """
    triplets: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, value in loaded_state_dict.items():
        m = _DENSE_LINEAR_TRIPLET_RE.match(key)
        if m is None:
            continue
        module_path = m.group(1)
        triplets.setdefault(module_path, {})[m.group(2)] = value

    registered = 0
    for module_path, parts in triplets.items():
        if len(parts) != 3:
            continue

        module = model
        for attr in module_path.split("."):
            module = getattr(module, attr)

        w = getattr(module, "weight", None)
        payloads = {suffix: _loaded_tensor_payload(parts[suffix]) for suffix in ("packed", "scale", "shape")}
        target_device = payloads["packed"].device
        if w is not None and w.device.type != "meta":
            target_device = w.device

        for suffix in ("packed", "scale", "shape"):
            module.register_buffer(
                f"weight_{suffix}",
                payloads[suffix].to(target_device),
                persistent=True,
            )

        if w is not None:
            w.data = _empty_storage_view(w.data)
        registered += 1

    logger.info("[INT4 dense register] Registered INT4 buffers for %s dense linears", registered)
    return registered
