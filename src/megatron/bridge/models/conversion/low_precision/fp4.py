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

"""DeepSeek-style FP4/FP8 direct-save helpers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from megatron.bridge.models.conversion.low_precision.common import (
    add_tensor_entry,
    prepare_empty_model_state,
)
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
from megatron.bridge.peft.fp8_utils import merge_gated_mlp_scale_inv, merge_qkv_scale_inv

logger = logging.getLogger(__name__)


def _patch_safetensors_float8_e8m0() -> None:
    """Teach older safetensors builds about DeepSeek-style E8M0 FP8 scales."""
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is None:
        return
    try:
        import safetensors.torch as safetensors_torch
    except Exception:
        return

    dtype_table = getattr(safetensors_torch, "_TYPES", None)
    if isinstance(dtype_table, dict):
        dtype_table.setdefault("F8_E8M0", e8m0_dtype)


_patch_safetensors_float8_e8m0()

__all__ = [
    "build_fp4_direct_model_state_dict",
    "convert_hf_weight_for_direct_save",
]


@dataclass(frozen=True)
class _QuantBundle:
    weight: torch.Tensor
    scale: torch.Tensor
    kind: str


def _scale_key_candidates(weight_key: str) -> tuple[str, ...]:
    if not weight_key.endswith(".weight"):
        return ()
    module_key = weight_key[: -len(".weight")]
    return (
        f"{module_key}.scale",
        f"{weight_key}_scale",
        f"{weight_key}_scale_inv",
        f"{module_key}.weight_scale",
    )


def _load_scale(weight_key: str, hf_state_dict: Mapping[str, torch.Tensor]) -> torch.Tensor | None:
    for key in _scale_key_candidates(weight_key):
        if key in hf_state_dict:
            return hf_state_dict[key]
    return None


def _classify_quant_weight(weight: torch.Tensor, scale: torch.Tensor | None) -> str | None:
    if scale is None:
        return None
    if weight.dtype == torch.int8:
        return "fp4"
    if weight.dtype == torch.float8_e4m3fn:
        return "fp8"
    return None


def _load_quant_bundle(
    weight_key: str,
    hf_state_dict: Mapping[str, torch.Tensor],
) -> _QuantBundle | None:
    if not weight_key.endswith(".weight") or weight_key not in hf_state_dict:
        return None
    weight = hf_state_dict[weight_key]
    scale = _load_scale(weight_key, hf_state_dict)
    kind = _classify_quant_weight(weight, scale)
    if kind is None or scale is None:
        return None
    return _QuantBundle(weight=weight, scale=scale, kind=kind)


def _load_quant_bundles(
    hf_param: Any,
    hf_state_dict: Mapping[str, torch.Tensor],
) -> _QuantBundle | dict[str, _QuantBundle] | None:
    if isinstance(hf_param, str):
        return _load_quant_bundle(hf_param, hf_state_dict)
    if isinstance(hf_param, dict):
        bundles = {
            role: _load_quant_bundles(key, hf_state_dict)
            for role, key in hf_param.items()
        }
        if all(isinstance(value, _QuantBundle) for value in bundles.values()):
            kinds = {value.kind for value in bundles.values() if isinstance(value, _QuantBundle)}
            if len(kinds) == 1:
                return bundles  # type: ignore[return-value]
    return None


def _bundle_kind(bundles: _QuantBundle | dict[str, _QuantBundle]) -> str:
    if isinstance(bundles, _QuantBundle):
        return bundles.kind
    kinds = {bundle.kind for bundle in bundles.values()}
    if len(kinds) != 1:
        raise RuntimeError(f"Mixed quant bundle kinds in one mapping: {sorted(kinds)}")
    return next(iter(kinds))


def _bundle_weights(bundles: _QuantBundle | dict[str, _QuantBundle]) -> torch.Tensor | dict[str, torch.Tensor]:
    if isinstance(bundles, _QuantBundle):
        return bundles.weight
    return {role: bundle.weight for role, bundle in bundles.items()}


def _merge_qkv_matrices_preserve_width(
    provider: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Merge QKV rows while preserving packed/scale trailing dimensions."""
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError(f"Expected 2D QKV matrices, got {q.shape=}, {k.shape=}, {v.shape=}")

    trailing_width = q.shape[-1]
    if k.shape[-1] != trailing_width or v.shape[-1] != trailing_width:
        raise ValueError(f"QKV trailing widths differ: {q.shape=}, {k.shape=}, {v.shape=}")

    head_num = provider.num_attention_heads
    num_query_groups = provider.num_query_groups
    heads_per_group = head_num // num_query_groups
    head_size = provider.kv_channels or (provider.hidden_size // head_num)
    q_head_size = head_size * 2 if getattr(provider, "attention_output_gate", False) else head_size

    q_reshaped = q.reshape(head_num, q_head_size, trailing_width)
    k_reshaped = k.reshape(num_query_groups, head_size, trailing_width)
    v_reshaped = v.reshape(num_query_groups, head_size, trailing_width)
    if getattr(provider, "attention_output_gate", False):
        q_reshaped, z_reshaped = torch.chunk(q_reshaped, 2, dim=1)

    qkv_weights = []
    for i in range(num_query_groups):
        q_group = q_reshaped[i * heads_per_group : (i + 1) * heads_per_group]
        k_group = k_reshaped[i : i + 1]
        v_group = v_reshaped[i : i + 1]
        if getattr(provider, "attention_output_gate", False):
            z_group = z_reshaped[i * heads_per_group : (i + 1) * heads_per_group]
            qkv_weights.extend([q_group, z_group, k_group, v_group])
        else:
            qkv_weights.extend([q_group, k_group, v_group])

    return torch.cat(qkv_weights, dim=0).reshape(-1, trailing_width)


def _convert_fp4_scale_for_direct_save(task: Any, bundles: _QuantBundle | dict[str, _QuantBundle]) -> torch.Tensor:
    mapping = task.mapping

    if getattr(mapping, "tp_size", 1) != 1:
        raise ValueError(
            "Direct FP4 converter currently supports only single-rank TP=1 checkpoint writes."
        )

    if isinstance(
        mapping,
        (AutoMapping, DirectMapping, ColumnParallelMapping, RowParallelMapping, ReplicatedMapping),
    ):
        if isinstance(bundles, _QuantBundle):
            converted = bundles.scale
            if isinstance(mapping, AutoMapping) and mapping.permute_dims is not None:
                converted = torch.permute(converted, mapping.permute_dims).contiguous()
            return converted
        raise RuntimeError(f"Expected a single FP4 scale tensor for {task.param_name}")

    if isinstance(mapping, QKVMapping):
        if not isinstance(bundles, dict):
            raise RuntimeError(f"Expected QKV FP4 scale tensors for {task.param_name}")
        config = mapping._get_config(task.megatron_module)
        return _merge_qkv_matrices_preserve_width(
            config,
            bundles["q"].scale,
            bundles["k"].scale,
            bundles["v"].scale,
        ).contiguous()

    if isinstance(mapping, GatedMLPMapping):
        if not isinstance(bundles, dict):
            raise RuntimeError(f"Expected gated-MLP FP4 scale tensors for {task.param_name}")
        return torch.cat([bundles["gate"].scale, bundles["up"].scale], dim=0).contiguous()

    raise RuntimeError(f"Unsupported FP4 scale mapping for {task.param_name}: {type(mapping).__name__}")


def _convert_fp8_scale_for_direct_save(task: Any, bundles: _QuantBundle | dict[str, _QuantBundle]) -> torch.Tensor:
    mapping = task.mapping

    if getattr(mapping, "tp_size", 1) != 1:
        raise ValueError(
            "Direct FP4/FP8 converter currently supports only single-rank TP=1 checkpoint writes."
        )

    if isinstance(
        mapping,
        (AutoMapping, DirectMapping, ColumnParallelMapping, RowParallelMapping, ReplicatedMapping),
    ):
        if isinstance(bundles, _QuantBundle):
            converted = bundles.scale
            if isinstance(mapping, AutoMapping) and mapping.permute_dims is not None:
                converted = torch.permute(converted, mapping.permute_dims).contiguous()
            return converted
        raise RuntimeError(f"Expected a single FP8 scale tensor for {task.param_name}")

    if isinstance(mapping, QKVMapping):
        if not isinstance(bundles, dict):
            raise RuntimeError(f"Expected QKV FP8 scale tensors for {task.param_name}")
        config = mapping._get_config(task.megatron_module)
        return merge_qkv_scale_inv(
            config,
            bundles["q"].scale,
            bundles["k"].scale,
            bundles["v"].scale,
        ).contiguous()

    if isinstance(mapping, GatedMLPMapping):
        if not isinstance(bundles, dict):
            raise RuntimeError(f"Expected gated-MLP FP8 scale tensors for {task.param_name}")
        return merge_gated_mlp_scale_inv(bundles["gate"].scale, bundles["up"].scale).contiguous()

    raise RuntimeError(f"Unsupported FP8 scale mapping for {task.param_name}: {type(mapping).__name__}")


def convert_hf_weight_for_direct_save(task: Any, hf_weights: Any) -> torch.Tensor:
    """Convert one HF tensor payload into the direct-save Megatron representation."""
    mapping = task.mapping

    if getattr(mapping, "tp_size", 1) != 1:
        raise ValueError(
            "Direct FP4 converter currently supports only single-rank TP=1 checkpoint writes."
        )

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
        if hf_weights["q"].dtype == torch.int8:
            return _merge_qkv_matrices_preserve_width(
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


def build_fp4_direct_model_state_dict(
    bridge: Any,
    hf_pretrained: Any,
    meta_model: list[Any],
    model_template: dict[str, Any],
) -> dict[str, Any]:
    """Build a direct-save Megatron model state for DeepSeek-style FP4/FP8 HF checkpoints."""
    if len(meta_model) != 1:
        raise ValueError(
            "Direct FP4 converter currently supports a single Megatron model chunk (no VP stages)."
        )

    model_state = prepare_empty_model_state(model_template)
    conversion_tasks = bridge.build_conversion_tasks(hf_pretrained, meta_model)
    hf_state_dict: Mapping[str, torch.Tensor] = hf_pretrained.state

    num_regular = 0
    num_fp4 = 0
    num_fp8 = 0
    total_tasks = len(conversion_tasks)
    log_interval = max(1, total_tasks // 100)
    t_start = time.monotonic()

    for i, task in enumerate(conversion_tasks):
        if task is None or task.megatron_module is None:
            continue

        bundles = _load_quant_bundles(task.mapping.hf_param, hf_state_dict)
        if bundles is None:
            hf_weights = bridge.maybe_modify_loaded_hf_weight(
                task.mapping.hf_param,
                hf_state_dict,
            )
            converted = convert_hf_weight_for_direct_save(task, hf_weights)
            add_tensor_entry(model_state, task.param_name, converted, model_template.get(task.param_name))
            num_regular += 1
        else:
            kind = _bundle_kind(bundles)
            converted_weight = convert_hf_weight_for_direct_save(task, _bundle_weights(bundles))
            if kind == "fp4":
                converted_scale = _convert_fp4_scale_for_direct_save(task, bundles)
                num_fp4 += 1
            elif kind == "fp8":
                converted_scale = _convert_fp8_scale_for_direct_save(task, bundles)
                num_fp8 += 1
            else:
                raise RuntimeError(f"Unsupported quant bundle kind for {task.param_name}: {kind}")

            add_tensor_entry(model_state, task.param_name, converted_weight)
            add_tensor_entry(model_state, f"{task.param_name}_scale", converted_scale)

        if (i + 1) % log_interval == 0 or (i + 1) == total_tasks:
            elapsed = time.monotonic() - t_start
            eta = elapsed / (i + 1) * (total_tasks - i - 1) if i + 1 < total_tasks else 0.0
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
            logger.info(
                "  [%s/%s] %s regular, %s FP4, %s FP8 | elapsed %s ETA %s | %s",
                i + 1,
                total_tasks,
                num_regular,
                num_fp4,
                num_fp8,
                elapsed_str,
                eta_str,
                task.param_name,
            )

    logger.info(
        "Prepared direct checkpoint state dict: %s regular tensors, %s FP4 tensors, %s FP8 tensors",
        num_regular,
        num_fp4,
        num_fp8,
    )
    return model_state
