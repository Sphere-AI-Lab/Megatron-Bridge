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

"""FP8 direct-save helpers."""

from __future__ import annotations

from copy import deepcopy
import time
from typing import Any, Iterable, Mapping

import torch

from megatron.bridge.orbit.low_precision.common import (
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
from megatron.bridge.orbit.quant.fp8_utils import merge_gated_mlp_scale_inv, merge_qkv_scale_inv

__all__ = [
    "apply_modelopt_fp8_to_meta_model",
    "build_fp8_direct_model_state_dict",
    "build_fp8_scale_inv_key",
    "build_merged_scale_inv_for_task",
    "convert_hf_weight_for_direct_save",
]


def apply_modelopt_fp8_to_meta_model(
    module: Any,
    module_names: Iterable[str] | None = None,
    *,
    compress_weights: bool = False,
) -> None:
    """Install FP8 blockwise ModelOpt quantizer modules on a meta Megatron module."""
    import modelopt.torch.quantization as mtq

    quant_cfg = deepcopy(mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG)
    if module_names is not None:
        quant_cfg["quant_cfg"] = quant_cfg.get("quant_cfg", {}).copy()
        enabled_weight_cfg = deepcopy(
            quant_cfg["quant_cfg"].get("*weight_quantizer", {"enable": True})
        )
        quant_cfg["quant_cfg"]["*weight_quantizer"] = {"enable": False}

        for module_name in sorted(set(module_names)):
            quant_cfg["quant_cfg"][f"{module_name}.weight_quantizer"] = deepcopy(
                enabled_weight_cfg
            )

    def _noop_forward_loop(_m):
        return None

    mtq.quantize(module, quant_cfg, _noop_forward_loop)
    if compress_weights:
        mtq.compress(module)


def build_fp8_scale_inv_key(param_name: str) -> str:
    """Build the sibling checkpoint key used for FP8 block scales."""
    return f"{param_name}_scale_inv"


def _resolve_modelopt_fp8_weight_key(
    param_name: str,
    converted: torch.Tensor,
    model_template: Mapping[str, Any],
) -> str:
    """Return the on-disk key expected by a restored ModelOpt-compressed module."""
    if converted.dtype != torch.float8_e4m3fn or not param_name.endswith(".weight"):
        return param_name

    modelopt_weight_key = f"{param_name}_w"
    return modelopt_weight_key


def _is_split_swiglu_fp8_weight(task: Any, converted: torch.Tensor) -> bool:
    """Return True for dense gated-MLP FC1 weights with Megatron split keying."""
    config = getattr(getattr(task, "megatron_module", None), "config", None)
    return (
        converted.dtype == torch.float8_e4m3fn
        and task.param_name.endswith(".mlp.linear_fc1.weight")
        and bool(getattr(config, "gated_linear_unit", False))
    )


def _add_fp8_weight_entries(
    model_state: dict[str, Any],
    task: Any,
    converted: torch.Tensor,
    model_template: Mapping[str, Any],
) -> tuple[int, int]:
    """Add converted weight tensors and return ``(weight_entries, renamed_entries)``."""
    if _is_split_swiglu_fp8_weight(task, converted):
        if converted.shape[0] % 2 != 0:
            raise ValueError(
                f"Cannot split SwiGLU FP8 weight with odd output dimension for {task.param_name}: "
                f"shape={tuple(converted.shape)}"
            )
        gate_weight, up_weight = torch.chunk(converted, 2, dim=0)
        weight_w_key = f"{task.param_name}_w"
        weight_v_key = f"{task.param_name}_v"
        add_tensor_entry(
            model_state,
            weight_w_key,
            gate_weight,
            model_template.get(weight_w_key),
        )
        add_tensor_entry(
            model_state,
            weight_v_key,
            up_weight,
            model_template.get(weight_v_key),
        )
        return 2, 2

    checkpoint_key = _resolve_modelopt_fp8_weight_key(task.param_name, converted, model_template)
    add_tensor_entry(
        model_state,
        checkpoint_key,
        converted,
        model_template.get(checkpoint_key),
    )
    return 1, int(checkpoint_key != task.param_name)


def _load_scale(weight_key: str, hf_state_dict: Mapping[str, torch.Tensor]) -> torch.Tensor | None:
    key = f"{weight_key}_scale_inv"
    if key in hf_state_dict:
        return hf_state_dict[key].float()

    legacy_key = f"{weight_key}_scale"
    if legacy_key in hf_state_dict:
        return 1.0 / hf_state_dict[legacy_key].float()

    return None


def convert_hf_weight_for_direct_save(task: Any, hf_weights: Any) -> torch.Tensor:
    """Convert one HF tensor payload into the direct-save Megatron representation."""
    mapping = task.mapping

    if getattr(mapping, "tp_size", 1) != 1:
        raise ValueError(
            "Direct FP8 converter currently supports only single-rank TP=1 checkpoint writes."
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


def build_merged_scale_inv_for_task(
    task: Any,
    hf_state_dict: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Build the persisted FP8 scale tensor for one converted task."""
    mapping = task.mapping
    hf_param = mapping.hf_param

    if isinstance(mapping, QKVMapping):
        q_scale = _load_scale(hf_param["q"], hf_state_dict)
        k_scale = _load_scale(hf_param["k"], hf_state_dict)
        v_scale = _load_scale(hf_param["v"], hf_state_dict)
        if q_scale is None or k_scale is None or v_scale is None:
            raise RuntimeError(f"Missing FP8 QKV scale for {task.param_name}")
        config = mapping._get_config(task.megatron_module)
        return merge_qkv_scale_inv(config, q_scale, k_scale, v_scale).contiguous()

    if isinstance(mapping, GatedMLPMapping):
        gate_scale = _load_scale(hf_param["gate"], hf_state_dict)
        up_scale = _load_scale(hf_param["up"], hf_state_dict)
        if gate_scale is None or up_scale is None:
            raise RuntimeError(f"Missing FP8 gated-MLP scale for {task.param_name}")
        return merge_gated_mlp_scale_inv(gate_scale, up_scale).contiguous()

    if isinstance(hf_param, str):
        scale = _load_scale(hf_param, hf_state_dict)
        if scale is None:
            raise RuntimeError(f"Missing FP8 scale for {task.param_name}")
        return scale.contiguous()

    raise RuntimeError(f"Unsupported FP8 scale mapping for {task.param_name}")


def build_fp8_direct_model_state_dict(
    bridge: Any,
    hf_pretrained: Any,
    meta_model: list[Any],
    model_template: dict[str, Any],
) -> dict[str, Any]:
    """Build a direct-save Megatron model state with explicit FP8 scale entries."""
    if len(meta_model) != 1:
        raise ValueError(
            "Direct FP8 converter currently supports a single Megatron model chunk (no VP stages)."
        )

    model_state = prepare_empty_model_state(model_template)
    conversion_tasks = bridge.build_conversion_tasks(hf_pretrained, meta_model)
    hf_state_dict: Mapping[str, torch.Tensor] = hf_pretrained.state

    num_weights = 0
    num_scales = 0
    num_modelopt_weight_keys = 0
    total_tasks = len(conversion_tasks)
    log_interval = max(1, total_tasks // 100)
    t_start = time.monotonic()

    for i, task in enumerate(conversion_tasks):
        if task is None or task.megatron_module is None:
            continue

        hf_weights = bridge.maybe_modify_loaded_hf_weight(task.mapping.hf_param, hf_state_dict)
        converted = convert_hf_weight_for_direct_save(task, hf_weights)
        weight_entries, renamed_entries = _add_fp8_weight_entries(
            model_state,
            task,
            converted,
            model_template,
        )
        num_weights += weight_entries
        num_modelopt_weight_keys += renamed_entries

        if converted.dtype == torch.float8_e4m3fn:
            scale_inv = build_merged_scale_inv_for_task(task, hf_state_dict)
            scale_inv_key = build_fp8_scale_inv_key(task.param_name)
            add_tensor_entry(
                model_state,
                scale_inv_key,
                scale_inv,
                model_template.get(scale_inv_key),
            )
            num_scales += 1

        if (i + 1) % log_interval == 0 or (i + 1) == total_tasks:
            elapsed = time.monotonic() - t_start
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            print(
                f"  [{i+1}/{total_tasks}] {num_weights} weights, {num_scales} scales, "
                f"{num_modelopt_weight_keys} modelopt weight keys | "
                f"elapsed {elapsed_str} | {task.param_name}",
                flush=True,
            )

    return model_state
