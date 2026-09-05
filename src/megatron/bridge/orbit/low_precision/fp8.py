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

import logging
import math
import time
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

import torch
from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory

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
    add_tensor_entry,
    prepare_empty_model_state,
)
from megatron.bridge.orbit.quant.fp8_utils import (
    FP8_WEIGHT_BLOCK_SIZE,
    merge_gated_mlp_scale_inv,
    merge_qkv_scale_inv,
)


logger = logging.getLogger(__name__)


__all__ = [
    "FP8ConversionPlan",
    "apply_modelopt_fp8_to_meta_model",
    "build_fp8_direct_model_state_dict",
    "build_fp8_scale_inv_key",
    "build_merged_scale_inv_for_task",
    "collect_fp8_target_module_names",
    "convert_hf_weight_for_direct_save",
    "preflight_fp8_conversion_tasks",
]


@dataclass(frozen=True)
class FP8ConversionPlan:
    """Immutable result of validating direct FP8 conversion tasks."""

    module_names: frozenset[str]
    fp8_task_ids: frozenset[int]
    _conversion_task_ids: tuple[int, ...]
    _state_dict_id: int
    _source_shapes: tuple[tuple[str, tuple[int, ...]], ...]

    def source_shape(self, key: str) -> tuple[int, ...] | None:
        """Return the prevalidated source shape for an FP8 tensor key."""
        return dict(self._source_shapes).get(key)


def _selective_weight_quant_cfg(
    quant_cfg: list[dict[str, Any]] | dict[str, dict[str, Any]],
    module_names: Iterable[str],
) -> list[dict[str, Any]] | dict[str, dict[str, Any]]:
    """Disable generic weight quantization and enable only exact modules."""
    exact_quantizers = [f"{name}.weight_quantizer" for name in sorted(set(module_names))]

    if isinstance(quant_cfg, list):
        generic_indices = [
            index
            for index, entry in enumerate(quant_cfg)
            if isinstance(entry, Mapping) and entry.get("quantizer_name") == "*weight_quantizer"
        ]
        if not generic_indices:
            raise ValueError("ModelOpt FP8 config is missing a generic '*weight_quantizer' entry")

        generic_index = generic_indices[-1]
        enabled_entry = deepcopy(quant_cfg[generic_index])
        disabled_entry = {"quantizer_name": "*weight_quantizer", "enable": False}
        exact_entries = []
        for quantizer_name in exact_quantizers:
            exact_entry = deepcopy(enabled_entry)
            exact_entry["quantizer_name"] = quantizer_name
            exact_entries.append(exact_entry)
        return [
            *quant_cfg[:generic_index],
            disabled_entry,
            *exact_entries,
            *quant_cfg[generic_index + 1 :],
        ]

    if isinstance(quant_cfg, dict):
        if "*weight_quantizer" not in quant_cfg:
            raise ValueError("ModelOpt FP8 config is missing a generic '*weight_quantizer' entry")

        enabled_cfg = deepcopy(quant_cfg["*weight_quantizer"])
        selective_cfg: dict[str, dict[str, Any]] = {}
        for quantizer_name, config in quant_cfg.items():
            if quantizer_name != "*weight_quantizer":
                selective_cfg[quantizer_name] = config
                continue
            selective_cfg[quantizer_name] = {"enable": False}
            for exact_quantizer in exact_quantizers:
                selective_cfg[exact_quantizer] = deepcopy(enabled_cfg)
        return selective_cfg

    raise TypeError(f"ModelOpt FP8 quant_cfg must be an ordered list or legacy dict, got {type(quant_cfg).__name__}")


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
        quant_cfg["quant_cfg"] = _selective_weight_quant_cfg(quant_cfg.get("quant_cfg", {}), module_names)

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
) -> str:
    """Return the on-disk key expected by a restored ModelOpt-compressed module."""
    if converted.dtype != torch.float8_e4m3fn or not param_name.endswith(".weight"):
        return param_name

    modelopt_weight_key = f"{param_name}_w"
    return modelopt_weight_key


def _is_linear_fc1_weight(param_name: str) -> bool:
    module_name, separator, leaf_name = param_name.rpartition(".")
    is_weight = leaf_name == "weight" or (leaf_name.startswith("weight") and leaf_name[6:].isdigit())
    return bool(separator) and module_name.rsplit(".", 1)[-1] == "linear_fc1" and is_weight


def _grouped_expert_index(param_name: str) -> int | None:
    module_name, separator, leaf_name = param_name.rpartition(".")
    if not separator or module_name.rsplit(".", 2)[-2:] not in (
        ["experts", "linear_fc1"],
        ["experts", "linear_fc2"],
    ):
        return None
    if not leaf_name.startswith("weight") or not leaf_name[6:].isdigit():
        return None
    return int(leaf_name[6:])


def _uses_swiglu_split_layout(
    task: Any,
    model_template: Mapping[str, Any],
) -> bool:
    if not _is_linear_fc1_weight(task.param_name):
        return False
    config = getattr(getattr(task, "megatron_module", None), "config", None)
    runtime_template = model_template.get(task.param_name)
    return (
        bool(getattr(config, "gated_linear_unit", False))
        or isinstance(runtime_template, ShardedTensorFactory)
        or f"{task.param_name}_w" in model_template
        or f"{task.param_name}_v" in model_template
    )


def _template_local_shape(template_entry: Any) -> tuple[int, ...] | None:
    if isinstance(template_entry, ShardedTensor):
        return tuple(template_entry.local_shape)
    if isinstance(template_entry, ShardedTensorFactory) and isinstance(template_entry.data, torch.Tensor):
        return tuple(template_entry.data.shape)
    return None


def _validate_template_shape(key: str, tensor: torch.Tensor, template_entry: Any | None) -> None:
    template_shape = _template_local_shape(template_entry)
    if template_shape is not None and template_shape != tuple(tensor.shape):
        raise ValueError(
            f"FP8 tensor {key!r} has shape {tuple(tensor.shape)}, but its model template expects {template_shape}"
        )


def _normalize_fp8_direct_template(
    param_name: str,
    checkpoint_key: str,
    converted: torch.Tensor,
    template_entry: Any | None,
) -> Any | None:
    """Convert MCore's shared-axis metadata to one direct FP8 tensor entry."""
    if template_entry is None:
        return template_entry
    if not isinstance(template_entry, ShardedTensor):
        raise ValueError(f"Direct FP8 tensor {checkpoint_key!r} requires a sharded-tensor model template")
    if template_entry.flattened_range is not None or template_entry.axis_fragmentations is None:
        raise ValueError(f"Direct FP8 tensor {checkpoint_key!r} requires regular, unflattened shard geometry")

    prepend = template_entry.prepend_axis_num
    expert_index = _grouped_expert_index(param_name)
    if expert_index is not None and template_entry.key not in {param_name, checkpoint_key}:
        if prepend == 0:
            raise ValueError(
                f"Direct FP8 grouped weight {param_name!r} has unsupported physical key {template_entry.key!r}"
            )
        expert_axis = prepend - 1
        global_expert_index = template_entry.global_offset[expert_axis]
        if global_expert_index != expert_index:
            raise ValueError(
                f"Direct FP8 grouped weight {param_name!r} maps local expert {expert_index} "
                f"to global expert {global_expert_index}; only EP=1 direct conversion is supported"
            )

    global_shape = tuple(template_entry.global_shape[prepend:])
    global_offset = tuple(template_entry.global_offset[prepend:])
    axis_fragmentations = tuple(template_entry.axis_fragmentations[prepend:])
    expected_shape = tuple(converted.shape)
    if (
        tuple(template_entry.local_shape) != expected_shape
        or global_shape != expected_shape
        or global_offset != (0, 0)
        or axis_fragmentations != (1, 1)
    ):
        raise ValueError(
            f"Direct FP8 tensor {checkpoint_key!r} is not single-rank geometry after removing its shared axes"
        )

    return replace(
        template_entry,
        key=checkpoint_key,
        data=None,
        global_shape=global_shape,
        global_offset=global_offset,
        axis_fragmentations=axis_fragmentations,
        prepend_axis_num=0,
    )


def _add_tensor_entry_preserving_template_key(
    model_state: dict[str, Any],
    key: str,
    tensor: torch.Tensor,
    template_entry: Any | None,
) -> None:
    """Materialize a regular Megatron entry without changing its physical key."""
    add_tensor_entry(model_state, key, tensor, template_entry)
    if isinstance(template_entry, (ShardedTensor, ShardedTensorFactory)):
        model_state[key] = replace(model_state[key], key=template_entry.key)


def _factory_swiglu_split_templates(
    param_name: str,
    converted: torch.Tensor,
    factory: ShardedTensorFactory,
) -> tuple[ShardedTensor, ShardedTensor]:
    """Convert MCore's same-key SwiGLU factory metadata to explicit FP8 halves."""
    if factory.flattened_range is not None:
        raise ValueError(f"Direct FP8 SwiGLU conversion does not support flattened factory {param_name!r}")
    if not isinstance(factory.data, torch.Tensor) or tuple(factory.data.shape) != tuple(converted.shape):
        actual_shape = tuple(factory.data.shape) if isinstance(factory.data, torch.Tensor) else None
        raise ValueError(
            f"FP8 SwiGLU template for {param_name!r} has shape {actual_shape}, "
            f"but converted weight has shape {tuple(converted.shape)}"
        )
    if converted.ndim != 2 or converted.shape[0] % 2 != 0:
        raise ValueError(f"Cannot split SwiGLU FP8 weight for {param_name!r}: shape={tuple(converted.shape)}")

    built = factory.build()
    if (
        not isinstance(built, (list, tuple))
        or len(built) != 2
        or not all(isinstance(entry, ShardedTensor) for entry in built)
    ):
        raise ValueError(f"FP8 SwiGLU factory for {param_name!r} must build exactly two sharded tensors")

    built_keys = [entry.key for entry in built]
    same_physical_key = built_keys[0] == built_keys[1] == factory.key
    explicit_physical_keys = built_keys == [f"{factory.key}_w", f"{factory.key}_v"]
    if not same_physical_key and not explicit_physical_keys:
        raise ValueError(f"FP8 SwiGLU factory for {param_name!r} produced unsupported physical keys {built_keys!r}")

    half_shape = (converted.shape[0] // 2, converted.shape[1])
    split_templates = []
    for suffix, entry in zip(("w", "v"), built):
        if tuple(entry.local_shape) != half_shape:
            raise ValueError(
                f"FP8 SwiGLU factory half {suffix!r} for {param_name!r} has local shape "
                f"{tuple(entry.local_shape)}, expected {half_shape}"
            )
        if entry.flattened_range is not None:
            raise ValueError(f"Direct FP8 SwiGLU conversion does not support flattened shard {param_name!r}")
        if entry.axis_fragmentations is None:
            raise ValueError(f"Direct FP8 SwiGLU conversion requires regular shard geometry for {param_name!r}")

        prepend = entry.prepend_axis_num
        global_shape = list(entry.global_shape[prepend:])
        global_offset = list(entry.global_offset[prepend:])
        axis_fragmentations = list(entry.axis_fragmentations[prepend:])
        if len(global_shape) != 2:
            raise ValueError(f"FP8 SwiGLU factory half {suffix!r} for {param_name!r} must describe a rank-2 weight")
        if same_physical_key:
            if global_shape[0] % 2 != 0 or axis_fragmentations[0] % 2 != 0:
                raise ValueError(f"FP8 SwiGLU factory for {param_name!r} has incompatible split geometry")
            global_shape[0] //= 2
            global_offset[0] %= global_shape[0]
            axis_fragmentations[0] //= 2

        if tuple(global_shape) != half_shape:
            raise ValueError(
                f"FP8 SwiGLU factory half {suffix!r} for {param_name!r} has global shape "
                f"{tuple(global_shape)}, expected {half_shape} for single-rank direct conversion"
            )
        if tuple(global_offset) != (0, 0) or tuple(axis_fragmentations) != (1, 1):
            raise ValueError(f"FP8 SwiGLU factory half {suffix!r} for {param_name!r} is not single-rank geometry")

        split_key = f"{param_name}_{suffix}"
        split_templates.append(
            ShardedTensor(
                key=split_key,
                data=None,
                dtype=entry.dtype,
                local_shape=tuple(entry.local_shape),
                global_shape=tuple(global_shape),
                global_offset=tuple(global_offset),
                axis_fragmentations=tuple(axis_fragmentations),
                replica_id=entry.replica_id,
                prepend_axis_num=0,
                allow_shape_mismatch=entry.allow_shape_mismatch,
            )
        )

    return split_templates[0], split_templates[1]


def _add_fp8_weight_entries(
    model_state: dict[str, Any],
    task: Any,
    converted: torch.Tensor,
    model_template: Mapping[str, Any],
) -> tuple[int, int]:
    """Add converted weight tensors and return ``(weight_entries, renamed_entries)``."""
    if converted.dtype == torch.float8_e4m3fn and _uses_swiglu_split_layout(task, model_template):
        if converted.shape[0] % 2 != 0:
            raise ValueError(
                f"Cannot split SwiGLU FP8 weight with odd output dimension for {task.param_name}: "
                f"shape={tuple(converted.shape)}"
            )
        gate_weight, up_weight = torch.chunk(converted, 2, dim=0)
        weight_w_key = f"{task.param_name}_w"
        weight_v_key = f"{task.param_name}_v"
        runtime_template = model_template.get(task.param_name)
        if isinstance(runtime_template, ShardedTensorFactory):
            weight_w_template, weight_v_template = _factory_swiglu_split_templates(
                task.param_name,
                converted,
                runtime_template,
            )
        else:
            has_weight_w = weight_w_key in model_template
            has_weight_v = weight_v_key in model_template
            if has_weight_w != has_weight_v:
                raise ValueError(
                    f"Model template has incomplete FP8 SwiGLU layout for {task.param_name!r}; "
                    f"both {weight_w_key!r} and {weight_v_key!r} are required"
                )
            if not has_weight_w:
                raise ValueError(
                    f"Direct FP8 conversion cannot determine the SwiGLU split layout for "
                    f"{task.param_name!r} without a factory or explicit split templates"
                )
            weight_w_template = model_template[weight_w_key]
            weight_v_template = model_template[weight_v_key]

        weight_w_template = _normalize_fp8_direct_template(
            task.param_name,
            weight_w_key,
            gate_weight,
            weight_w_template,
        )
        weight_v_template = _normalize_fp8_direct_template(
            task.param_name,
            weight_v_key,
            up_weight,
            weight_v_template,
        )

        _validate_template_shape(weight_w_key, gate_weight, weight_w_template)
        _validate_template_shape(weight_v_key, up_weight, weight_v_template)
        add_tensor_entry(
            model_state,
            weight_w_key,
            gate_weight,
            weight_w_template,
        )
        add_tensor_entry(
            model_state,
            weight_v_key,
            up_weight,
            weight_v_template,
        )
        return 2, 2

    checkpoint_key = _resolve_modelopt_fp8_weight_key(task.param_name, converted)
    checkpoint_template = model_template.get(checkpoint_key)
    if checkpoint_template is None:
        runtime_template = model_template.get(task.param_name)
        if isinstance(runtime_template, (ShardedTensor, ShardedTensorFactory)):
            checkpoint_template = runtime_template
    _validate_template_shape(checkpoint_key, converted, checkpoint_template)
    if converted.dtype == torch.float8_e4m3fn:
        checkpoint_template = _normalize_fp8_direct_template(
            task.param_name,
            checkpoint_key,
            converted,
            checkpoint_template,
        )
        add_tensor_entry(model_state, checkpoint_key, converted, checkpoint_template)
    else:
        _add_tensor_entry_preserving_template_key(
            model_state,
            checkpoint_key,
            converted,
            checkpoint_template,
        )
    return 1, int(checkpoint_key != task.param_name)


@dataclass(frozen=True)
class _SourceTensorMetadata:
    dtype: torch.dtype | str
    shape: tuple[int, ...]


def _source_tensor_metadata(
    key: str,
    hf_state_dict: Mapping[str, torch.Tensor],
) -> _SourceTensorMetadata:
    if key not in hf_state_dict:
        raise ValueError(f"Missing FP8 source weight {key!r}")

    from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource

    source = getattr(hf_state_dict, "source", None)
    if isinstance(source, SafeTensorsStateSource):
        filename = source.key_to_filename_map.get(key)
        if filename is None:
            raise ValueError(f"Missing safetensors shard metadata for FP8 source weight {key!r}")

        from safetensors import safe_open

        with safe_open(source.path / filename, framework="pt", device="cpu") as handle:
            tensor_slice = handle.get_slice(key)
            return _SourceTensorMetadata(
                dtype=tensor_slice.get_dtype(),
                shape=tuple(int(dim) for dim in tensor_slice.get_shape()),
            )

    weight = hf_state_dict[key]
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"FP8 source weight {key!r} must be a tensor")
    return _SourceTensorMetadata(dtype=weight.dtype, shape=tuple(weight.shape))


def _fp8_scale_grid_shape(
    weight_or_shape: torch.Tensor | tuple[int, ...],
    *,
    key: str,
) -> tuple[int, int]:
    shape = tuple(weight_or_shape.shape) if isinstance(weight_or_shape, torch.Tensor) else weight_or_shape
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"FP8 source weight {key!r} must be a non-empty rank-2 tensor, got shape={shape}")
    return (
        math.ceil(int(shape[0]) / FP8_WEIGHT_BLOCK_SIZE),
        math.ceil(int(shape[1]) / FP8_WEIGHT_BLOCK_SIZE),
    )


def _scale_sibling_keys(weight_key: str, hf_state_dict: Mapping[str, torch.Tensor]) -> tuple[str, ...]:
    return tuple(key for key in (f"{weight_key}_scale_inv", f"{weight_key}_scale") if key in hf_state_dict)


def _load_scale(
    weight_key: str,
    hf_state_dict: Mapping[str, torch.Tensor],
    *,
    weight_shape: tuple[int, ...] | None = None,
) -> torch.Tensor | None:
    """Load and validate one source scale, normalized to the exact FP8 block grid."""
    scale_keys = _scale_sibling_keys(weight_key, hf_state_dict)
    if not scale_keys:
        return None
    if len(scale_keys) != 1:
        raise ValueError(
            f"FP8 source weight {weight_key!r} must have exactly one scale sibling: "
            f"{weight_key}_scale_inv or {weight_key}_scale"
        )

    if weight_shape is None:
        weight_shape = _source_tensor_metadata(weight_key, hf_state_dict).shape
    expected_shape = _fp8_scale_grid_shape(weight_shape, key=weight_key)

    scale_key = scale_keys[0]
    raw_scale = hf_state_dict[scale_key]
    if not isinstance(raw_scale, torch.Tensor):
        raise TypeError(f"Invalid FP8 scale {scale_key!r}: scale must be a tensor")
    if not raw_scale.is_floating_point():
        raise TypeError(f"Invalid FP8 scale {scale_key!r}: scale must be floating, got {raw_scale.dtype}")

    scale = raw_scale.detach().to(dtype=torch.float32)
    if scale.numel() == 0 or not bool(torch.all(torch.isfinite(scale) & (scale > 0)).item()):
        raise ValueError(f"Invalid FP8 scale {scale_key!r}: scale values must be finite and positive")

    if scale.numel() == 1:
        scale = scale.reshape(()).expand(expected_shape).clone()
    elif scale.ndim != 2 or tuple(scale.shape) != expected_shape:
        raise ValueError(
            f"Invalid FP8 scale {scale_key!r}: expected scalar or FP8 scale grid {expected_shape} "
            f"for weight shape {weight_shape}, got {tuple(scale.shape)}"
        )

    if scale_key.endswith("_scale"):
        scale = scale.reciprocal()
        if not bool(torch.all(torch.isfinite(scale) & (scale > 0)).item()):
            raise ValueError(f"Invalid FP8 scale {scale_key!r}: reciprocal scale values must be finite and positive")

    return scale.contiguous()


def _source_weight_keys(hf_param: Any) -> tuple[str, ...]:
    if isinstance(hf_param, str):
        return (hf_param,)
    if isinstance(hf_param, Mapping):
        keys = tuple(hf_param.values())
        if keys and all(isinstance(key, str) for key in keys):
            return keys
    raise RuntimeError(f"Unsupported FP8 source mapping: {hf_param!r}")


def _is_e4m3fn_dtype(dtype: torch.dtype | str) -> bool:
    return dtype == torch.float8_e4m3fn or str(dtype) in {
        "F8_E4M3",
        "F8_E4M3FN",
        "torch.float8_e4m3fn",
    }


def _is_float8_dtype(dtype: torch.dtype | str) -> bool:
    dtype_name = str(dtype)
    return dtype_name.startswith(("torch.float8_", "float8_", "F8_"))


def _validate_fp8_source_weight(
    weight_key: str,
    hf_state_dict: Mapping[str, torch.Tensor],
) -> tuple[bool, tuple[int, ...]]:
    metadata = _source_tensor_metadata(weight_key, hf_state_dict)

    scale_keys = _scale_sibling_keys(weight_key, hf_state_dict)
    if _is_e4m3fn_dtype(metadata.dtype):
        _fp8_scale_grid_shape(metadata.shape, key=weight_key)
        if len(scale_keys) != 1:
            raise ValueError(
                f"FP8 source weight {weight_key!r} must have exactly one scale sibling: "
                f"{weight_key}_scale_inv or {weight_key}_scale"
            )
        _load_scale(weight_key, hf_state_dict, weight_shape=metadata.shape)
        return True, metadata.shape

    if _is_float8_dtype(metadata.dtype):
        raise TypeError(
            f"Unsupported FP8 source weight {weight_key!r}: only torch.float8_e4m3fn is supported, "
            f"got {metadata.dtype}"
        )
    if scale_keys:
        raise ValueError(
            f"FP8 scale metadata {scale_keys!r} is attached to non-FP8 weight {weight_key!r} "
            f"with dtype {metadata.dtype}"
        )
    return False, metadata.shape


def _is_supported_fp8_target_param(param_name: str) -> bool:
    module_name, separator, leaf_name = param_name.rpartition(".")
    is_weight = leaf_name == "weight" or (leaf_name.startswith("weight") and leaf_name[6:].isdigit())
    linear_name = module_name.rsplit(".", 1)[-1]
    return (
        bool(separator)
        and is_weight
        and linear_name
        in {
            "linear_qkv",
            "linear_proj",
            "linear_fc1",
            "linear_fc2",
        }
    )


def _validate_fused_fp8_source_geometry(
    task: Any,
    source_shapes: Mapping[str, tuple[int, ...]],
) -> None:
    mapping = task.mapping
    if isinstance(mapping, GatedMLPMapping):
        gate_shape = source_shapes[mapping.hf_param["gate"]]
        up_shape = source_shapes[mapping.hf_param["up"]]
        if gate_shape != up_shape:
            raise ValueError(
                f"FP8 gate and up weights for {task.param_name!r} must have matching shapes, "
                f"got {gate_shape} and {up_shape}"
            )
        if gate_shape[0] % FP8_WEIGHT_BLOCK_SIZE != 0:
            raise ValueError(
                f"FP8 gated-MLP source for {task.param_name!r} has a concatenation boundary "
                f"inside a {FP8_WEIGHT_BLOCK_SIZE}-element scale block"
            )
        return

    if not isinstance(mapping, QKVMapping):
        return

    q_shape = source_shapes[mapping.hf_param["q"]]
    k_shape = source_shapes[mapping.hf_param["k"]]
    v_shape = source_shapes[mapping.hf_param["v"]]
    if len({q_shape[1], k_shape[1], v_shape[1]}) != 1:
        raise ValueError(
            f"FP8 QKV source weights for {task.param_name!r} must have matching input dimensions, "
            f"got {q_shape}, {k_shape}, and {v_shape}"
        )

    config = mapping._get_config(task.megatron_module)
    num_heads = int(config.num_attention_heads)
    num_query_groups = int(config.num_query_groups)
    head_size = int(config.kv_channels)
    if num_heads <= 0 or num_query_groups <= 0 or num_heads % num_query_groups != 0 or head_size <= 0:
        raise ValueError(
            f"Invalid FP8 QKV geometry for {task.param_name!r}: "
            f"heads={num_heads}, query_groups={num_query_groups}, head_size={head_size}"
        )
    if head_size % FP8_WEIGHT_BLOCK_SIZE != 0:
        raise ValueError(
            f"FP8 QKV head for {task.param_name!r} has a boundary inside a "
            f"{FP8_WEIGHT_BLOCK_SIZE}-element scale block: head_size={head_size}"
        )

    q_multiplier = 2 if bool(getattr(config, "attention_output_gate", False)) else 1
    expected_q_rows = num_heads * head_size * q_multiplier
    expected_kv_rows = num_query_groups * head_size
    if q_shape[0] != expected_q_rows or k_shape[0] != expected_kv_rows or v_shape[0] != expected_kv_rows:
        raise ValueError(
            f"FP8 QKV source shapes for {task.param_name!r} disagree with model geometry: "
            f"q={q_shape}, k={k_shape}, v={v_shape}, expected output rows "
            f"({expected_q_rows}, {expected_kv_rows}, {expected_kv_rows})"
        )


def _validate_complete_direct_tasks(conversion_tasks: list[Any]) -> None:
    for index, task in enumerate(conversion_tasks):
        if task is None or task.megatron_module is None:
            raise RuntimeError(
                f"Direct FP8 conversion has an incomplete conversion task at index {index}; "
                "single-rank direct conversion requires every model parameter to have a local HF mapping"
            )


def preflight_fp8_conversion_tasks(
    conversion_tasks: list[Any],
    hf_state_dict: Mapping[str, torch.Tensor],
    *,
    require_complete: bool = False,
) -> FP8ConversionPlan:
    """Validate task families once without materializing safetensors source weights."""
    if require_complete:
        _validate_complete_direct_tasks(conversion_tasks)
    module_names: set[str] = set()
    fp8_task_ids: set[int] = set()
    source_results: dict[str, tuple[bool, tuple[int, ...]]] = {}
    fp8_source_shapes: dict[str, tuple[int, ...]] = {}
    for task in conversion_tasks:
        if task is None or task.megatron_module is None:
            continue

        source_keys = _source_weight_keys(task.mapping.hf_param)
        for key in source_keys:
            if key not in source_results:
                source_results[key] = _validate_fp8_source_weight(key, hf_state_dict)
        source_is_fp8 = [source_results[key][0] for key in source_keys]
        if any(source_is_fp8) and not all(source_is_fp8):
            raise ValueError(
                f"FP8 mapping for {task.param_name!r} mixes E4M3FN and non-FP8 source weights: {source_keys!r}"
            )
        if not source_is_fp8 or not all(source_is_fp8):
            continue
        if not isinstance(
            task.mapping,
            (
                AutoMapping,
                DirectMapping,
                ColumnParallelMapping,
                RowParallelMapping,
                ReplicatedMapping,
                QKVMapping,
                GatedMLPMapping,
            ),
        ):
            raise RuntimeError(
                f"Direct FP8 conversion has unsupported mapping type {type(task.mapping).__name__} "
                f"for {task.param_name!r}"
            )
        if isinstance(task.mapping, AutoMapping) and task.mapping.permute_dims is not None:
            raise RuntimeError(
                f"Direct FP8 conversion does not support a permuted FP8 mapping for {task.param_name!r}"
            )
        if not _is_supported_fp8_target_param(task.param_name):
            raise ValueError(f"FP8 source family maps to unsupported target parameter {task.param_name!r}")
        if len(source_keys) > 1 and not isinstance(task.mapping, (QKVMapping, GatedMLPMapping)):
            raise RuntimeError(f"Unsupported FP8 fused scale mapping for {task.param_name!r}")

        _validate_fused_fp8_source_geometry(
            task,
            {key: source_results[key][1] for key in source_keys},
        )
        module_names.add(task.param_name.rsplit(".", 1)[0])
        fp8_task_ids.add(id(task))
        fp8_source_shapes.update({key: source_results[key][1] for key in source_keys})

    return FP8ConversionPlan(
        module_names=frozenset(module_names),
        fp8_task_ids=frozenset(fp8_task_ids),
        _conversion_task_ids=tuple(id(task) for task in conversion_tasks),
        _state_dict_id=id(hf_state_dict),
        _source_shapes=tuple(sorted(fp8_source_shapes.items())),
    )


def _validate_fp8_plan_inputs(
    plan: FP8ConversionPlan,
    conversion_tasks: list[Any],
    hf_state_dict: Mapping[str, torch.Tensor],
) -> None:
    if plan._conversion_task_ids != tuple(id(task) for task in conversion_tasks):
        raise ValueError("FP8 preflight plan does not match the supplied conversion tasks")
    if plan._state_dict_id != id(hf_state_dict):
        raise ValueError("FP8 preflight plan does not match the supplied source state")


def _classify_fp8_conversion_tasks(
    conversion_tasks: list[Any],
    hf_state_dict: Mapping[str, torch.Tensor],
) -> tuple[set[str], set[int]]:
    """Compatibility wrapper for callers expecting the previous classification tuple."""
    plan = preflight_fp8_conversion_tasks(conversion_tasks, hf_state_dict)
    return set(plan.module_names), set(plan.fp8_task_ids)


def collect_fp8_target_module_names(
    conversion_tasks: list[Any],
    hf_state_dict: Mapping[str, torch.Tensor],
) -> set[str]:
    """Return ModelOpt module names backed by complete, validated E4M3FN source families."""
    return set(preflight_fp8_conversion_tasks(conversion_tasks, hf_state_dict).module_names)


def _canonicalize_merged_scale_inv(
    task: Any,
    converted: torch.Tensor,
    scale_inv: torch.Tensor,
) -> torch.Tensor:
    expected_shape = _fp8_scale_grid_shape(converted, key=task.param_name)
    scale_inv = scale_inv.to(dtype=torch.float32)
    if scale_inv.numel() == 1:
        scale_inv = scale_inv.reshape(()).expand(expected_shape).clone()
    elif scale_inv.ndim != 2 or tuple(scale_inv.shape) != expected_shape:
        raise ValueError(
            f"Merged FP8 scale for {task.param_name!r} must have grid {expected_shape}, got {tuple(scale_inv.shape)}"
        )
    if not bool(torch.all(torch.isfinite(scale_inv) & (scale_inv > 0)).item()):
        raise ValueError(f"Merged FP8 scale for {task.param_name!r} must be finite and positive")
    return scale_inv.contiguous()


def convert_hf_weight_for_direct_save(task: Any, hf_weights: Any) -> torch.Tensor:
    """Convert one HF tensor payload into the direct-save Megatron representation."""
    mapping = task.mapping

    if getattr(mapping, "tp_size", 1) != 1:
        raise ValueError("Direct FP8 converter currently supports only single-rank TP=1 checkpoint writes.")

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
    *,
    source_shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> torch.Tensor:
    """Build the persisted FP8 scale tensor for one converted task."""
    mapping = task.mapping
    hf_param = mapping.hf_param
    source_shapes = source_shapes or {}

    if isinstance(mapping, QKVMapping):
        q_scale = _load_scale(hf_param["q"], hf_state_dict, weight_shape=source_shapes.get(hf_param["q"]))
        k_scale = _load_scale(hf_param["k"], hf_state_dict, weight_shape=source_shapes.get(hf_param["k"]))
        v_scale = _load_scale(hf_param["v"], hf_state_dict, weight_shape=source_shapes.get(hf_param["v"]))
        if q_scale is None or k_scale is None or v_scale is None:
            raise RuntimeError(f"Missing FP8 QKV scale for {task.param_name}")
        config = mapping._get_config(task.megatron_module)
        return merge_qkv_scale_inv(config, q_scale, k_scale, v_scale).contiguous()

    if isinstance(mapping, GatedMLPMapping):
        gate_scale = _load_scale(
            hf_param["gate"],
            hf_state_dict,
            weight_shape=source_shapes.get(hf_param["gate"]),
        )
        up_scale = _load_scale(
            hf_param["up"],
            hf_state_dict,
            weight_shape=source_shapes.get(hf_param["up"]),
        )
        if gate_scale is None or up_scale is None:
            raise RuntimeError(f"Missing FP8 gated-MLP scale for {task.param_name}")
        return merge_gated_mlp_scale_inv(gate_scale, up_scale).contiguous()

    if isinstance(hf_param, str):
        scale = _load_scale(hf_param, hf_state_dict, weight_shape=source_shapes.get(hf_param))
        if scale is None:
            raise RuntimeError(f"Missing FP8 scale for {task.param_name}")
        return scale.contiguous()

    raise RuntimeError(f"Unsupported FP8 scale mapping for {task.param_name}")


def build_fp8_direct_model_state_dict(
    bridge: Any,
    hf_pretrained: Any,
    meta_model: list[Any],
    model_template: dict[str, Any],
    *,
    conversion_tasks: list[Any] | None = None,
    fp8_plan: FP8ConversionPlan | None = None,
) -> dict[str, Any]:
    """Build a direct-save Megatron model state with explicit FP8 scale entries."""
    if len(meta_model) != 1:
        raise ValueError("Direct FP8 converter currently supports a single Megatron model chunk (no VP stages).")

    model_state = prepare_empty_model_state(model_template)
    if conversion_tasks is None:
        conversion_tasks = bridge.build_conversion_tasks(hf_pretrained, meta_model)
    _validate_complete_direct_tasks(conversion_tasks)
    hf_state_dict: Mapping[str, torch.Tensor] = hf_pretrained.state
    if fp8_plan is None:
        fp8_plan = preflight_fp8_conversion_tasks(
            conversion_tasks,
            hf_state_dict,
            require_complete=True,
        )
    else:
        _validate_fp8_plan_inputs(fp8_plan, conversion_tasks, hf_state_dict)
    fp8_task_ids = fp8_plan.fp8_task_ids
    if not fp8_task_ids:
        raise RuntimeError("Direct FP8 conversion found no complete FP8 mappings in the source checkpoint")
    source_shapes = dict(fp8_plan._source_shapes)

    num_weights = 0
    num_fp8 = 0
    num_scales = 0
    num_modelopt_weight_keys = 0
    total_tasks = len(conversion_tasks)
    log_interval = max(1, total_tasks // 100)
    t_start = time.monotonic()

    for i, task in enumerate(conversion_tasks):
        if task is None or task.megatron_module is None:
            continue

        source_is_fp8 = id(task) in fp8_task_ids
        hf_weights = bridge.maybe_modify_loaded_hf_weight(task.mapping.hf_param, hf_state_dict)
        converted = convert_hf_weight_for_direct_save(task, hf_weights)
        if source_is_fp8 and converted.dtype != torch.float8_e4m3fn:
            raise TypeError(
                f"FP8 source family for {task.param_name!r} converted to unsupported dtype {converted.dtype}"
            )
        if not source_is_fp8 and _is_float8_dtype(converted.dtype):
            raise TypeError(
                f"Non-FP8 source family for {task.param_name!r} unexpectedly converted to {converted.dtype}"
            )
        weight_entries, renamed_entries = _add_fp8_weight_entries(
            model_state,
            task,
            converted,
            model_template,
        )
        num_weights += weight_entries
        num_modelopt_weight_keys += renamed_entries

        if source_is_fp8:
            scale_inv = build_merged_scale_inv_for_task(
                task,
                hf_state_dict,
                source_shapes=source_shapes,
            )
            scale_inv = _canonicalize_merged_scale_inv(task, converted, scale_inv)
            scale_inv_key = build_fp8_scale_inv_key(task.param_name)
            scale_template = model_template.get(scale_inv_key)
            _validate_template_shape(scale_inv_key, scale_inv, scale_template)
            scale_template = _normalize_fp8_direct_template(
                task.param_name,
                scale_inv_key,
                scale_inv,
                scale_template,
            )
            add_tensor_entry(
                model_state,
                scale_inv_key,
                scale_inv,
                scale_template,
            )
            num_fp8 += 1
            num_scales += 1

        if (i + 1) % log_interval == 0 or (i + 1) == total_tasks:
            elapsed = time.monotonic() - t_start
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            logger.info(
                "[%d/%d] %d weights, %d scales, %d modelopt weight keys | elapsed %s | %s",
                i + 1,
                total_tasks,
                num_weights,
                num_scales,
                num_modelopt_weight_keys,
                elapsed_str,
                task.param_name,
            )

    if num_fp8 != num_scales:
        raise RuntimeError(f"Direct FP8 conversion produced {num_fp8} FP8 weights but {num_scales} scales")
    return model_state
