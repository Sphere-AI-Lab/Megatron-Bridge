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

logger = logging.getLogger(__name__)

__all__ = [
    "build_int4_direct_model_state_dict",
    "convert_hf_weight_for_direct_save",
    "dequantize_int4",
    "hf_param_uses_int4",
    "quantize_to_int4",
    "register_int4_buffers_after_load_dense",
    "transform_sharded_state_dict_for_int4_dense",
]


@dataclass(frozen=True)
class _Int4Triplet:
    packed: torch.Tensor
    scale: torch.Tensor
    shape: torch.Tensor


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

    from megatron.bridge.peft.triton_oft.int4_dequant import dequantize_int4_triton

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
            f"weight_scale with {weight_scale.numel()} values cannot be grouped by "
            f"out_features={out_features}"
        )
    scale_groups = weight_scale.numel() // out_features
    if scale_groups <= 0:
        raise ValueError(f"scale_groups must be positive, got {scale_groups}")
    if in_features % scale_groups != 0:
        raise ValueError(
            f"in_features={in_features} is not divisible by scale_groups={scale_groups}"
        )
    return in_features // scale_groups


def quantize_to_int4(
    weight: torch.Tensor,
    group_size: int = 32,
    scale_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize bfloat16/float16 weights to INT4 packed format."""
    out_features, in_features = weight.shape
    weight_shape = torch.tensor([out_features, in_features], dtype=torch.int32)

    w = weight.float()
    num_groups = (in_features + group_size - 1) // group_size
    w_grouped = w.view(out_features, num_groups, -1)

    group_max = w_grouped.abs().amax(dim=-1)
    scale = group_max / 7.0
    scale = scale.clamp(min=1e-10)

    scale_expanded = scale.unsqueeze(-1).expand_as(w_grouped)
    w_q = (w_grouped / scale_expanded).round().clamp(-8, 7)
    w_q = w_q.view(out_features, -1)[:, :in_features]
    w_q = (w_q + 8).to(torch.uint8)

    assert in_features % 8 == 0, f"in_features must be divisible by 8, got {in_features}"

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


def hf_weight_has_int4_triplet(
    weight_key: str, hf_state_dict: Mapping[str, torch.Tensor]
) -> bool:
    if not weight_key.endswith(".weight"):
        return False
    base = weight_key[: -len(".weight")]
    return f"{base}.weight_packed" in hf_state_dict


def hf_param_uses_int4(
    hf_param: Any, hf_state_dict: Mapping[str, torch.Tensor]
) -> bool:
    if isinstance(hf_param, str):
        return hf_weight_has_int4_triplet(hf_param, hf_state_dict)
    if isinstance(hf_param, dict):
        return any(hf_param_uses_int4(value, hf_state_dict) for value in hf_param.values())
    return False


def convert_hf_weight_for_direct_save(task: Any, hf_weights: Any) -> torch.Tensor:
    """HF -> Megatron conversion for the TP=1 direct-save path."""
    mapping = task.mapping

    if getattr(mapping, "tp_size", 1) != 1:
        raise ValueError(
            "Direct INT4 converter currently supports only single-rank TP=1 checkpoint writes."
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


def _load_hf_int4_triplet(weight_key: str, hf_state_dict: Mapping[str, torch.Tensor]) -> _Int4Triplet | None:
    if not weight_key.endswith(".weight"):
        return None
    base = weight_key[: -len(".weight")]
    packed_key = base + ".weight_packed"
    scale_key = base + ".weight_scale"
    shape_key = base + ".weight_shape"
    if packed_key not in hf_state_dict or scale_key not in hf_state_dict or shape_key not in hf_state_dict:
        return None
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
        triplets = {
            role: _load_hf_int4_triplets(key, hf_state_dict)
            for role, key in hf_param.items()
        }
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
        raise ValueError(
            "Direct INT4 converter currently supports only single-rank TP=1 checkpoint writes."
        )

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
) -> dict[str, Any]:
    """Create the prebuilt ``state_dict['model']`` for direct INT4 checkpoint save."""
    if len(meta_model) != 1:
        raise ValueError(
            "Direct INT4 converter currently supports a single Megatron model chunk (no VP stages)."
        )

    model_state = prepare_empty_model_state(model_template)
    conversion_tasks = int4_bridge.build_conversion_tasks(hf_pretrained, meta_model)
    hf_state_dict: Mapping[str, torch.Tensor] = hf_pretrained.state

    num_regular = 0
    num_int4 = 0
    total_tasks = len(conversion_tasks)
    log_interval = max(1, total_tasks // 100)
    t_start = time.monotonic()

    for i, task in enumerate(conversion_tasks):
        if task is None or task.megatron_module is None:
            continue

        if hf_param_uses_int4(task.mapping.hf_param, hf_state_dict):
            hf_triplets = _load_hf_int4_triplets(task.mapping.hf_param, hf_state_dict)
            converted_triplet = _convert_hf_int4_triplet_for_direct_save(task, hf_triplets)
            if converted_triplet is None:
                raise RuntimeError(
                    "Cannot preserve INT4 triplets without dequantizing: "
                    f"param={task.param_name} "
                    f"mapping={type(task.mapping).__name__} "
                    f"hf_param={task.mapping.hf_param!r}. "
                    "Add a packed/scale/shape-preserving handler for this mapping."
                )

            packed = converted_triplet.packed
            scale = converted_triplet.scale
            shape = converted_triplet.shape

            add_tensor_entry(model_state, f"{task.param_name}_packed", packed)
            add_tensor_entry(model_state, f"{task.param_name}_scale", scale)
            add_tensor_entry(model_state, f"{task.param_name}_shape", shape)
            num_int4 += 1
        else:
            hf_weights = int4_bridge.maybe_modify_loaded_hf_weight(
                task.mapping.hf_param,
                hf_state_dict,
            )
            converted = convert_hf_weight_for_direct_save(task, hf_weights)
            add_tensor_entry(model_state, task.param_name, converted)
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
    return model_state


# -------------------------------------------------------------------------- #
# Dense-model dist-checkpoint load helpers
# (Llama INT4 compressed-tensors / pack-quantized)
# -------------------------------------------------------------------------- #

# Dense Megatron linear weight keys: attention QKV / output proj + gated MLP
# fc1 / fc2. Matches the names produced by the Llama bridge's mapping registry
# for GPTModel (`decoder.layers.{i}.self_attention.linear_qkv.weight`, etc.).
_DENSE_LINEAR_WEIGHT_RE = re.compile(
    r"^(.*\.(?:linear_qkv|linear_proj|linear_fc1|linear_fc2|router))\.weight$"
)
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

    Parallels the expert-only ``peft.int4_utils.transform_sharded_state_dict_for_int4``
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
            built_sub = value.build()
            if isinstance(built_sub, list):
                sub_sh_ten = built_sub[0]
                sub_sh_tens = built_sub
            else:
                sub_sh_ten = next(iter(built_sub.values()))
                sub_sh_tens = list(built_sub.values())

            # Restore fused SwiGLU metadata so the checkpoint loader sees the
            # original linear_fc1 tensor rather than the gate/up split halves.
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
                if axis_fragmentations_override[out_axis] % split_factor != 0:
                    raise ValueError(
                        f"Unexpected SwiGLU factory fragmentation for {key}: "
                        f"{axis_fragmentations_override[out_axis]} not divisible by {split_factor}"
                    )
                axis_fragmentations_override[out_axis] //= split_factor
                global_out_override = sh_ten.global_shape[out_axis]
            else:
                global_out_override = sh_ten.global_shape[out_axis] * split_factor

            local_out_override = fused_local_out
            out_offset_override = rank_offset * fused_local_out
            axis_fragmentations_override = tuple(axis_fragmentations_override)
        else:
            sh_ten = value
            local_out_override = None
            global_out_override = None
            out_offset_override = None
            axis_fragmentations_override = sh_ten.axis_fragmentations
        # Megatron emits linear weights in one of two layouts:
        #   - stacked: key like "decoder.layers.self_attention.linear_proj.weight"
        #     with prepend_axis_num=1, global_shape=(num_layers, out, in),
        #     local_shape=(num_local_layers, out, in). One entry covers every
        #     layer on this rank.
        #   - per-layer: key like "decoder.layers.0.self_attention.linear_proj.weight"
        #     with prepend_axis_num=0, global_shape=(out, in).
        # The converter writes per-layer triplets on disk, so the stacked
        # template has to be unrolled into one triplet set per layer index
        # (analogous to how the expert transform unrolls the EP axis).
        prepend = sh_ten.prepend_axis_num
        local_out = local_out_override if local_out_override is not None else sh_ten.local_shape[-2]
        local_in = sh_ten.local_shape[-1]
        global_out = global_out_override if global_out_override is not None else sh_ten.global_shape[-2]
        global_in = sh_ten.global_shape[-1]

        if local_in % 8 != 0:
            raise ValueError(
                f"INT4 requires in-features divisible by 8, got {local_in} for {key}"
            )
        if local_in % group_size != 0:
            raise ValueError(
                f"INT4 requires in-features divisible by group_size={group_size}, "
                f"got {local_in} for {key}"
            )

        packed_in = local_in // 8
        global_packed_in = global_in // 8
        num_groups = local_in // group_size
        global_num_groups = global_in // group_size

        weight_axis_fragmentations = tuple(axis_fragmentations_override[prepend:])
        out_offset_full = (
            out_offset_override
            if out_offset_override is not None
            else (
                sh_ten.global_offset[prepend]
                if len(sh_ten.global_offset) > prepend
                else 0
            )
        )
        in_offset_full = (
            sh_ten.global_offset[prepend + 1]
            if len(sh_ten.global_offset) > prepend + 1
            else 0
        )

        has_explicit_layer_idx = _EXPLICIT_LAYER_KEY_RE.search(key) is not None
        if prepend > 0 and not has_explicit_layer_idx:
            # Stacked layers: emit one per-layer triplet set per local layer slot.
            num_local_layers = sh_ten.local_shape[0]
            layer_start = sh_ten.global_offset[prepend - 1]
            layer_indices = [layer_start + i for i in range(num_local_layers)]
        else:
            layer_indices = [None]

        for global_layer_idx in layer_indices:
            if global_layer_idx is None:
                ckpt_key = key
            else:
                ckpt_key = re.sub(
                    r"(^|\.)layers\.", rf"\1layers.{global_layer_idx}.", key, count=1
                )

            triplets = [
                ("_packed", (local_out, packed_in), (global_out, global_packed_in),
                 (out_offset_full, in_offset_full // 8), torch.int32),
                ("_scale", (local_out, num_groups), (global_out, global_num_groups),
                 (out_offset_full, in_offset_full // group_size), scale_dtype),
                ("_shape", (2,), (2,), (0,), torch.int64),
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
                axis_frags = (
                    weight_axis_fragmentations
                    if suffix != "_shape"
                    else (1,)
                )
                new_sd[ckpt_key + suffix] = ShardedTensor(
                    key=ckpt_key + suffix,
                    data=data,
                    dtype=dtype,
                    local_shape=local_sh,
                    global_shape=global_sh,
                    global_offset=off,
                    axis_fragmentations=axis_frags,
                    replica_id=(
                        _replica_id_with_current_tp_rank(sh_ten.replica_id)
                        if suffix == "_shape"
                        else sh_ten.replica_id
                    ),
                    prepend_axis_num=0,
                    allow_shape_mismatch=(suffix == "_shape"),
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
