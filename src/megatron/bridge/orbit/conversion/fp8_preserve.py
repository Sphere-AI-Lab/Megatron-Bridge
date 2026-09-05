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

"""Generic FP8-preserving load adapter for HF -> Megatron conversion.

Block-FP8 HF checkpoints (``quant_method: fp8``, e4m3fn weights with
``weight_scale_inv`` siblings) are dequantized to BF16 by the standard load
path, because ``ColumnParallelMapping.hf_to_megatron`` casts to the Megatron
param dtype and ``scatter_to_tp_ranks`` allocates a BF16 output buffer.

:class:`BlockFP8PreserveMixin` keeps FP8 weights in FP8 end to end by temporarily
setting the target parameter to FP8 dtype before the mapping runs — ``chunk``,
``cat``, ``view`` and ``reshape`` are dtype-agnostic, so QKV merge, gate/up
concat and TP split work unchanged on FP8 tensors. The matching
``weight_scale_inv`` tensors are merged/split alongside, keyed purely by
mapping type (QKV / GatedMLP / row-parallel / other), and registered as
buffers on the owning module.

Nothing in this module is model-specific; it composes with any registered
bridge (mixin-first MRO) via the shared factories.

Why preserve instead of dequantize (format rationale):
    Block-FP8 stores one ``float8_e4m3fn`` byte per weight element — the
    tensor is element-addressable, so ``cat`` / ``chunk`` / interleave run
    directly on the stored bytes, and the per-``[128, 128]``-block
    ``scale_inv`` tensors follow their rows/columns through the same cuts.
    Dequantizing here would only cost:

    1. memory — a BF16 intermediate of the whole model is exactly what
       this path exists to avoid;
    2. fidelity — re-quantizing after the merge recomputes block scales,
       producing a re-encoded model rather than the published bytes;
    3. novelty — FP8 -> BF16 dequant is already the upstream default load
       behavior, which this mixin exists to escape.

    Contrast with the packed formats (compressed-tensors INT4, ModelOpt
    NVFP4): those are not element-addressable and must pass through a BF16
    transit form — see ``compressed_tensors_int4`` and ``modelopt_nvfp4``.
"""

import contextlib
import logging
from collections.abc import Mapping

import torch


logger = logging.getLogger(__name__)


class BlockFP8PreserveMixin:
    """Architecture-independent FP8-preserving weight load for model bridges."""

    def load_weights_hf_to_megatron(
        self,
        hf_pretrained,
        megatron_model,
        allowed_mismatched_params: list[str] | None = None,
    ):
        """Load HF weights into Megatron, keeping FP8 tensors in FP8."""
        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        with contextlib.ExitStack() as stack:
            if hasattr(megatron_model[0], "hide_teacher_model"):
                stack.enter_context(megatron_model[0].hide_teacher_model())
            if hasattr(megatron_model[0], "hide_loss_modules"):
                stack.enter_context(megatron_model[0].hide_loss_modules())
            hf_to_megatron_tasks = self.build_conversion_tasks(hf_pretrained, megatron_model)

        hf_state_dict: Mapping[str, torch.Tensor] = hf_pretrained.state if hasattr(hf_pretrained, "state") else {}
        from megatron.bridge.orbit.low_precision.fp8 import preflight_fp8_conversion_tasks

        fp8_plan = preflight_fp8_conversion_tasks(hf_to_megatron_tasks, hf_state_dict)
        source_shapes = {}
        for task in hf_to_megatron_tasks:
            if task is None:
                continue
            hf_param = task.mapping.hf_param
            source_keys = (hf_param,) if isinstance(hf_param, str) else tuple(hf_param.values())
            for source_key in source_keys:
                source_shape = fp8_plan.source_shape(source_key)
                if source_shape is not None:
                    source_shapes[source_key] = source_shape
        prevalidated_scales = {}
        for task in hf_to_megatron_tasks:
            if task is None or id(task) not in fp8_plan.fp8_task_ids:
                continue
            if task.param_weight is None:
                raise RuntimeError(f"FP8 conversion task {task.param_name!r} has no target parameter")
            prevalidated_scales[id(task)] = _prepare_scale_inv(
                task,
                hf_state_dict,
                source_shapes=source_shapes,
            )

        description = f"Loading FP8 from {hf_pretrained.model_name_or_path}"
        for task in self._with_progress_tracking(hf_to_megatron_tasks, description):
            if task is None or task.megatron_module is None:
                continue

            # 1) Fetch source tensor(s)
            hf_weights = self.maybe_modify_loaded_hf_weight(task.mapping.hf_param, hf_state_dict)

            # 2) Enforce the preflight family classification after bridge hooks.
            is_fp8 = id(task) in fp8_plan.fp8_task_ids
            loaded_is_fp8 = _is_fp8(hf_weights)
            if loaded_is_fp8 != is_fp8:
                raise TypeError(
                    f"Loaded dtype for {task.param_name!r} disagrees with the prevalidated FP8 source family"
                )

            if is_fp8 and task.param_weight is not None:
                # --- FP8 path: set target param to FP8 so mapping won't cast ---
                task.param_weight.data = torch.empty(
                    task.param_weight.shape,
                    dtype=torch.float8_e4m3fn,
                    device=task.param_weight.device,
                )

            # 3) Run mapping (QKV merge, TP split — dtype-agnostic)
            converted_weights = task.mapping.hf_to_megatron(hf_weights, task.megatron_module)

            # 4) Store result
            if converted_weights is not None:
                assert task.param_weight is not None

                if converted_weights.shape != task.param_weight.shape:
                    raise ValueError(
                        f"Shape mismatch for {task.mapping.megatron_param}: "
                        f"expected {task.param_weight.shape}, got {converted_weights.shape}"
                    )

                if is_fp8:
                    if converted_weights.dtype != torch.float8_e4m3fn:
                        raise TypeError(
                            f"FP8 source family for {task.param_name!r} converted to {converted_weights.dtype}"
                        )
                    # Direct assignment — no .copy_() which would cast.
                    task.param_weight.data = converted_weights
                    # Load + transform the associated scale_inv.
                    self._store_scale_inv(
                        task,
                        hf_state_dict,
                        source_shapes=source_shapes,
                        prevalidated_scale=prevalidated_scales[id(task)],
                    )
                else:
                    if converted_weights.dtype == torch.float8_e4m3fn:
                        raise TypeError(f"Non-FP8 source family for {task.param_name!r} unexpectedly converted to FP8")
                    task.param_weight.data.copy_(converted_weights)

        self._broadcast_shared_embeddings(megatron_model)
        return megatron_model

    def _store_scale_inv(
        self,
        task,
        hf_state_dict: Mapping[str, torch.Tensor],
        *,
        source_shapes: Mapping[str, tuple[int, ...]] | None = None,
        prevalidated_scale: torch.Tensor | None = None,
    ) -> None:
        """Load, merge/split, and register ``weight_scale_inv`` for one task."""
        from megatron.bridge.models.conversion.utils import get_module_and_param_from_name

        scale = prevalidated_scale
        if scale is None:
            scale = _prepare_scale_inv(task, hf_state_dict, source_shapes=source_shapes)
        scale = scale.to(device=task.param_weight.device)

        # --- Register buffer on the owning module ---
        module, _ = get_module_and_param_from_name(task.megatron_module, task.param_name)
        weight_name = task.param_name.rsplit(".", 1)[-1]
        scale_name = f"{weight_name}_scale_inv"
        module.register_buffer(scale_name, scale.contiguous(), persistent=True)
        logger.debug("Registered %s on %s: shape=%s", scale_name, task.param_name, list(scale.shape))


def _prepare_scale_inv(
    task,
    hf_state_dict: Mapping[str, torch.Tensor],
    *,
    source_shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> torch.Tensor:
    """Build and validate one task's rank-local FP8 scale before loading weights."""
    from megatron.bridge.models.conversion.param_mapping import GatedMLPMapping, QKVMapping, RowParallelMapping
    from megatron.bridge.orbit.low_precision.fp8 import (
        _canonicalize_merged_scale_inv,
        build_merged_scale_inv_for_task,
    )

    mapping = task.mapping
    tp_rank = mapping.tp_rank
    tp_size = mapping.tp_size
    merged = build_merged_scale_inv_for_task(
        task,
        hf_state_dict,
        source_shapes=source_shapes,
    )

    if isinstance(mapping, (QKVMapping, GatedMLPMapping)):
        shard_dim = 0
    elif isinstance(mapping.hf_param, str):
        shard_dim = 1 if isinstance(mapping, RowParallelMapping) or _is_row_parallel(mapping) else 0
    else:
        raise RuntimeError(f"Unsupported FP8 scale mapping for {task.param_name!r}")

    _validate_tp_scale_chunk(task, tp_size, tp_rank, dim=shard_dim)
    scale = _tp_chunk(merged, tp_size, tp_rank, dim=shard_dim)
    return _canonicalize_merged_scale_inv(task, task.param_weight, scale)


def _is_fp8(hf_weights) -> bool:
    """Check if the loaded HF weight(s) are FP8."""
    if isinstance(hf_weights, dict):
        if not all(isinstance(weight, torch.Tensor) for weight in hf_weights.values()):
            raise TypeError("All members of a fused FP8 source family must be tensors")
        source_is_fp8 = [weight.dtype == torch.float8_e4m3fn for weight in hf_weights.values()]
        if any(source_is_fp8) and not all(source_is_fp8):
            raise ValueError("FP8 source family mixes E4M3FN and non-FP8 tensors")
        return bool(source_is_fp8) and all(source_is_fp8)
    if not isinstance(hf_weights, torch.Tensor):
        raise TypeError(f"FP8 source weight must be a tensor, got {type(hf_weights).__name__}")
    return hf_weights.dtype == torch.float8_e4m3fn


def _load_scale(weight_key: str, hf_state_dict: Mapping[str, torch.Tensor]) -> torch.Tensor | None:
    """Load and validate ``weight_scale_inv`` for a given weight key."""
    from megatron.bridge.orbit.low_precision.fp8 import _load_scale as _load_validated_scale

    return _load_validated_scale(weight_key, hf_state_dict)


def _tp_chunk(tensor: torch.Tensor, tp_size: int, tp_rank: int, dim: int) -> torch.Tensor:
    """Chunk a tensor along *dim* for TP, returning this rank's shard."""
    if tp_size <= 1:
        return tensor
    return torch.chunk(tensor, tp_size, dim=dim)[tp_rank]


def _validate_tp_scale_chunk(task, tp_size: int, tp_rank: int, dim: int) -> None:
    """Reject TP weight shards whose boundary bisects a 128-value FP8 block."""
    if tp_size <= 1:
        return

    from megatron.bridge.orbit.quant.fp8_utils import (
        FP8_WEIGHT_BLOCK_SIZE,
        _validate_fp8_scale_shard_boundaries,
    )

    local_shape = tuple(int(size) for size in task.param_weight.shape)
    if dim >= len(local_shape):
        raise ValueError(f"Cannot shard FP8 weight with shape {local_shape} along dimension {dim}")
    global_shape = list(local_shape)
    global_shape[dim] *= tp_size
    global_offset = [0] * len(local_shape)
    global_offset[dim] = local_shape[dim] * tp_rank
    axis_fragmentations = [1] * len(local_shape)
    axis_fragmentations[dim] = tp_size
    _validate_fp8_scale_shard_boundaries(
        local_shape,
        tuple(global_shape),
        tuple(global_offset),
        tuple(axis_fragmentations),
        FP8_WEIGHT_BLOCK_SIZE,
    )


def _is_row_parallel(mapping) -> bool:
    """Heuristic: check if an AutoMapping resolved to RowParallel."""
    from megatron.bridge.models.conversion.param_mapping import RowParallelMapping

    inner = getattr(mapping, "_mapping", None)
    return isinstance(inner, RowParallelMapping)


def fp8_bridge_class_for(base_cls: type, *, extra_mixins: tuple[type, ...] = ()) -> type:
    """Compose :class:`BlockFP8PreserveMixin` with a bridge class, cached."""
    from megatron.bridge.orbit.conversion.bridge_compose import quant_bridge_class_for

    return quant_bridge_class_for(BlockFP8PreserveMixin, base_cls, extra_mixins=extra_mixins, name_prefix="FP8")


def fp8_bridge_for(auto_bridge):
    """Return the registered bridge for the architecture, FP8-preserve-composed."""
    from megatron.bridge.orbit.conversion.bridge_compose import quant_bridge_for

    return quant_bridge_for(BlockFP8PreserveMixin, auto_bridge, name_prefix="FP8")
