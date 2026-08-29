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

        description = f"Loading FP8 from {hf_pretrained.model_name_or_path}"
        for task in self._with_progress_tracking(hf_to_megatron_tasks, description):
            if task is None or task.megatron_module is None:
                continue

            # 1) Fetch source tensor(s)
            hf_weights = self.maybe_modify_loaded_hf_weight(task.mapping.hf_param, hf_state_dict)

            # 2) Detect FP8
            is_fp8 = _is_fp8(hf_weights)

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
                    # Direct assignment — no .copy_() which would cast.
                    task.param_weight.data = converted_weights
                    # Load + transform the associated scale_inv.
                    self._store_scale_inv(task, hf_state_dict)
                else:
                    task.param_weight.data.copy_(converted_weights)

        self._broadcast_shared_embeddings(megatron_model)
        return megatron_model

    def _store_scale_inv(self, task, hf_state_dict: Mapping[str, torch.Tensor]) -> None:
        """Load, merge/split, and register ``weight_scale_inv`` for one task."""
        from megatron.bridge.models.conversion.param_mapping import GatedMLPMapping, QKVMapping, RowParallelMapping
        from megatron.bridge.models.conversion.utils import get_module_and_param_from_name
        from megatron.bridge.orbit.quant.fp8_utils import merge_gated_mlp_scale_inv, merge_qkv_scale_inv

        mapping = task.mapping
        tp_rank = mapping.tp_rank
        tp_size = mapping.tp_size
        hf_param = mapping.hf_param

        # --- Load raw scale_inv(s) from HF state dict ---
        if isinstance(mapping, QKVMapping):
            q_s = _load_scale(hf_param["q"], hf_state_dict)
            k_s = _load_scale(hf_param["k"], hf_state_dict)
            v_s = _load_scale(hf_param["v"], hf_state_dict)
            if q_s is None:
                return
            config = mapping._get_config(task.megatron_module)
            merged = merge_qkv_scale_inv(config, q_s, k_s, v_s)
            # QKV is ColumnParallel -> split dim 0
            scale = _tp_chunk(merged, tp_size, tp_rank, dim=0)

        elif isinstance(mapping, GatedMLPMapping):
            gate_s = _load_scale(hf_param["gate"], hf_state_dict)
            up_s = _load_scale(hf_param["up"], hf_state_dict)
            if gate_s is None:
                return
            merged = merge_gated_mlp_scale_inv(gate_s, up_s)
            # GatedMLP is ColumnParallel -> split dim 0
            scale = _tp_chunk(merged, tp_size, tp_rank, dim=0)

        elif isinstance(hf_param, str):
            raw = _load_scale(hf_param, hf_state_dict)
            if raw is None:
                return
            if isinstance(mapping, RowParallelMapping) or _is_row_parallel(mapping):
                scale = _tp_chunk(raw, tp_size, tp_rank, dim=1)
            else:
                # ColumnParallel / Replicated
                scale = _tp_chunk(raw, tp_size, tp_rank, dim=0)
        else:
            return

        # --- Register buffer on the owning module ---
        module, _ = get_module_and_param_from_name(task.megatron_module, task.param_name)
        module.register_buffer("weight_scale_inv", scale.contiguous(), persistent=False)
        logger.debug("Registered weight_scale_inv on %s: shape=%s", task.param_name, list(scale.shape))


def _is_fp8(hf_weights) -> bool:
    """Check if the loaded HF weight(s) are FP8."""
    if isinstance(hf_weights, dict):
        return any(w.dtype == torch.float8_e4m3fn for w in hf_weights.values())
    return hf_weights.dtype == torch.float8_e4m3fn


def _load_scale(weight_key: str, hf_state_dict: Mapping[str, torch.Tensor]) -> torch.Tensor | None:
    """Load ``weight_scale_inv`` for a given weight key, or ``None``."""
    key = weight_key + "_scale_inv"
    if key in hf_state_dict:
        return hf_state_dict[key]
    # Some checkpoints use weight_scale instead of weight_scale_inv
    key2 = weight_key + "_scale"
    if key2 in hf_state_dict:
        return 1.0 / hf_state_dict[key2].float()
    return None


def _tp_chunk(tensor: torch.Tensor, tp_size: int, tp_rank: int, dim: int) -> torch.Tensor:
    """Chunk a tensor along *dim* for TP, returning this rank's shard."""
    if tp_size <= 1:
        return tensor
    return torch.chunk(tensor, tp_size, dim=dim)[tp_rank]


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
