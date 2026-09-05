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

"""INT4-preserving bridge for DeepSeek-V3 / Kimi-K2 models.

Handles HuggingFace checkpoints where expert weights are stored in Kimi-K2
native INT4 format (``weight_packed`` + ``weight_scale`` + ``weight_shape``)
while attention / norm weights are BF16.

Conversion flow:
    1. The architecture-independent
       :class:`~megatron.bridge.orbit.conversion.compressed_tensors_int4.CompressedTensorsINT4DequantMixin`
       synthesizes virtual ``.weight`` keys and dequants INT4 -> BF16 on read
       so the standard bridge can do QKV merge + TP split.
    2. After all weights are loaded, ``_requantize_experts_int4`` walks the
       model and re-quantises expert weights BF16 -> INT4.  The BF16 Parameter
       data is zeroed and ``weight_packed`` / ``weight_scale`` / ``weight_shape``
       are registered as persistent buffers.
    3. ``save_megatron_model`` saves everything — dist_checkpointing preserves
       int32 buffers and bf16 buffers.

At training time, OFTLinear detects the INT4 buffers and dequants on the fly.
"""

import logging

import torch
import torch.nn as nn

from megatron.bridge.models.conversion.utils import unwrap_model
from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge
from megatron.bridge.models.kimi_vl.utils import quantize_to_int4
from megatron.bridge.orbit.conversion.compressed_tensors_int4 import CompressedTensorsINT4DequantMixin
from megatron.bridge.orbit.low_precision.int4 import (
    _convert_hf_int4_triplet_for_direct_save,
    _load_hf_int4_triplets,
    _validate_hf_int4_triplets,
    _validate_int4_triplet,
    requantize_int4_with_scales,
)


logger = logging.getLogger(__name__)


class DeepSeekV3INT4Bridge(CompressedTensorsINT4DequantMixin, DeepSeekV3Bridge):
    """DeepSeek-V3 / Kimi-K2 bridge that preserves INT4 through conversion."""

    # --------------------------------------------------------------------- #
    # Re-quantise expert weights after loading, before save
    # --------------------------------------------------------------------- #

    def load_weights_hf_to_megatron(
        self,
        hf_pretrained,
        megatron_model,
        allowed_mismatched_params: list[str] | None = None,
    ):
        """Load weights normally (with INT4 dequant), then re-quantise experts."""
        models = list(megatron_model) if isinstance(megatron_model, (list, tuple)) else [megatron_model]
        if len(models) != 1:
            raise ValueError(
                "DeepSeek INT4 conversion supports a single Megatron model chunk; "
                "virtual pipeline parallelism is not supported"
            )

        unwrapped_model = unwrap_model(models)[0]
        model_config = getattr(unwrapped_model, "config", None)
        expert_tensor_parallel_size = getattr(model_config, "expert_tensor_parallel_size", 1)
        if expert_tensor_parallel_size is None:
            expert_tensor_parallel_size = 1
        if expert_tensor_parallel_size != 1:
            raise ValueError(
                f"DeepSeek INT4 conversion requires expert tensor parallel size 1; got {expert_tensor_parallel_size}"
            )

        conversion_tasks = self.build_conversion_tasks(hf_pretrained, models)
        source_scales = self._collect_source_scales(
            hf_pretrained,
            models,
            conversion_tasks=conversion_tasks,
        )
        result = super().load_weights_hf_to_megatron(hf_pretrained, models, allowed_mismatched_params)
        # Re-quantise expert linear weights to INT4, reusing the checkpoint's
        # own per-group scales so the integers round back bitwise.
        self._requantize_experts_int4(unwrapped_model, source_scales=source_scales)
        return result

    def _collect_source_scales(
        self,
        hf_pretrained,
        megatron_model: list,
        *,
        conversion_tasks=None,
    ) -> dict[str, torch.Tensor]:
        """Map Megatron param names to their merged SOURCE scale tensors.

        Recomputing scales as ``amax / 7`` re-grids every group whose source
        used the valid INT4 value ``-8`` (``quantize_to_int4`` can never emit
        it). Reusing the checkpoint's scales keeps re-quantization bitwise.
        Rows are merged in the same order as the weights via the
        triplet-preserving direct-save converter.

        The expert checkpoint load schema declares FP16 scales. Reject other
        source dtypes here rather than casting them onto a different grid.
        """
        hf_state_dict = hf_pretrained.state
        scales: dict[str, torch.Tensor] = {}
        if conversion_tasks is None:
            conversion_tasks = self.build_conversion_tasks(hf_pretrained, megatron_model)
        for task in conversion_tasks:
            if task is None or task.megatron_module is None:
                continue
            if ".experts." not in task.param_name or ".weight" not in task.param_name:
                continue
            triplets = _load_hf_int4_triplets(task.mapping.hf_param, hf_state_dict)
            if triplets is None:
                raise ValueError(
                    "DeepSeek expert weight has a missing or unsupported INT4 triplet: "
                    f"megatron_param={task.param_name}, hf_param={task.mapping.hf_param!r}"
                )
            _validate_hf_int4_triplets(
                task.mapping.hf_param,
                triplets,
                group_size=32,
                fallback_key=task.param_name,
            )
            merged = _convert_hf_int4_triplet_for_direct_save(task, triplets)
            if merged is None:
                raise ValueError(
                    "DeepSeek expert weight has a missing or unsupported INT4 triplet mapping: "
                    f"megatron_param={task.param_name}, mapping={type(task.mapping).__name__}"
                )
            _validate_int4_triplet(merged, group_size=32, key=task.param_name)
            if merged.scale.dtype != torch.float16:
                raise ValueError(
                    "DeepSeek INT4 expert weight_scale must be torch.float16 to match the strict "
                    f"checkpoint load schema; got {merged.scale.dtype} for {task.param_name!r}"
                )
            scales[task.param_name] = merged.scale
        return scales

    @torch.no_grad()
    def _requantize_experts_int4(
        self,
        model: nn.Module,
        group_size: int = 32,
        source_scales: dict[str, torch.Tensor] | None = None,
    ):
        """Walk the model and re-quantise expert weights BF16 -> INT4.

        For each expert linear (grouped or regular), replace the BF16 weight
        data with INT4 packed buffers.
        """
        total_saved = 0

        for name, module in model.named_modules():
            # Grouped expert linears (TEGroupedLinear): weight0, weight1, ...
            num_gemms = getattr(module, "num_gemms", 0)
            if num_gemms > 0:
                for idx in range(num_gemms):
                    w_name = f"weight{idx}"
                    w = getattr(module, w_name, None)
                    if w is None or w.dtype != torch.bfloat16:
                        continue
                    w_data = w.data if isinstance(w, nn.Parameter) else w
                    if w_data.ndim != 2:
                        continue
                    saved = self._quantize_one_weight(
                        module,
                        w_name,
                        w_data,
                        group_size,
                        source_scale=(source_scales or {}).get(f"{name}.{w_name}"),
                    )
                    total_saved += saved
                continue

            # Regular expert linears (under .experts. path)
            if "expert" not in name.lower():
                continue
            w = getattr(module, "weight", None)
            if w is None or not isinstance(w, nn.Parameter):
                continue
            if w.dtype != torch.bfloat16 or w.ndim != 2:
                continue
            saved = self._quantize_one_weight(
                module,
                "weight",
                w.data,
                group_size,
                source_scale=(source_scales or {}).get(f"{name}.weight"),
            )
            total_saved += saved

        logger.info("Re-quantised experts to INT4, saved %.1f GB", total_saved / 1e9)

    def _quantize_one_weight(
        self,
        module: nn.Module,
        weight_name: str,
        w_data: torch.Tensor,
        group_size: int,
        source_scale: torch.Tensor | None = None,
    ) -> int:
        """Quantise one weight to INT4 and register buffers on the module."""
        bf16_bytes = w_data.numel() * w_data.element_size()

        if source_scale is not None:
            packed, scale, shape = requantize_int4_with_scales(w_data, source_scale.to(w_data.device))
        else:
            logger.warning(
                "No source scales for %s; re-quantizing with recomputed amax/7 scales "
                "(groups whose source used -8 will be re-gridded)",
                weight_name,
            )
            packed, scale, shape = quantize_to_int4(
                w_data,
                group_size=group_size,
                scale_dtype=torch.float16,
            )

        # Register INT4 tensors as persistent buffers (saved by dist_checkpointing).
        module.register_buffer(f"{weight_name}_packed", packed, persistent=True)
        module.register_buffer(f"{weight_name}_scale", scale, persistent=True)
        module.register_buffer(f"{weight_name}_shape", shape, persistent=True)

        # Replace BF16 weight with zeros of the same shape.  We can't change the
        # shape (breaks sharded_state_dict) or delete it (breaks state_dict).
        # The disk overhead is acceptable for now — the INT4 buffers are the
        # source of truth and OFTLinear ignores this placeholder at training time.
        # TODO: hook into save to skip these placeholder weights.
        getattr(module, weight_name).data = torch.zeros_like(w_data)

        int4_bytes = packed.numel() * packed.element_size() + scale.numel() * scale.element_size()
        return bf16_bytes - int4_bytes
