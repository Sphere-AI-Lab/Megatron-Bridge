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

"""INT4-preserving bridge for DeepSeek-V3 / Kimi-K2 models.

Handles HuggingFace checkpoints where expert weights are stored in Kimi-K2
native INT4 format (``weight_packed`` + ``weight_scale`` + ``weight_shape``)
while attention / norm weights are BF16.

Conversion flow:
    1. ``maybe_modify_loaded_hf_weight`` intercepts INT4 weights and dequants
       to BF16 so the standard bridge can do QKV merge + TP split.
    2. After all weights are loaded, ``_requantize_experts_int4`` walks the
       model and re-quantises expert weights BF16 → INT4.  The BF16 Parameter
       data is emptied and ``weight_packed`` / ``weight_scale`` / ``weight_shape``
       are registered as persistent buffers.
    3. ``save_megatron_model`` saves everything — dist_checkpointing preserves
       int32 buffers and bf16 buffers.

At training time, OFTLinear detects the INT4 buffers and dequants on the fly.
"""

import logging
from typing import List, Mapping, Optional, Union

import torch
import torch.nn as nn

from megatron.bridge.orbit.low_precision.int4 import (
    dequantize_int4,
    quantize_to_int4,
)
from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge

logger = logging.getLogger(__name__)


class DeepSeekV3INT4Bridge(DeepSeekV3Bridge):
    """DeepSeek-V3 / Kimi-K2 bridge that preserves INT4 through conversion."""

    # --------------------------------------------------------------------- #
    # Virtual key synthesis: make INT4 triplets visible as .weight keys
    # --------------------------------------------------------------------- #

    def build_conversion_tasks(self, hf_pretrained, megatron_model) -> List:
        """Synthesize virtual 'weight' keys from INT4 quantized triplets.

        The HF checkpoint stores quantized expert weights as triplets
        (weight_packed, weight_scale, weight_shape) without a plain 'weight'
        key.  We synthesize virtual 'weight' keys so the mapping registry
        can find them, then maybe_modify_loaded_hf_weight handles dequant.
        """
        original_get_all_keys = hf_pretrained.state.source.get_all_keys

        def _get_all_keys_with_virtual():
            keys = original_get_all_keys()
            all_keys_set = set(keys)
            virtual_keys = []
            for key in keys:
                if key.endswith("_packed"):
                    base = key[:-7]  # "...weight_packed" → "...weight"
                    if f"{base}_scale" in all_keys_set and f"{base}_shape" in all_keys_set:
                        virtual_keys.append(base)
            return keys + virtual_keys

        hf_pretrained.state.source.get_all_keys = _get_all_keys_with_virtual
        try:
            return super().build_conversion_tasks(hf_pretrained, megatron_model)
        finally:
            hf_pretrained.state.source.get_all_keys = original_get_all_keys

    # --------------------------------------------------------------------- #
    # Step 1: Dequant INT4 → BF16 during HF weight loading
    # --------------------------------------------------------------------- #

    def maybe_modify_loaded_hf_weight(
        self, hf_param: Union[str, dict], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(hf_param, dict):
            return {role: self._maybe_dequant_int4(key, hf_state_dict)
                    for role, key in hf_param.items()}
        return self._maybe_dequant_int4(hf_param, hf_state_dict)

    def _maybe_dequant_int4(
        self, hf_key: str, hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        if hf_key.endswith(".weight"):
            base = hf_key[: -len(".weight")]
            packed_key = base + ".weight_packed"
            if packed_key in hf_state_dict:
                w = dequantize_int4(
                    hf_state_dict[packed_key],
                    hf_state_dict[base + ".weight_scale"],
                    hf_state_dict[base + ".weight_shape"],
                    device=hf_state_dict[packed_key].device,
                )
                logger.info("Dequantised INT4 → BF16: %s  shape=%s", hf_key, list(w.shape))
                return w
        return hf_state_dict[hf_key]

    # --------------------------------------------------------------------- #
    # Step 2: Re-quantise expert weights after loading, before save
    # --------------------------------------------------------------------- #

    def load_weights_hf_to_megatron(
        self,
        hf_pretrained,
        megatron_model,
        allowed_mismatched_params: Optional[List[str]] = None,
    ):
        """Load weights normally (with INT4 dequant), then re-quantise experts."""
        result = super().load_weights_hf_to_megatron(
            hf_pretrained, megatron_model, allowed_mismatched_params
        )
        # Re-quantise expert linear weights to INT4.
        models = megatron_model if isinstance(megatron_model, list) else [megatron_model]
        for model in models:
            self._requantize_experts_int4(model)
        return result

    @torch.no_grad()
    def _requantize_experts_int4(self, model: nn.Module, group_size: int = 32):
        """Walk the model and re-quantise expert weights BF16 → INT4.

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
                    saved = self._quantize_one_weight(module, w_name, w_data, group_size)
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
            saved = self._quantize_one_weight(module, "weight", w.data, group_size)
            total_saved += saved

        logger.info("Re-quantised experts to INT4, saved %.1f GB", total_saved / 1e9)

    def _quantize_one_weight(
        self, module: nn.Module, weight_name: str, w_data: torch.Tensor, group_size: int
    ) -> int:
        """Quantise one weight to INT4 and register buffers on the module."""
        bf16_bytes = w_data.numel() * w_data.element_size()

        packed, scale, shape = quantize_to_int4(w_data, group_size=group_size)

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

    @staticmethod
    @torch.no_grad()
    def deregister_quantized_weights(model: nn.Module):
        """Remove empty weight Parameters that were replaced by INT4 buffers.

        Must be called before ``sharded_state_dict()`` / checkpoint save.
        After this, the model is no longer usable for forward — only for saving.
        """
        to_delete = []  # collect (module, param_name) to delete after iteration
        for name, module in model.named_modules():
            num_gemms = getattr(module, "num_gemms", 0)
            if num_gemms > 0:
                for idx in range(num_gemms):
                    w_name = f"weight{idx}"
                    if hasattr(module, f"{w_name}_packed"):
                        to_delete.append((module, w_name))
            elif hasattr(module, "weight_packed"):
                to_delete.append((module, "weight"))

        for module, w_name in to_delete:
            if w_name in module._parameters:
                del module._parameters[w_name]

        logger.info("Deregistered %d empty weight Parameters before save", len(to_delete))
