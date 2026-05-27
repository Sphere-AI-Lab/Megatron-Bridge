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

"""INT4-preserving bridges for Qwen3 models."""

import logging
from typing import List, Mapping, Union

import torch

from megatron.bridge.models.conversion.low_precision.int4 import dequantize_int4
from megatron.bridge.models.qwen.qwen3_bridge import Qwen3Bridge
from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge

logger = logging.getLogger(__name__)


class _Qwen3CompressedTensorsINT4Mixin:
    def build_conversion_tasks(self, hf_pretrained, megatron_model) -> List:
        """Synthesize virtual ``.weight`` keys from INT4 quantized triplets."""
        original_get_all_keys = hf_pretrained.state.source.get_all_keys

        def _get_all_keys_with_virtual():
            keys = original_get_all_keys()
            all_keys_set = set(keys)
            virtual_keys = []
            for key in keys:
                if key.endswith("_packed"):
                    base = key[:-7]  # "...weight_packed" -> "...weight"
                    if f"{base}_scale" in all_keys_set and f"{base}_shape" in all_keys_set:
                        virtual_keys.append(base)
            return keys + virtual_keys

        hf_pretrained.state.source.get_all_keys = _get_all_keys_with_virtual
        try:
            return super().build_conversion_tasks(hf_pretrained, megatron_model)
        finally:
            hf_pretrained.state.source.get_all_keys = original_get_all_keys

    def maybe_modify_loaded_hf_weight(
        self,
        hf_param: Union[str, dict],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Load HF weights, dequantizing INT4 triplets when present."""
        if isinstance(hf_param, dict):
            return {
                role: self._maybe_dequant_int4(key, hf_state_dict)
                for role, key in hf_param.items()
            }
        return self._maybe_dequant_int4(hf_param, hf_state_dict)

    def _maybe_dequant_int4(
        self,
        hf_key: str,
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if hf_key.endswith(".weight"):
            base = hf_key[: -len(".weight")]
            packed_key = base + ".weight_packed"
            if packed_key in hf_state_dict:
                weight = dequantize_int4(
                    hf_state_dict[packed_key],
                    hf_state_dict[base + ".weight_scale"],
                    hf_state_dict[base + ".weight_shape"],
                    device=hf_state_dict[packed_key].device,
                )
                logger.info("Dequantised INT4 -> BF16: %s shape=%s", hf_key, list(weight.shape))
                return weight
        return hf_state_dict[hf_key]


class Qwen3INT4Bridge(_Qwen3CompressedTensorsINT4Mixin, Qwen3Bridge):
    """Dense Qwen3 bridge that preserves compressed-tensors INT4 direct writes."""


class Qwen3MoEINT4Bridge(_Qwen3CompressedTensorsINT4Mixin, Qwen3MoEBridge):
    """Qwen3 MoE bridge that preserves compressed-tensors INT4 direct writes."""
