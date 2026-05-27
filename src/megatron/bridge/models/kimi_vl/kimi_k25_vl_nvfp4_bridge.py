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

"""NVFP4-preserving bridge for Kimi-K2.5 VL checkpoints."""

from __future__ import annotations

from typing import Dict, Mapping

import torch

from megatron.bridge.models.conversion.low_precision.nvfp4 import (
    dequantize_nvfp4,
    quantize_to_nvfp4,
)
from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
from megatron.bridge.models.kimi_vl.kimi_k25_vl_bridge import KimiK25VLBridge


class KimiK25VLNVFP4Bridge(KimiK25VLBridge):
    """Kimi-K2.5 bridge variant that consumes and emits NVFP4 expert bundles."""

    def _load_and_dequantize(self, key: str, hf_state_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Load a weight, dequantizing ModelOpt NVFP4 packed tensors when present."""
        scale_key = f"{key}_scale"
        scale_2_key = f"{key}_scale_2"
        if key.endswith(".weight") and scale_key in hf_state_dict and scale_2_key in hf_state_dict:
            packed = hf_state_dict[key]
            shape = torch.tensor(
                [packed.shape[0], packed.shape[1] * 2],
                dtype=torch.int64,
                device=packed.device,
            )
            return dequantize_nvfp4(
                packed,
                hf_state_dict[scale_key],
                hf_state_dict[scale_2_key],
                shape,
                dtype=torch.bfloat16,
                device=packed.device,
            )

        return super()._load_and_dequantize(key, hf_state_dict)

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Load HF weights, dequantizing NVFP4 bundles when present."""
        if isinstance(hf_param, str):
            return self._load_and_dequantize(hf_param, hf_state_dict)
        return {role: self._load_and_dequantize(key, hf_state_dict) for role, key in hf_param.items()}

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: Dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Re-quantize converted Kimi expert weights to NVFP4 buffer keys."""
        result = {}
        for fqn, tensor in converted_weights_dict.items():
            if self._is_quantized_expert_key(fqn):
                base = fqn[: -len(".weight")] if fqn.endswith(".weight") else fqn
                packed, scale, scale_2, shape = quantize_to_nvfp4(tensor)
                result[f"{base}_packed_fp4"] = packed
                result[f"{base}_scale_fp4"] = scale
                result[f"{base}_scale_2_fp4"] = scale_2
                result[f"{base}_shape_fp4"] = shape
            else:
                result[fqn] = tensor
        return result
