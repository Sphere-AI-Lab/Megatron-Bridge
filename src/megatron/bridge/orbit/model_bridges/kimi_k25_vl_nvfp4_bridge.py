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

"""NVFP4-preserving bridge for Kimi-K2.5 VL checkpoints.

The load side is the architecture-independent
:class:`~megatron.bridge.orbit.conversion.modelopt_nvfp4.ModelOptNVFP4DequantMixin`;
keys without an NVFP4 bundle fall through to the Kimi base bridge, which
keeps its built-in INT4 triplet handling. What stays here is the emit side:
re-quantizing converted expert weights into the ``*_packed_fp4`` /
``*_scale_fp4`` / ``*_scale_2_fp4`` / ``*_shape_fp4`` buffer keys that the
direct NVFP4 save path consumes.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
from megatron.bridge.models.kimi_vl.kimi_k25_vl_bridge import KimiK25VLBridge
from megatron.bridge.orbit.conversion.modelopt_nvfp4 import ModelOptNVFP4DequantMixin
from megatron.bridge.orbit.low_precision.nvfp4 import quantize_to_nvfp4


class KimiK25VLNVFP4Bridge(ModelOptNVFP4DequantMixin, KimiK25VLBridge):
    """Kimi-K2.5 bridge variant that consumes and emits NVFP4 expert bundles."""

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
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
