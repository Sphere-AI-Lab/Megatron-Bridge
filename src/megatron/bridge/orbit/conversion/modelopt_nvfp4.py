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

"""Generic ModelOpt-NVFP4 dequant adapter for HF -> Megatron conversion.

ModelOpt NVFP4 HF exports store each quantized linear as a bundle:
``<name>.weight`` (uint8, two e2m1 values per byte), ``<name>.weight_scale``
(e4m3 per-16-element block scales) and ``<name>.weight_scale_2`` (fp32 global
scale). The packed ``.weight`` has half the logical column count, so a stock
bridge that indexes it directly produces garbage shapes.

:class:`ModelOptNVFP4DequantMixin` dequantizes such bundles to BF16 on read
through upstream's public ``maybe_modify_loaded_hf_weight`` hook and defers
every other key to the next hook in the MRO — so it composes with any
registered bridge and stacks over bridges that do their own load handling
(e.g. Kimi-K2.5's built-in INT4 path).
"""

import logging
from collections.abc import Iterable, Mapping

import torch


logger = logging.getLogger(__name__)


def is_nvfp4_bundle_key(key: str, keys: set[str] | Mapping[str, torch.Tensor]) -> bool:
    """Whether ``key`` is a ``.weight`` with NVFP4 scale siblings."""
    return key.endswith(".weight") and f"{key}_scale" in keys and f"{key}_scale_2" in keys


def hf_state_has_nvfp4_bundles(keys: Iterable[str]) -> bool:
    """Whether any checkpoint key forms a ModelOpt NVFP4 bundle."""
    key_set = set(keys)
    return any(is_nvfp4_bundle_key(k, key_set) for k in key_set)


class ModelOptNVFP4DequantMixin:
    """Architecture-independent NVFP4 dequant-on-read for model bridges.

    Compose mixin-first with any registered bridge class::

        bridge_cls = nvfp4_bridge_class_for(KimiK25VLBridge)

    Keys without an NVFP4 bundle defer to ``super()`` per key, so a base
    bridge's own load handling (or another stacked adapter) still runs.
    """

    def maybe_modify_loaded_hf_weight(
        self,
        hf_param: str | dict,
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Load HF weights, dequantizing ModelOpt NVFP4 bundles when present."""
        if isinstance(hf_param, dict):
            return {role: self._maybe_dequant_nvfp4(key, hf_state_dict) for role, key in hf_param.items()}
        return self._maybe_dequant_nvfp4(hf_param, hf_state_dict)

    def _maybe_dequant_nvfp4(
        self,
        hf_key: str,
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if is_nvfp4_bundle_key(hf_key, hf_state_dict):
            from megatron.bridge.orbit.low_precision.nvfp4 import dequantize_nvfp4

            packed = hf_state_dict[hf_key]
            shape = torch.tensor(
                [packed.shape[0], packed.shape[1] * 2],
                dtype=torch.int64,
                device=packed.device,
            )
            weight = dequantize_nvfp4(
                packed,
                hf_state_dict[f"{hf_key}_scale"],
                hf_state_dict[f"{hf_key}_scale_2"],
                shape,
                dtype=torch.bfloat16,
                device=packed.device,
            )
            logger.info("Dequantized NVFP4 -> BF16: %s shape=%s", hf_key, list(weight.shape))
            return weight
        return super().maybe_modify_loaded_hf_weight(hf_key, hf_state_dict)


def nvfp4_bridge_class_for(base_cls: type, *, extra_mixins: tuple[type, ...] = ()) -> type:
    """Compose :class:`ModelOptNVFP4DequantMixin` with a bridge class, cached."""
    from megatron.bridge.orbit.conversion.bridge_compose import quant_bridge_class_for

    return quant_bridge_class_for(ModelOptNVFP4DequantMixin, base_cls, extra_mixins=extra_mixins, name_prefix="NVFP4")


def nvfp4_bridge_for(auto_bridge):
    """Return the registered bridge for the architecture, NVFP4-composed."""
    from megatron.bridge.orbit.conversion.bridge_compose import quant_bridge_for

    return quant_bridge_for(ModelOptNVFP4DequantMixin, auto_bridge, name_prefix="NVFP4")
