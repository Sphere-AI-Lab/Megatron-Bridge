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

"""Generic compressed-tensors INT4 adapter for HF -> Megatron conversion.

HF checkpoints in compressed-tensors ``pack-quantized`` form (and Kimi's
native INT4 layout, which shares it) store each quantized linear as a triplet
``<name>.weight_packed`` / ``<name>.weight_scale`` / ``<name>.weight_shape``
with no plain ``<name>.weight`` key. Upstream mapping registries look up
``*.weight``, so a stock bridge cannot read such checkpoints.

This module is the architecture-independent adapter. It contains no
model-specific code: the two behaviors it adds — synthesizing virtual
``.weight`` keys during task construction and dequantizing triplets on
read — are pure key-string and tensor operations that compose with any
registered bridge through upstream's public hooks (``build_conversion_tasks``
and ``maybe_modify_loaded_hf_weight``).

Composition mirrors ``oft_export.oft_export_bridge_for``: the mixin is placed
first in the MRO of a dynamically created subclass of the architecture's
registered bridge, so per-model INT4 bridge classes reduce to one-line
compositions and unlisted architectures get INT4 support for free.

Dequantization uses the pure-torch
:func:`megatron.bridge.models.kimi_vl.utils.dequantize_int4`,
which runs on CPU and CUDA and derives the group count from the scale shape,
so Kimi-native (group 32) and W4A16 (group 128) checkpoints both work
without configuration.

Why dequantize (format rationale):
    Packed formats are not element-addressable: every int32 stores eight
    4-bit weights along the input dimension, and Q/K/V each carry their own
    per-group scales. Conversion must run tensor surgery — head-interleaved
    QKV merge, gate/up concat, TP chunking — and none of those operations
    are defined on packed storage without re-implementing them in "packed
    space" while keeping the 8-per-int32 packing and the scale-group
    boundaries aligned through every cut. BF16 is therefore used as a
    *transit* representation: dequantize one weight at a time, run the
    standard dtype-agnostic mapping machinery unchanged, and re-quantize
    afterwards wherever INT4 is to be kept. The BF16 transit itself is
    exact, but re-quantization is only bitwise when the SOURCE scales are
    reused: real checkpoints use the full signed range including -8, which
    an amax/7 scale recomputation can never reproduce (it re-grids those
    groups). The direct-save builder therefore preserves triplets outright,
    and DeepSeekV3INT4Bridge's expert requantize reuses the source scales
    (``requantize_int4_with_scales``). Contrast with block-FP8 (one byte
    per element), where the
    surgery runs natively on the stored bytes and orbit preserves instead —
    see ``fp8_preserve``.
"""

import logging
from collections.abc import Iterable, Mapping

import torch


logger = logging.getLogger(__name__)

_PACKED_SUFFIX = "_packed"


def synthesize_virtual_weight_keys(keys: list[str]) -> list[str]:
    """Return ``keys`` plus a virtual ``.weight`` key for every INT4 triplet.

    A triplet is ``<base>_packed`` accompanied by ``<base>_scale`` and
    ``<base>_shape``; the virtual key is ``<base>`` (i.e. ``...weight``).
    Keys that already exist are never duplicated.

    Args:
        keys: All keys present in the HF checkpoint source.

    Returns:
        The original keys followed by the synthesized virtual keys.
    """
    all_keys = set(keys)
    virtual_keys = []
    for key in keys:
        if key.endswith(_PACKED_SUFFIX):
            base = key[: -len(_PACKED_SUFFIX)]  # "...weight_packed" -> "...weight"
            if f"{base}_scale" in all_keys and f"{base}_shape" in all_keys and base not in all_keys:
                virtual_keys.append(base)
    return list(keys) + virtual_keys


def hf_state_has_int4_triplets(keys: Iterable[str]) -> bool:
    """Whether any checkpoint key forms a compressed-tensors INT4 triplet."""
    keys = list(keys)
    return len(synthesize_virtual_weight_keys(keys)) > len(keys)


class CompressedTensorsINT4DequantMixin:
    """Architecture-independent INT4 triplet support for model bridges.

    Compose mixin-first with any registered bridge class::

        bridge_cls = int4_bridge_class_for(Qwen3Bridge)

    The mixin overrides two upstream hooks:

    - :meth:`build_conversion_tasks` temporarily widens the source's key
      listing with virtual ``.weight`` keys so the parent bridge's mapping
      registry (QKVMapping, GatedMLPMapping, AutoMapping, ...) finds every
      quantized linear.
    - :meth:`maybe_modify_loaded_hf_weight` dequantizes a triplet to BF16 on
      read, so QKV merge, gate/up concat, and TP split run unchanged.
    """

    def build_conversion_tasks(self, hf_pretrained, megatron_model) -> list:
        """Build conversion tasks with virtual ``.weight`` keys visible."""
        source = hf_pretrained.state.source
        original_get_all_keys = source.get_all_keys

        def _get_all_keys_with_virtual():
            return synthesize_virtual_weight_keys(original_get_all_keys())

        source.get_all_keys = _get_all_keys_with_virtual
        try:
            return super().build_conversion_tasks(hf_pretrained, megatron_model)
        finally:
            source.get_all_keys = original_get_all_keys

    def maybe_modify_loaded_hf_weight(
        self,
        hf_param: str | dict,
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Load HF weights, dequantizing INT4 triplets when present."""
        if isinstance(hf_param, dict):
            return {role: self._maybe_dequant_int4(key, hf_state_dict) for role, key in hf_param.items()}
        return self._maybe_dequant_int4(hf_param, hf_state_dict)

    def _maybe_dequant_int4(
        self,
        hf_key: str,
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if hf_key.endswith(".weight"):
            packed_key = hf_key + _PACKED_SUFFIX
            if packed_key in hf_state_dict:
                from megatron.bridge.models.kimi_vl.utils import dequantize_int4

                weight = dequantize_int4(
                    hf_state_dict[packed_key],
                    hf_state_dict[hf_key + "_scale"],
                    hf_state_dict[hf_key + "_shape"],
                    device=hf_state_dict[packed_key].device,
                )
                logger.info("Dequantized INT4 -> BF16: %s shape=%s", hf_key, list(weight.shape))
                return weight
        # Not an INT4 triplet: defer to the next maybe_modify_loaded_hf_weight
        # in the MRO (upstream default indexes the state dict; a base bridge or
        # another stacked adapter may transform the tensor instead).
        return super().maybe_modify_loaded_hf_weight(hf_key, hf_state_dict)


def int4_bridge_class_for(base_cls: type, *, extra_mixins: tuple[type, ...] = ()) -> type:
    """Compose :class:`CompressedTensorsINT4DequantMixin` with a bridge class, cached.

    Args:
        base_cls: The architecture's bridge class (usually the one registered
            with ``@MegatronModelBridge.register_bridge``).
        extra_mixins: Additional orbit mixins placed between the INT4 mixin
            and ``base_cls`` in the MRO (e.g. a provider-ext mixin).

    Returns:
        A subclass ``INT4<base_cls.__name__>`` with mixin-first MRO.
    """
    from megatron.bridge.orbit.conversion.bridge_compose import quant_bridge_class_for

    return quant_bridge_class_for(
        CompressedTensorsINT4DequantMixin, base_cls, extra_mixins=extra_mixins, name_prefix="INT4"
    )


def int4_bridge_for(auto_bridge):
    """Return the registered bridge for ``auto_bridge``'s architecture, INT4-composed."""
    from megatron.bridge.orbit.conversion.bridge_compose import quant_bridge_for

    return quant_bridge_for(CompressedTensorsINT4DequantMixin, auto_bridge, name_prefix="INT4")
