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

Dequantization uses upstream's pure-torch
:func:`megatron.bridge.models.conversion.quantization_utils.dequantize_int4`,
which runs on CPU and CUDA and derives the group count from the scale shape,
so Kimi-native (group 32) and W4A16 (group 128) checkpoints both work
without configuration.
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


class CompressedTensorsINT4Mixin:
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
                from megatron.bridge.models.conversion.quantization_utils import dequantize_int4

                weight = dequantize_int4(
                    hf_state_dict[packed_key],
                    hf_state_dict[hf_key + "_scale"],
                    hf_state_dict[hf_key + "_shape"],
                    device=hf_state_dict[packed_key].device,
                )
                logger.info("Dequantized INT4 -> BF16: %s shape=%s", hf_key, list(weight.shape))
                return weight
        return hf_state_dict[hf_key]


_INT4_BRIDGE_CLASS_CACHE: dict[tuple[type, tuple[type, ...]], type] = {}


def int4_bridge_class_for(base_cls: type, *, extra_mixins: tuple[type, ...] = ()) -> type:
    """Compose :class:`CompressedTensorsINT4Mixin` with a bridge class, cached.

    Args:
        base_cls: The architecture's bridge class (usually the one registered
            with ``@MegatronModelBridge.register_bridge``).
        extra_mixins: Additional orbit mixins placed between the INT4 mixin
            and ``base_cls`` in the MRO (e.g. a provider-ext mixin).

    Returns:
        A subclass ``INT4<base_cls.__name__>`` with mixin-first MRO.
    """
    cache_key = (base_cls, extra_mixins)
    int4_cls = _INT4_BRIDGE_CLASS_CACHE.get(cache_key)
    if int4_cls is None:
        int4_cls = type(
            f"INT4{base_cls.__name__}",
            (CompressedTensorsINT4Mixin, *extra_mixins, base_cls),
            {},
        )
        _INT4_BRIDGE_CLASS_CACHE[cache_key] = int4_cls
    return int4_cls


def int4_bridge_for(auto_bridge):
    """Return the registered bridge for ``auto_bridge``'s architecture, INT4-composed.

    Mirrors ``oft_export.oft_export_bridge_for``: resolve the architecture's
    registered bridge through upstream dispatch, compose the INT4 mixin first
    in the MRO, and attach ``hf_pretrained`` / ``hf_config`` to the fresh
    instance the way upstream's ``_get_model_bridge_impl`` does.
    """
    from megatron.bridge.models.conversion import model_bridge as model_bridge_mod

    base = model_bridge_mod.get_model_bridge(auto_bridge._causal_lm_architecture)
    bridge = int4_bridge_class_for(type(base))()

    hf_pretrained = getattr(auto_bridge, "hf_pretrained", None)
    if hf_pretrained is not None:
        bridge.hf_pretrained = hf_pretrained
        bridge.hf_config = hf_pretrained.config if hasattr(hf_pretrained, "config") else hf_pretrained
    return bridge
