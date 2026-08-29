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

"""Shared composition core for orbit quantization bridge adapters.

Every quantization adapter (INT4 / FP8 / NVFP4) composes the same way: its
mixin is placed first in the MRO of a dynamically created subclass of an
architecture's registered bridge, mirroring ``oft_export.oft_export_bridge_for``.
This module holds that mechanism once so the per-format modules only supply
their mixin.
"""

_QUANT_BRIDGE_CLASS_CACHE: dict[tuple[type, type, tuple[type, ...], str], type] = {}


def quant_bridge_class_for(
    mixin_cls: type,
    base_cls: type,
    *,
    extra_mixins: tuple[type, ...] = (),
    name_prefix: str | None = None,
) -> type:
    """Compose ``mixin_cls`` (mixin-first MRO) with a bridge class, cached.

    Args:
        mixin_cls: The quantization mixin to place first in the MRO.
        base_cls: The architecture's bridge class (usually the one registered
            with ``@MegatronModelBridge.register_bridge``).
        extra_mixins: Additional orbit mixins placed between the quantization
            mixin and ``base_cls`` (e.g. a provider-ext mixin).
        name_prefix: Prefix for the generated class name; defaults to the
            mixin name without its ``Mixin`` suffix.

    Returns:
        A subclass ``<prefix><base_cls.__name__>``.
    """
    prefix = name_prefix if name_prefix is not None else mixin_cls.__name__.removesuffix("Mixin")
    cache_key = (mixin_cls, base_cls, extra_mixins, prefix)
    composed = _QUANT_BRIDGE_CLASS_CACHE.get(cache_key)
    if composed is None:
        composed = type(
            f"{prefix}{base_cls.__name__}",
            (mixin_cls, *extra_mixins, base_cls),
            {},
        )
        _QUANT_BRIDGE_CLASS_CACHE[cache_key] = composed
    return composed


def quant_bridge_for(
    mixin_cls: type,
    auto_bridge,
    *,
    extra_mixins: tuple[type, ...] = (),
    name_prefix: str | None = None,
):
    """Return the registered bridge for ``auto_bridge``'s architecture, composed.

    Mirrors ``oft_export.oft_export_bridge_for``: resolve the architecture's
    registered bridge through upstream dispatch, compose the mixin first in
    the MRO, and attach ``hf_pretrained`` / ``hf_config`` to the fresh
    instance the way upstream's ``_get_model_bridge_impl`` does.
    """
    from megatron.bridge.models.conversion import model_bridge as model_bridge_mod

    base = model_bridge_mod.get_model_bridge(auto_bridge._causal_lm_architecture)
    bridge = quant_bridge_class_for(mixin_cls, type(base), extra_mixins=extra_mixins, name_prefix=name_prefix)()

    hf_pretrained = getattr(auto_bridge, "hf_pretrained", None)
    if hf_pretrained is not None:
        bridge.hf_pretrained = hf_pretrained
        bridge.hf_config = hf_pretrained.config if hasattr(hf_pretrained, "config") else hf_pretrained
    return bridge
