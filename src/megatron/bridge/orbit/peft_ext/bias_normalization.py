# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bias-placeholder normalization for PEFT wrapping (orbit fork).

Extracted from ``megatron.bridge.peft.utils`` and invoked by
:class:`megatron.bridge.orbit.peft_ext.peft_mixin.OrbitPEFTMixin` before the
upstream PEFT transformation walks and freezes the model.
"""

from torch import nn

from megatron.bridge.peft.utils import HAVE_TE, TECL, TERL


def _compose_module_name(name: str | None = None, prefix: str | None = None) -> str:
    if prefix and name:
        return f"{prefix}.{name}"
    if prefix:
        return prefix
    return name or ""


def module_bias_enabled(
    module: nn.Module,
    name: str | None = None,
    prefix: str | None = None,
) -> bool | None:
    """Return whether ``module`` should actively use bias.

    The explicit module flags are the primary source of truth. When a module
    has not exposed those flags yet, fall back to Megatron config-driven bias
    settings for the common transformer linear names used by adapters.
    """
    for attr_name in ("use_bias", "apply_bias"):
        if hasattr(module, attr_name):
            return bool(getattr(module, attr_name))

    config = getattr(module, "config", None)
    if config is None:
        return None

    full_name = _compose_module_name(name=name, prefix=prefix)
    leaf_name = full_name.rsplit(".", 1)[-1] if full_name else ""
    add_bias_linear = getattr(config, "add_bias_linear", None)
    add_qkv_bias = getattr(config, "add_qkv_bias", False)

    if leaf_name == "linear_qkv" and add_bias_linear is not None:
        return bool(add_bias_linear or add_qkv_bias)

    if leaf_name.startswith("linear_") and add_bias_linear is not None:
        return bool(add_bias_linear)

    # Catch-all for linear modules that don't follow the linear_* naming
    # convention. Restrict to known Linear-like types (stock + TE) so
    # this branch doesn't misclassify non-linear modules that happen to
    # carry a `bias` parameter — notably DeepSeek-V3/V4 / Mixtral
    # router gates whose bias is integral to expert selection, not a
    # placeholder governed by `config.add_bias_linear`.
    if hasattr(module, "bias") and add_bias_linear is not None:
        if isinstance(module, nn.Linear):
            return bool(add_bias_linear)
        te_linear_types = tuple(candidate for candidate in (*TECL, *TERL) if isinstance(candidate, type))
        if HAVE_TE and te_linear_types and isinstance(module, te_linear_types):
            return bool(add_bias_linear)

    return None


def normalize_disabled_bias_placeholders(
    module: nn.Module,
    name: str | None = None,
    prefix: str | None = None,
) -> nn.Module:
    """Remove placeholder bias parameters from modules that are logically no-bias.

    Under meta-init + TE, some modules can still carry ``bias`` Parameters even
    though the model configuration has bias disabled. Normalizing those modules
    before adapter wrapping keeps later PEFT transforms from inheriting the
    inconsistent placeholder state.
    """
    if module_bias_enabled(module, name=name, prefix=prefix) is not False:
        return module

    parameter_names = [
        param_name
        for param_name, param in list(getattr(module, "_parameters", {}).items())
        if param_name.startswith("bias") and param is not None
    ]
    for param_name in parameter_names:
        module.register_parameter(param_name, None)

    if hasattr(module, "bias") and "bias" not in getattr(module, "_parameters", {}):
        try:
            setattr(module, "bias", None)
        except (AttributeError, TypeError):
            pass

    return module
