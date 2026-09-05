# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Packed low-precision ModelOpt checkpoint restore (orbit fork).

Extracted from ``megatron.bridge.training.post_training.checkpointing``; the
shared module keeps a marked hook in ``load_modelopt_state``. See
``docs/orbit/UPSTREAM-SEAMS.md``.
"""

import logging
import re

import torch
from megatron.core import dist_checkpointing
from megatron.core.transformer.module import MegatronModule

from megatron.bridge.training.post_training.checkpointing import _get_modelopt_checkpoint_path


logger = logging.getLogger(__name__)

_FP8_SPLIT_WEIGHT_RE = re.compile(r"^(?P<base>.+\.(?:weight|weight\d+))_w$")
_GROUPED_FP8_WEIGHT_RE = re.compile(r"^(?P<base>.+\.experts\.linear_fc[12]\.weight\d+)$")


def _checkpoint_uses_packed_main_weight_layout(checkpoint_path: str) -> bool:
    """Return whether the main checkpoint stores packed low-precision weights.

    Direct NVFP4 conversion writes packed ``uint8`` tensors under regular
    ``*.weight`` keys. Direct FP8 conversion writes ``float8_e4m3fn`` tensors
    under ModelOpt's ``*.weight_w`` key, with a sibling
    ``*.weight_scale_inv`` entry (and ``*.weight_v`` for split SwiGLU).
    Grouped-expert linears use indexed ``*.weightN`` keys, with SwiGLU FC1
    represented as ``*.weightN_w`` and ``*.weightN_v``. Those checkpoints need
    the restored ModelOpt module tree to be compressed before DCP shape
    validation runs; otherwise the model still emits dense BF16 entries.
    """
    checkpoint_path = _get_modelopt_checkpoint_path(checkpoint_path)
    try:
        tensors_metadata = dist_checkpointing.load_tensors_metadata(checkpoint_path)
    except Exception:
        return False

    tensor_dtypes = {}
    for key, entry in tensors_metadata.items():
        entry_dtype = getattr(entry, "dtype", None)
        if entry_dtype is None:
            entry_dtype = getattr(getattr(entry, "properties", None), "dtype", None)
        tensor_dtypes[str(key)] = entry_dtype

    for key, entry_dtype in tensor_dtypes.items():
        if key.endswith(".weight") and entry_dtype == torch.uint8:
            return True

        if entry_dtype != torch.float8_e4m3fn:
            continue

        split_match = _FP8_SPLIT_WEIGHT_RE.fullmatch(key)
        grouped_match = _GROUPED_FP8_WEIGHT_RE.fullmatch(key)
        if split_match is not None:
            weight_key = split_match.group("base")
        elif grouped_match is not None:
            weight_key = grouped_match.group("base")
        else:
            continue
        if f"{weight_key}_scale_inv" not in tensor_dtypes:
            continue

        weight_v_key = f"{weight_key}_v"
        if weight_v_key in tensor_dtypes and tensor_dtypes[weight_v_key] != torch.float8_e4m3fn:
            continue
        return True

    return False


def _patch_modelopt_pack_for_grouped_moe() -> None:
    """Add a ``hasattr(m, "weight")`` guard to ModelOpt's compress walk.

    Megatron's MoE expert layers (``TEColumnParallelGroupedLinear`` /
    ``TERowParallelGroupedLinear``) hold per-expert weights as ``weight0`` ...
    ``weightN-1`` rather than a single ``.weight`` attribute. ModelOpt's
    ``pack_real_quantize_weight`` (``modelopt.torch.quantization.qtensor.
    base_qtensor``) walks every module after restore and inside its
    ``weight_quantizer`` guard accesses ``module.weight.element_size()``
    *without* checking ``hasattr(module, "weight")`` first, which raises
    ``AttributeError: 'RealQuantQuantTEColumnParallelGroupedLinear' object has
    no attribute 'weight'`` on the grouped expert linears.

    Those grouped MoE modules already carry per-expert NVFP4 buffers restored
    by ``restore_sharded_modelopt_state`` and don't need ModelOpt's BF16->packed
    compress step — only the dense attention/MLP linears do. Skipping modules
    without ``.weight`` keeps the dense compress path intact and lets the
    standard sharded-state-dict loader fill in the per-expert buffers.

    Idempotent — called from ``_maybe_compress_restored_modelopt_model`` before
    ``mtq.compress``. Both the QOFT finetune path and the orbit RL load path go
    through the same pre-load pipeline, so this fix benefits both.
    """
    import modelopt.torch.quantization.qtensor.base_qtensor as _mt_qbase

    if getattr(_mt_qbase, "_megatron_bridge_grouped_moe_patch_applied", False):
        return

    from modelopt.torch.quantization.nn import SequentialQuantizer
    from modelopt.torch.quantization.qtensor.base_qtensor import (
        QTensorWrapper,
        fsdp2_aware_weight_update,
        patch_fsdp_mp_dtypes,
    )

    def pack_real_quantize_weight(module, force_quantize: bool = False):
        def _compress_and_update_module_weight(m):
            # NEW guard: skip modules without a `.weight` attribute (grouped MoE).
            if not hasattr(m, "weight"):
                return False
            if m.weight is None:
                return False
            # NOTE: original ModelOpt code skipped meta-device weights, expecting
            # `set_extra_state` during load_state_dict to install the
            # QTensorWrapper. That breaks shape validation: sharded_state_dict()
            # runs *before* set_extra_state, so the SwiGLU factory chunks the
            # unpacked meta weight `[out, in]` into halves `[out/2, in]` and
            # mismatches the checkpoint's packed `[out/2, in/2]` halves. We
            # process meta weights too — the quantizer call propagates the
            # meta-ness, producing a meta QTensorWrapper with the correct
            # packed shape, which sharded_state_dict() then reports correctly.
            if (
                hasattr(m, "weight_quantizer")
                and m.weight_quantizer.is_enabled
                and not m.weight_quantizer._fake_quant
                and m.weight.element_size() > 1
            ):
                if force_quantize:
                    m.weight_quantizer._dequantize = False
                real_quant_tensor = m.weight_quantizer(m.weight)
                m.weight = QTensorWrapper(real_quant_tensor)
                return True
            return False

        with (
            SequentialQuantizer.convert_to_single_quantizer(module),
            torch.no_grad(),
            patch_fsdp_mp_dtypes(),
        ):
            for name, m in module.named_modules():
                if name != "":
                    with fsdp2_aware_weight_update(module, m):
                        _compress_and_update_module_weight(m)

    _mt_qbase.pack_real_quantize_weight = pack_real_quantize_weight

    # Also re-bind the function reference at every import site. ModelOpt's
    # `qtensor/__init__.py` does `from .base_qtensor import *` and
    # `compress.py` does `from .qtensor import ... pack_real_quantize_weight`,
    # so each of those modules holds its own copy of the original at
    # import-time. Without re-binding, `mtq.compress` keeps calling the
    # un-patched version even though `base_qtensor` now has the new one.
    #
    # Note: `modelopt.torch.quantization.compress` is both a function (the
    # public `mtq.compress`) and a submodule. `import ... as` resolves to the
    # function because `quantization/__init__.py` re-exports it, so we have to
    # grab the submodule via `sys.modules`.
    import sys as _sys

    import modelopt.torch.quantization.compress  # noqa: F401  (load submodule)
    import modelopt.torch.quantization.qtensor as _mt_qtensor_pkg

    _mt_compress = _sys.modules["modelopt.torch.quantization.compress"]

    _mt_qtensor_pkg.pack_real_quantize_weight = pack_real_quantize_weight
    _mt_compress.pack_real_quantize_weight = pack_real_quantize_weight

    # Same `module.weight` access bug exists in `update_compress_metadata`
    # (compress.py:189) — it walks `RealQuantLinear` modules and reads
    # `module.weight` without a hasattr guard, which crashes on grouped MoE
    # for the same reason. Patch with a guarded variant that mirrors
    # ModelOpt's logic but skips modules without a `.weight`.
    from modelopt.torch.quantization.nn.modules.quant_linear import RealQuantLinear
    from modelopt.torch.quantization.qtensor.base_qtensor import QTensorWrapper

    def update_compress_metadata(model, config, metadata):
        # save scales state in weight quantizer
        real_quantizer_state = {}
        for name, m in model.named_modules():
            if isinstance(m, RealQuantLinear):
                real_quantizer_state[name] = m.weight_quantizer.get_modelopt_state()

        # real quant tensor states — guard `m.weight` access for grouped MoE
        q_tensor_state = {}
        for name, m in model.named_modules():
            if isinstance(m, RealQuantLinear) and hasattr(m, "weight") and isinstance(m.weight, QTensorWrapper):
                q_tensor_state[name] = m.weight.get_state()

        metadata["real_quantizer_state"] = real_quantizer_state
        metadata["q_tensor_state"] = q_tensor_state

    _mt_compress.update_compress_metadata = update_compress_metadata

    # `mode.py` does `from .compress import update_compress_metadata` at import
    # time, capturing the unpatched function reference. ``mtq.compress`` reaches
    # the metadata path via ``mode.update_state_before_new_mode_func``, which
    # returns that captured reference — so rebinding only `_mt_compress` is not
    # enough. Rebind the name on `mode` too.
    import modelopt.torch.quantization.mode as _mt_mode

    _mt_mode.update_compress_metadata = update_compress_metadata

    # ModelOpt's megatron plugin has the same bug in two more sites that the
    # standard checkpoint save/load path hits on every grouped MoE module:
    #   plugins/megatron.py:75   `real_quant_module_get_extra_state` (get path)
    #   plugins/megatron.py:142  `real_quant_module_set_extra_state` (set path)
    # Both access `self.weight` unconditionally. Replace them with no-op-on-grouped
    # variants. For grouped MoE, the per-expert NVFP4 buffers (`weightN`, scaleN,
    # amaxN) are loaded via the standard sharded-state-dict path, so an empty
    # extra_state for these modules is correct.
    import modelopt.torch.quantization.plugins.megatron as _mt_megatron_plugin

    def real_quant_module_get_extra_state(self) -> dict:
        extra_state: dict = {}
        if isinstance(self, RealQuantLinear) and hasattr(self, "weight") and isinstance(self.weight, QTensorWrapper):
            extra_state["modelopt_real_quantizer_state"] = self.weight_quantizer.get_modelopt_state()
            extra_state["modelopt_q_tensor_state"] = self.weight.get_state()
        elif isinstance(self, RealQuantLinear) and hasattr(self, "weight"):
            extra_state["modelopt_real_quantizer_state"] = self.weight_quantizer.get_modelopt_state()
            extra_state["modelopt_q_tensor_state"] = {}
        else:
            # Grouped MoE (no `.weight`) and non-RealQuantLinear modules: no real
            # quantizer state to save here.
            extra_state["modelopt_real_quantizer_state"] = None
            extra_state["modelopt_q_tensor_state"] = None
        return extra_state

    def real_quant_module_set_extra_state(self, state):
        # If the module doesn't expose a `.weight` (grouped MoE) there's nothing
        # to restore through this path — the per-expert weights live in
        # `weightN` and are restored via sharded_state_dict.
        if not hasattr(self, "weight"):
            return
        q_tensor_state = state.get("modelopt_q_tensor_state", None)
        if not q_tensor_state:
            return
        q_tensor_metadata = q_tensor_state["metadata"]
        q_tensor_metadata["shape"] = self.weight.shape
        q_tensor_data_dtype = q_tensor_state["quantized_data.dtype"]
        q_tensor_shape = self.weight.shape
        if q_tensor_data_dtype == torch.uint8:
            q_tensor_shape = list(q_tensor_shape)
            q_tensor_shape[-1] = q_tensor_shape[-1] // 2
            q_tensor_shape = torch.Size(q_tensor_shape)
        self._parameters["weight"] = QTensorWrapper(
            qtensor=torch.empty(
                q_tensor_shape,
                dtype=q_tensor_data_dtype,
                device=self.weight.device,
            ),
            metadata=q_tensor_metadata,
        )

    _mt_megatron_plugin.real_quant_module_get_extra_state = real_quant_module_get_extra_state
    _mt_megatron_plugin.real_quant_module_set_extra_state = real_quant_module_set_extra_state

    _mt_qbase._megatron_bridge_grouped_moe_patch_applied = True


def _maybe_compress_restored_modelopt_model(
    model: list[MegatronModule],
    checkpoint_path: str,
) -> None:
    """Compress a restored ModelOpt model when the checkpoint stores packed weights."""
    if not _checkpoint_uses_packed_main_weight_layout(checkpoint_path):
        return

    # Make ModelOpt's compress step grouped-MoE-safe before invoking it. See
    # _patch_modelopt_pack_for_grouped_moe() docstring for details.
    _patch_modelopt_pack_for_grouped_moe()

    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.compress import is_real_quantized

    if len(model) != 1:
        raise ValueError("Packed ModelOpt checkpoint restore only supports a single model chunk.")

    # Idempotent: callers in the verl/Megatron-Bridge restore pipeline may
    # invoke this twice (once during sharded modelopt-state restore, once via
    # the explicit load_modelopt_state hook). The second `mtq.compress` would
    # otherwise raise `AssertionError: Cannot add real_quantize after
    # real_quantize` because ModelOpt's mode manager refuses to re-apply a
    # mode the model is already in.
    if is_real_quantized(model[0]):
        return

    logger.info(
        "Detected packed low-precision main checkpoint layout; "
        "compressing restored ModelOpt modules before distributed weight load."
    )
    mtq.compress(model[0])
