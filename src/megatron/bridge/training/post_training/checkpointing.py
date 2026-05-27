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

"""Input/output checkpointing for ModelOpt."""

try:
    from modelopt.torch.opt.plugins import restore_sharded_modelopt_state
except ImportError as e:
    raise ImportError('Required `"nvidia-modelopt[torch]"` is not installed!') from e

import os

import torch
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.strategies.common import COMMON_STATE_FNAME
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import unwrap_model


def _get_modelopt_checkpoint_path(checkpoint_path: str) -> str:
    """Get the path to use for ModelOpt operations (handles iteration directories)."""
    if not checkpoint_path or not os.path.isdir(checkpoint_path):
        return checkpoint_path

    # Check for iter_* folders
    try:
        iter_folders = [
            f
            for f in os.listdir(checkpoint_path)
            if os.path.isdir(os.path.join(checkpoint_path, f)) and f.startswith("iter_")
        ]
    except (OSError, FileNotFoundError):
        # Directory doesn't exist or can't be accessed
        return checkpoint_path

    if iter_folders:
        # Find the folder with the largest iteration number from state dict
        latest_iter_num = -1
        latest_iter_folder = None

        for folder in iter_folders:
            folder_path = os.path.join(checkpoint_path, folder)
            try:
                state_dict = dist_checkpointing.load_common_state_dict(folder_path)
                if state_dict is not None:
                    iter_num = state_dict.get("iteration", 0)
                    if iter_num > latest_iter_num:
                        latest_iter_num = iter_num
                        latest_iter_folder = folder
            except Exception:
                # Skip checkpoints that fail to load
                continue

        if latest_iter_folder is not None:
            return os.path.join(checkpoint_path, latest_iter_folder)

    return checkpoint_path  # No iteration dirs, use root


def has_modelopt_state(checkpoint_path: str) -> bool:
    """Check if ModelOpt state exists inside the checkpoint path.

    Checks for modelopt_state in iteration directories (iter_*) or root directory.
    NOTE: Ignores distillation state which is deprecated and unused.

    Args:
        checkpoint_path: Path to the checkpoint directory

    Returns:
        True if modelopt_state folder exists and contains nontrivial state, else False.
    """
    modelopt_checkpoint_path = _get_modelopt_checkpoint_path(checkpoint_path)
    modelopt_state_path = os.path.join(modelopt_checkpoint_path, "modelopt_state")
    if not os.path.isdir(modelopt_state_path):
        return False

    modelopt_state = torch.load(modelopt_state_path + "/" + COMMON_STATE_FNAME, weights_only=True)
    modes = modelopt_state["modelopt_state_dict"]
    if len(modes) == 1 and modes[0][0] == "kd_loss":
        # Ignore KD state
        modes.pop()

    return len(modes) > 0


def _checkpoint_uses_packed_main_weight_layout(checkpoint_path: str) -> bool:
    """Return whether the main checkpoint stores packed low-precision weights.

    Direct NVFP4 conversion writes packed ``uint8`` tensors under regular
    ``*.weight`` keys. Those checkpoints need the restored ModelOpt module tree
    to be compressed before DCP shape validation runs; otherwise the model still
    emits dense BF16 ``*.weight`` entries.
    """
    checkpoint_path = _get_modelopt_checkpoint_path(checkpoint_path)
    try:
        tensors_metadata = dist_checkpointing.load_tensors_metadata(checkpoint_path)
    except Exception:
        return False

    for key, entry in tensors_metadata.items():
        entry_dtype = getattr(entry, "dtype", None)
        if entry_dtype is None:
            entry_dtype = getattr(getattr(entry, "properties", None), "dtype", None)
        if str(key).endswith(".weight") and entry_dtype == torch.uint8:
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
    import modelopt.torch.quantization.qtensor as _mt_qtensor_pkg
    import modelopt.torch.quantization.compress  # noqa: F401  (load submodule)

    _mt_compress = _sys.modules["modelopt.torch.quantization.compress"]

    _mt_qtensor_pkg.pack_real_quantize_weight = pack_real_quantize_weight
    _mt_compress.pack_real_quantize_weight = pack_real_quantize_weight

    # Same `module.weight` access bug exists in `update_compress_metadata`
    # (compress.py:189) — it walks `RealQuantLinear` modules and reads
    # `module.weight` without a hasattr guard, which crashes on grouped MoE
    # for the same reason. Patch with a guarded variant that mirrors
    # ModelOpt's logic but skips modules without a `.weight`.
    from modelopt.torch.quantization.qtensor.base_qtensor import QTensorWrapper
    from modelopt.torch.quantization.nn.modules.quant_linear import RealQuantLinear

    def update_compress_metadata(model, config, metadata):
        # save scales state in weight quantizer
        real_quantizer_state = {}
        for name, m in model.named_modules():
            if isinstance(m, RealQuantLinear):
                real_quantizer_state[name] = m.weight_quantizer.get_modelopt_state()

        # real quant tensor states — guard `m.weight` access for grouped MoE
        q_tensor_state = {}
        for name, m in model.named_modules():
            if (
                isinstance(m, RealQuantLinear)
                and hasattr(m, "weight")
                and isinstance(m.weight, QTensorWrapper)
            ):
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
        if isinstance(self, RealQuantLinear) and hasattr(self, "weight") and isinstance(
            self.weight, QTensorWrapper
        ):
            extra_state["modelopt_real_quantizer_state"] = (
                self.weight_quantizer.get_modelopt_state()
            )
            extra_state["modelopt_q_tensor_state"] = self.weight.get_state()
        elif isinstance(self, RealQuantLinear) and hasattr(self, "weight"):
            extra_state["modelopt_real_quantizer_state"] = (
                self.weight_quantizer.get_modelopt_state()
            )
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

    print(
        "Detected packed low-precision main checkpoint layout; "
        "compressing restored ModelOpt modules before distributed weight load."
    )
    mtq.compress(model[0])


def load_modelopt_state(model: list[MegatronModule], checkpoint_path: str) -> None:
    """Load modelopt_state from a checkpoint.
    Args:
        model: The model to load the modelopt_state into
        checkpoint_path: Path to the checkpoint directory
    """
    modelopt_checkpoint_path = _get_modelopt_checkpoint_path(checkpoint_path)
    unwrapped_model = unwrap_model(model)
    restore_sharded_modelopt_state(unwrapped_model, modelopt_checkpoint_path)
    _maybe_compress_restored_modelopt_model(unwrapped_model, modelopt_checkpoint_path)
