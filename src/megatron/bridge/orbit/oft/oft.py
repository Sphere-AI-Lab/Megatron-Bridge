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

"""OFT (Orthogonal Fine-Tuning) PEFT method for megatron-bridge.

OFT fine-tunes models by learning orthogonal rotations applied to the input
of linear layers. Unlike LoRA which adds a low-rank residual (W' = W + BA),
OFT applies a multiplicative orthogonal transform (y = W @ R @ x).

Reference: https://arxiv.org/abs/2306.07280
"""

import logging
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.utils import unwrap_model

from megatron.bridge.orbit.oft.oft_layers import (
    OFTLinear,
    OFTRotationModule,
    OFTTopKRouter,
    TEOFTLayerNormLinear,
    _clear_disabled_bias_parameters,
    _fp8_activation_qdq_per_token_group_ste,
    _get_active_bias_tensor,
    _get_oft_fp8_activation_quant_mode,
    _is_direct_fp8_runtime_weight,
    _module_bias_enabled,
    _oft_fp8_debug_log,
    _prepare_raw_column_parallel_input,
    _validate_oft_hyperparameters,
)
from megatron.bridge.orbit.peft_ext.adapter_attrs import get_oft_adapter_attributes_from_linear
from megatron.bridge.orbit.peft_ext.peft_mixin import OrbitPEFTMixin
from megatron.bridge.orbit.quant.qwen3_fp8_gemm import (
    maybe_qwen3_native_block_fp8_linear,
    should_attempt_qwen3_native_fp8_gemm,
)
from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.module_matcher import ModuleMatcher
from megatron.bridge.peft.utils import is_expert_linear
from megatron.bridge.utils.import_utils import safe_import_from


logger = logging.getLogger(__name__)

TELayerNormColumnParallelLinear, HAVE_TE_LN_COL_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine",
    "TELayerNormColumnParallelLinear",
)


def _localize_oft_row_parallel_geometry(full_in_features: int, r: int, tp_size: int) -> tuple[int, int]:
    """Return TP-local input width and block count without splitting OFT blocks."""
    if tp_size <= 0:
        raise ValueError(f"tensor-parallel size must be positive, got {tp_size}")
    if full_in_features % tp_size != 0:
        raise ValueError(f"in_features ({full_in_features}) must be divisible by tensor-parallel size ({tp_size})")
    if r > 0 and r % tp_size != 0:
        raise ValueError(f"r ({r}) must be divisible by tensor-parallel size ({tp_size})")
    local_r = r // tp_size if r > 0 else r
    return full_in_features // tp_size, local_r


class _SplitLNOFTLinear(nn.Module):
    """Splits a fused TELayerNormColumnParallelLinear into LN + OFT + Linear.

    Running OFT inside TE's custom autograd.Function severs gradient flow to
    OFT parameters. This module splits the fused layer into three separate
    operations:
      1. LayerNorm (reusing fused module's LN weights, via torch.nn.functional)
      2. OFT rotation (normal nn.Module, autograd tracks params)
      3. Linear projection (TE path for dense/FP8 weights, de-fused path for INT4/NVFP4)

    No redundant computation — each op runs exactly once.
    """

    def __init__(self, orig_module: nn.Module, adapter: nn.Module) -> None:
        super().__init__()
        self._orig_module = orig_module
        _clear_disabled_bias_parameters(self._orig_module)
        self.adapter = adapter
        self._adapter_enabled = True

        # Build standalone norm using fused module's LN weights (shared, no copy)
        normalization = getattr(orig_module, "normalization", "LayerNorm")
        eps = getattr(orig_module, "eps", 1e-5)
        hidden_size = orig_module.layer_norm_weight.shape[0]
        self._normalization = normalization
        self._eps = eps
        self._hidden_size = hidden_size
        # LN weights are registered on orig_module; we reference them directly
        # so checkpoint save/load still finds them at the original FQN.

        # Build standalone TE Linear reusing the fused module's weight/bias.
        # We construct a TEColumnParallelLinear that shares the same parameters.
        from megatron.core.extensions.transformer_engine import TEColumnParallelLinear

        config = orig_module.config
        out_features, in_features = (
            orig_module.weight.shape
            if orig_module.weight.numel() > 0
            else (
                getattr(orig_module, "out_features", 0),
                getattr(orig_module, "in_features", 0),
            )
        )
        # Get the actual sizes from config
        tp_size = getattr(orig_module, "tp_size", 1)
        has_bias = _module_bias_enabled(orig_module)

        self.linear = TEColumnParallelLinear(
            input_size=in_features * tp_size,  # full size before TP split
            output_size=out_features * tp_size,
            config=config,
            init_method=lambda w: None,  # no init, we'll share weights
            gather_output=False,
            bias=has_bias,
            skip_bias_add=getattr(orig_module, "te_return_bias", False),
            is_expert=False,
            tp_comm_buffer_name=getattr(orig_module, "ub_name", None),
        )
        # Share weight and bias parameters (no extra memory)
        self.linear.weight = orig_module.weight
        if has_bias:
            self.linear.bias = orig_module.bias
        _clear_disabled_bias_parameters(self.linear)

        self._base_fp8 = _is_direct_fp8_runtime_weight(
            self.linear.weight,
            getattr(self._orig_module, "weight_scale_inv", None),
        )
        self._base_int4 = hasattr(orig_module, "weight_packed")
        self._base_nvfp4 = self._is_base_nvfp4()

    def _is_base_fp8(self) -> bool:
        """Return whether the split base linear currently carries direct FP8 weights."""
        weight = getattr(self.linear, "weight", None)
        orig_module = self._get_orig_module()
        scale_inv = getattr(orig_module, "weight_scale_inv", None) if orig_module is not None else None
        return _is_direct_fp8_runtime_weight(weight, scale_inv)

    def _debug_fp8_detection_miss(self, x: torch.Tensor) -> None:
        if self._base_fp8 or self._is_base_fp8():
            return
        weight = getattr(self.linear, "weight", None)
        orig_module = self._get_orig_module()
        scale_inv = getattr(orig_module, "weight_scale_inv", None) if orig_module is not None else None
        _oft_fp8_debug_log(
            "split_ln_detection_miss",
            wrapper=type(self).__name__,
            orig_module=type(orig_module).__name__ if orig_module is not None else None,
            cached_base_fp8=self._base_fp8,
            weight=weight,
            scale_inv=scale_inv,
            has_scale_inv=scale_inv is not None,
            input=x,
        )

    def _is_base_nvfp4(self) -> bool:
        """Return whether the wrapped module currently carries ModelOpt NVFP4 weight state."""
        orig_module = self._get_orig_module()
        if orig_module is None:
            return False
        weight = getattr(orig_module, "weight", None)
        quantizer = getattr(orig_module, "weight_quantizer", None)
        return (
            getattr(weight, "dtype", None) == torch.uint8
            and quantizer is not None
            and hasattr(quantizer, "_scale")
            and hasattr(quantizer, "_double_scale")
        )

    def _apply_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LayerNorm or RMSNorm using the fused module's weights."""
        ln_weight = self._orig_module.layer_norm_weight
        ln_bias = self._orig_module.layer_norm_bias
        if getattr(self._orig_module, "zero_centered_gamma", False):
            ln_weight = ln_weight + 1.0
        if self._normalization == "RMSNorm":
            return torch.nn.functional.rms_norm(x, (self._hidden_size,), ln_weight, self._eps)
        else:
            return torch.nn.functional.layer_norm(x, (self._hidden_size,), ln_weight, ln_bias, self._eps)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        # 1. LayerNorm
        ln_out = self._apply_norm(x)
        # 2. OFT rotation (normal nn.Module — autograd tracks OFT params)
        rotated = self.adapter(ln_out.contiguous()) if self.__dict__.get("_adapter_enabled", True) else ln_out
        # 3. Linear projection — dense/FP8 use TE; packed bases use split linear semantics.
        if self._base_int4:
            return self._linear_int4(rotated)
        if getattr(self, "_base_nvfp4", False) or self._is_base_nvfp4():
            return self._linear_nvfp4(rotated)
        if self._base_fp8 or self._is_base_fp8():
            return self._linear_fp8(rotated)
        self._debug_fp8_detection_miss(rotated)
        out = self.linear(rotated)
        return out

    def enable_adapter_layers(self) -> None:
        self._adapter_enabled = True

    def disable_adapter_layers(self) -> None:
        self._adapter_enabled = False

    def _nvfp4_weight_shape(self, packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        weight_shape = getattr(self._orig_module, "weight_shape", None)
        if weight_shape is not None:
            return weight_shape

        device = getattr(packed, "device", None) or getattr(scale, "device", torch.device("cpu"))
        if getattr(scale, "ndim", 0) >= 2:
            from megatron.bridge.orbit.low_precision.nvfp4 import NVFP4_GROUP_SIZE

            return torch.tensor(
                [int(scale.shape[-2]), int(scale.shape[-1]) * NVFP4_GROUP_SIZE],
                dtype=torch.int64,
                device=device,
            )

        return torch.tensor(
            [int(packed.shape[0]), int(packed.shape[1]) * 2],
            dtype=torch.int64,
            device=device,
        )

    def _linear_nvfp4(self, x: torch.Tensor):
        """NVFP4 forward using a temporary dequantized compute weight.

        ModelOpt restores direct NVFP4 bridge weights as packed ``uint8`` plus
        quantizer scale buffers. TE Linear cannot consume that packed parameter
        in the de-fused OFT path, so mirror the INT4 branch and run the GEMM
        against a bf16/fp16 dequantized view while keeping autograd saves packed.
        """
        from megatron.bridge.orbit.low_precision.nvfp4 import dequantize_nvfp4

        packed = self._orig_module.weight
        quantizer = self._orig_module.weight_quantizer
        scale = quantizer._scale
        scale_2 = quantizer._double_scale
        shape = self._nvfp4_weight_shape(packed, scale)

        w_compute = dequantize_nvfp4(packed, scale, scale_2, shape, dtype=x.dtype, device=x.device)
        w_compute_ptr = w_compute.data_ptr()
        bias = _get_active_bias_tensor(self._orig_module)
        has_bias = bias is not None

        x = _prepare_raw_column_parallel_input(self._orig_module, x)
        x = x.to(w_compute.dtype)
        te_return_bias = getattr(self._orig_module, "te_return_bias", False)

        def pack(tensor):
            if tensor.data_ptr() == w_compute_ptr:
                return (
                    packed,
                    scale,
                    scale_2,
                    shape,
                    tensor.dtype,
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    tensor.storage_offset(),
                )
            return tensor

        def unpack(packed_tuple):
            if isinstance(packed_tuple, tuple) and len(packed_tuple) == 8:
                p, s, s2, sh, dtype, saved_shape, saved_stride, saved_storage_offset = packed_tuple
                base = dequantize_nvfp4(p, s, s2, sh, dtype=dtype, device=p.device)
                return base.as_strided(saved_shape, saved_stride, saved_storage_offset)
            return packed_tuple

        try:
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                if te_return_bias and has_bias:
                    out = F.linear(x, w_compute, None)
                    return out, bias
                out = F.linear(x, w_compute, bias if has_bias else None)
                return out, None
        finally:
            del w_compute

    def _linear_int4(self, x: torch.Tensor):
        """INT4 forward using the split module's linear semantics.

        We keep the split LN -> OFT -> linear structure, but TE weights cannot
        be rebound to a temporary dense bf16 tensor safely here because TE owns
        a custom parameter/tensor type. Instead, we mirror the existing
        de-fused linear behavior and source the GEMM weight from the INT4
        triplet on each forward.
        """
        from megatron.bridge.orbit.low_precision.int4 import dequantize_int4

        packed = self._orig_module.weight_packed
        scale = self._orig_module.weight_scale
        shape = self._orig_module.weight_shape

        w_compute = dequantize_int4(packed, scale, shape, device=x.device).to(x.dtype)
        w_compute_ptr = w_compute.data_ptr()
        bias = _get_active_bias_tensor(self._orig_module)
        has_bias = bias is not None

        x = _prepare_raw_column_parallel_input(self._orig_module, x)
        x = x.to(w_compute.dtype)
        te_return_bias = getattr(self._orig_module, "te_return_bias", False)

        def pack(tensor):
            if tensor.data_ptr() == w_compute_ptr:
                return (
                    packed,
                    scale,
                    shape,
                    tensor.dtype,
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    tensor.storage_offset(),
                )
            return tensor

        def unpack(packed_tuple):
            if isinstance(packed_tuple, tuple) and len(packed_tuple) == 7:
                p, s, sh, dtype, saved_shape, saved_stride, saved_storage_offset = packed_tuple
                base = dequantize_int4(p, s, sh, device=p.device).to(dtype)
                return base.as_strided(saved_shape, saved_stride, saved_storage_offset)
            if isinstance(packed_tuple, tuple) and len(packed_tuple) == 4:
                p, s, sh, dtype = packed_tuple
                return dequantize_int4(p, s, sh, device=p.device).to(dtype)
            return packed_tuple

        try:
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                if te_return_bias and has_bias:
                    out = F.linear(x, w_compute, None)
                    return out, bias
                out = F.linear(x, w_compute, bias if has_bias else None)
                return out, None
        finally:
            del w_compute

    def _linear_fp8(self, x: torch.Tensor):
        """FP8 forward using the split module's linear semantics.

        TE Linear cannot safely consume a temporary dense BF16 weight through
        ``weight.data`` rebinding here. Mirror the INT4/NVFP4 split branches:
        dequantize the direct FP8 checkpoint weight for the local GEMM, and use
        saved tensor hooks so backward stores only the FP8 weight handle.
        """
        from megatron.bridge.orbit.quant.fp8_utils import dequant_fp8

        w_fp8 = self.linear.weight.data
        scale_inv = getattr(self._orig_module, "weight_scale_inv", None)
        has_scale_inv = scale_inv is not None
        if scale_inv is None:
            scale_inv = torch.ones(1, device=w_fp8.device, dtype=torch.float32)

        _oft_fp8_debug_log(
            "split_ln",
            wrapper=type(self).__name__,
            orig_module=type(self._orig_module).__name__,
            weight=w_fp8,
            scale_inv=scale_inv,
            has_scale_inv=has_scale_inv,
            input=x,
            te_return_bias=getattr(self._orig_module, "te_return_bias", False),
            sequence_parallel=getattr(self._orig_module, "sequence_parallel", False),
        )

        bias = _get_active_bias_tensor(self._orig_module)
        has_bias = bias is not None
        w_compute = None
        w_compute_ptr = None

        def get_w_compute() -> torch.Tensor:
            nonlocal w_compute, w_compute_ptr
            if w_compute is None:
                w_compute = dequant_fp8(w_fp8, scale_inv, out_dtype=x.dtype)
                w_compute_ptr = w_compute.data_ptr()
            return w_compute

        x = _prepare_raw_column_parallel_input(self._orig_module, x)
        te_return_bias = getattr(self._orig_module, "te_return_bias", False)
        native_bias = None if te_return_bias else (bias if has_bias else None)

        def pack(tensor):
            if w_compute_ptr is not None and tensor.data_ptr() == w_compute_ptr:
                return (
                    w_fp8,
                    scale_inv,
                    tensor.dtype,
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    tensor.storage_offset(),
                )
            return tensor

        def unpack(packed):
            if isinstance(packed, tuple) and len(packed) == 6:
                fp8, sinv, dtype, saved_shape, saved_stride, saved_storage_offset = packed
                base = dequant_fp8(fp8, sinv, out_dtype=dtype)
                return base.as_strided(saved_shape, saved_stride, saved_storage_offset)
            return packed

        try:
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                if should_attempt_qwen3_native_fp8_gemm():
                    try:
                        native_output = maybe_qwen3_native_block_fp8_linear(
                            x,
                            w_fp8,
                            scale_inv,
                            bias=native_bias,
                            module_name=f"{type(self._orig_module).__name__}.split_ln",
                        )
                    except ValueError:
                        backend = os.environ.get("MEGATRON_QWEN3_FP8_GEMM_BACKEND", "auto").strip().lower()
                        if backend != "auto":
                            raise
                        native_output = None
                    if native_output is not None:
                        if te_return_bias and has_bias:
                            return native_output, bias
                        return native_output, None

                if _get_oft_fp8_activation_quant_mode() == "w8a8":
                    x = _fp8_activation_qdq_per_token_group_ste(x)

                w_compute = get_w_compute()
                x = x.to(w_compute.dtype)
                if te_return_bias and has_bias:
                    out = F.linear(x, w_compute, None)
                    return out, bias
                out = F.linear(x, w_compute, bias if has_bias else None)
                return out, None
        finally:
            if w_compute is not None:
                del w_compute

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError as exc:
            orig_module = self._get_orig_module()
            if orig_module is None:
                raise exc
            return getattr(orig_module, name)

    def _get_orig_module(self):
        modules = self.__dict__.get("_modules", {})
        if "_orig_module" in modules:
            return modules["_orig_module"]
        return self.__dict__.get("_orig_module")


@dataclass
class OFT(OrbitPEFTMixin, PEFT, ModuleMatcher):
    """
    Implements OFT (Orthogonal Fine-Tuning) for parameter-efficient fine-tuning.

    OFT learns block-diagonal orthogonal rotations applied to the input of linear layers.
    The orthogonal constraint preserves the angular structure of the pretrained weight space,
    which can improve generalization compared to unconstrained low-rank methods.

    Args:
        target_modules (List[str], optional): A list of module names to apply OFT to.
            Defaults to all linear layers ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2'].
            Target modules can also contain wildcards, e.g.
                target_modules=['*.layers.0.*.linear_qkv'] to add OFT to only linear_qkv
                on the first layer.
        exclude_modules (List[str], optional): A list of module names not to apply OFT to.
        r (int): Number of OFT blocks per adapted layer. Mutually exclusive with block_size.
            Higher r = smaller blocks = fewer parameters. Default: 0 (use block_size instead).
        block_size (int): Size of each orthogonal block. Mutually exclusive with r.
            Default: 32.
        coft (bool): Whether to use Constrained OFT, which projects rotations
            to an epsilon-ball around identity. Default: False.
        eps (float): Epsilon controlling rotation strength for COFT. Default: 6e-5.
        block_share (bool): Whether all blocks share parameters. Default: False.
        module_dropout (float): Probability of replacing an OFT block with identity
            during training (multiplicative dropout). Default: 0.0.
    """

    target_modules: list[str] = field(
        default_factory=lambda: ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    )
    r: int = 0
    block_size: int = 32
    coft: bool = False
    eps: float = 6e-5
    block_share: bool = False
    module_dropout: float = 0.0

    def __post_init__(self) -> None:
        _validate_oft_hyperparameters(
            r=self.r,
            block_size=self.block_size,
            coft=self.coft,
            eps=self.eps,
            module_dropout=self.module_dropout,
        )

    def transform(self, module: nn.Module, name: str | None = None, prefix: str | None = None) -> nn.Module:
        """
        Applies OFT to a specific module within the model architecture.

        Args:
            module (nn.Module): The module to apply OFT to.
            name (str, optional): Name of the module. Defaults to None.
            prefix (str, optional): Prefix for the module name. Defaults to None.

        Returns:
            nn.Module: The modified module with OFT applied, or the original module if not a target.
        """
        # Skip already transformed modules
        if isinstance(module, (OFTLinear, OFTTopKRouter, _SplitLNOFTLinear)):
            return module

        if (ans := self.match(module, name, prefix)) is not None:
            (match, full_name) = ans
            module_leaf = full_name.rsplit(".", 1)[-1]

            if module_leaf in ("output_layer", "word_embeddings"):
                raise NotImplementedError(
                    f"--oft-type oft (legacy shared-R OFT) does not support OFT on "
                    f"{module_leaf!r} (matched by {match!r} at {full_name}). "
                    f"Explicitly select --oft-type canonical_oft, which supports the all-mode targets."
                )

            # Fused LN+Linear layers (TELayerNormColumnParallelLinear):
            # OFT needs to insert rotation between LN and GEMM. The fused
            # TE custom autograd Function cannot track OFT parameters captured
            # inside its forward boundary. De-fuse the module so the rotation
            # remains an ordinary nn.Module tracked by autograd.
            if (
                HAVE_TE_LN_COL_LINEAR
                and isinstance(TELayerNormColumnParallelLinear, type)
                and isinstance(module, TELayerNormColumnParallelLinear)
            ):
                logger.warning(
                    f"OFT on fused TELayerNormColumnParallelLinear ({full_name}): "
                    f"de-fusing to separate LN + OFTLinear to ensure correct gradient flow. "
                    f"Consider using `config.model.transformer_layer_spec = local_layer_spec` "
                    f"for a cleaner setup."
                )
                model_parallel_config = getattr(module, "config", None)
                is_expert = is_expert_linear(full_name)
                attrs = get_oft_adapter_attributes_from_linear(module, is_expert=is_expert)
                assert not attrs.input_is_parallel

                adapter = OFTRotationModule(
                    in_features=attrs.in_features,
                    r=self.r,
                    block_size=self.block_size,
                    coft=self.coft,
                    eps=self.eps,
                    block_share=self.block_share,
                    module_dropout=self.module_dropout,
                    model_parallel_config=model_parallel_config,
                    input_is_parallel=attrs.input_is_parallel,
                    is_expert=is_expert,
                )
                return _SplitLNOFTLinear(module, adapter)

            # Determine input features for the rotation module.
            model_parallel_config = getattr(module, "config", None)

            rotation_r = self.r
            if model_parallel_config is not None:
                # Megatron parallel linear — use get_adapter_attributes_from_linear
                is_expert = is_expert_linear(full_name)
                attrs = get_oft_adapter_attributes_from_linear(module, is_expert=is_expert)

                # attrs.in_features is the FULL (un-sharded) dimension.
                # For RowParallel (input_is_parallel), the rotation operates on
                # the TP-local shard, so we divide by TP size.
                if attrs.input_is_parallel:
                    if is_expert:
                        tp_size = parallel_state.get_expert_tensor_parallel_world_size()
                    else:
                        tp_size = parallel_state.get_tensor_model_parallel_world_size()
                    rotation_in_features, rotation_r = _localize_oft_row_parallel_geometry(
                        attrs.in_features, self.r, tp_size
                    )
                else:
                    rotation_in_features = attrs.in_features

                input_is_parallel = attrs.input_is_parallel
            else:
                # Plain nn.Linear or other non-Megatron linear
                is_expert = False
                input_is_parallel = False
                rotation_in_features = module.in_features

            logging.debug(
                f"Adding OFT to: {full_name} (in_features={rotation_in_features}, "
                f"input_is_parallel={input_is_parallel})"
            )

            adapter = OFTRotationModule(
                in_features=rotation_in_features,
                r=rotation_r,
                block_size=self.block_size,
                coft=self.coft,
                eps=self.eps,
                block_share=self.block_share,
                module_dropout=self.module_dropout,
                model_parallel_config=model_parallel_config,
                input_is_parallel=input_is_parallel,
                is_expert=is_expert,
            )
            if isinstance(module, TopKRouter):
                return OFTTopKRouter(module, adapter)
            return OFTLinear(module, adapter)
        return module


@dataclass
class VLMOFT(OFT):
    """
    Implements OFT for Vision-Language Models.
    VLMOFT additionally allows the user to specify whether the language or vision
    models should be frozen.
    For example, a common finetuning workload for multimodal models is to apply adapters to language model and fully
    finetune the vision model.
    """

    freeze_vision_model: bool = True
    freeze_vision_projection: bool = True
    freeze_language_model: bool = True

    def freeze_model(self, model: nn.Module, training: bool = True) -> None:
        unwrapped = unwrap_model(model)
        model_chunks = unwrapped if isinstance(unwrapped, list) else [unwrapped]

        for model_chunk in model_chunks:
            vlm = getattr(model_chunk, "llava_model", model_chunk)
            components = (
                (self.freeze_vision_model, getattr(vlm, "vision_model", None)),
                (self.freeze_vision_projection, getattr(vlm, "vision_projection", None)),
                (self.freeze_language_model, getattr(vlm, "language_model", None)),
            )
            for should_freeze, component in components:
                if should_freeze and component is not None:
                    for param in component.parameters():
                        param.requires_grad = False

        if training:
            for model_chunk in model_chunks:
                model_chunk.train(mode=True)


def _collect_oft_merge_wrappers(model, wrapper_types: tuple[type[nn.Module], ...]) -> list[nn.Module]:
    """Collect wrapper roots without descending into their owned base/adapter children."""
    wrappers: list[nn.Module] = []
    seen: set[int] = set()

    def visit(module: nn.Module) -> None:
        if id(module) in seen:
            return
        seen.add(id(module))
        if isinstance(module, wrapper_types):
            wrappers.append(module)
            return
        for child in module._modules.values():
            if child is not None:
                visit(child)

    roots = model if isinstance(model, list) else [model]
    for root in roots:
        visit(root)
    return wrappers


@dataclass(frozen=True)
class _OFTMergeUpdate:
    """One already-computed dense weight replacement for an OFT merge."""

    holder: nn.Module
    name: str
    weight: torch.Tensor
    merged_weight: torch.Tensor


@dataclass(frozen=True)
class _OFTWrapperMergePlan:
    """The mutation-free result of preparing one wrapper for merge."""

    wrapper: nn.Module
    replacement: nn.Module
    updates: tuple[_OFTMergeUpdate, ...]


def _tensor_storage_span(tensor: torch.Tensor) -> tuple[int, int] | None:
    """Return the conservative half-open byte span touched by a strided tensor."""
    if tensor.numel() == 0:
        return None
    storage_start = tensor.untyped_storage().data_ptr()
    if storage_start == 0:
        return None
    min_offset = max_offset = tensor.storage_offset()
    for size, stride in zip(tensor.shape, tensor.stride()):
        extent = (size - 1) * stride
        min_offset += min(0, extent)
        max_offset += max(0, extent)
    element_size = tensor.element_size()
    return (
        storage_start + min_offset * element_size,
        storage_start + (max_offset + 1) * element_size,
    )


def _tensors_share_storage(first: torch.Tensor, second: torch.Tensor) -> bool:
    """Return whether two live strided tensors touch overlapping storage bytes."""
    if first is second:
        return True
    if first.device != second.device or first.layout != torch.strided or second.layout != torch.strided:
        return False
    try:
        first_span = _tensor_storage_span(first)
        second_span = _tensor_storage_span(second)
    except (NotImplementedError, RuntimeError, TypeError):
        return False
    if first_span is None or second_span is None:
        return False
    return max(first_span[0], second_span[0]) < min(first_span[1], second_span[1])


def _surviving_registered_tensors(
    model,
    plans: tuple[_OFTWrapperMergePlan, ...],
) -> list[tuple[_OFTWrapperMergePlan | None, nn.Module, str, torch.Tensor]]:
    """Collect tensor owners that remain after every planned wrapper is removed.

    Split fused-LN wrappers deliberately contain temporary linears sharing the
    original module's parameters. Traversing the replacement tree, rather than
    every child of the old wrapper, ignores those discarded implementation
    aliases while retaining any consumer reachable outside the wrapper. Module
    identity alone cannot identify a consumer: the same base module may also be
    reachable through an untargeted path, so each acyclic path is retained with
    the wrapper plan (if any) that introduced its replacement.
    """
    plan_by_wrapper = {id(plan.wrapper): plan for plan in plans}
    tensors: list[tuple[_OFTWrapperMergePlan | None, nn.Module, str, torch.Tensor]] = []
    active: set[int] = set()

    def visit(module: nn.Module, replacement_plan: _OFTWrapperMergePlan | None = None) -> None:
        module_id = id(module)
        if module_id in active:
            return
        active.add(module_id)
        try:
            plan = plan_by_wrapper.get(module_id)
            if plan is not None:
                visit(plan.replacement, plan)
                return
            for collection in (module._parameters, module._buffers):
                for name, tensor in collection.items():
                    if isinstance(tensor, torch.Tensor):
                        tensors.append((replacement_plan, module, name, tensor))
            for child in module._modules.values():
                if child is not None:
                    visit(child, replacement_plan)
        finally:
            active.remove(module_id)

    roots = model if isinstance(model, list) else [model]
    for root in roots:
        visit(root)
    return tensors


def _validate_unaliased_merge_weights(model, plans: tuple[_OFTWrapperMergePlan, ...]) -> None:
    """Reject a merge target used by another surviving tensor consumer.

    A destructive OFT fold changes the semantics of the underlying Parameter.
    Applying it to a tied embedding/output weight would therefore also change
    an untargeted consumer, while two targeted consumers may require different
    rotation orientation or values. There is no generally correct local fold,
    so fail before copying any prepared weight.
    """
    updates = [update for plan in plans for update in plan.updates]
    for index, update in enumerate(updates):
        for other in updates[index + 1 :]:
            if _tensors_share_storage(update.weight, other.weight):
                raise ValueError(
                    "OFT merge refuses aliased merge targets with shared storage: "
                    f"{type(update.holder).__name__}.{update.name} and "
                    f"{type(other.holder).__name__}.{other.name}"
                )

    surviving = _surviving_registered_tensors(model, plans)
    for plan in plans:
        for update in plan.updates:
            for replacement_plan, owner, name, tensor in surviving:
                if replacement_plan is plan and owner is update.holder and name == update.name:
                    continue
                if _tensors_share_storage(update.weight, tensor):
                    raise ValueError(
                        "OFT merge refuses a target with shared storage owned by another surviving consumer: "
                        f"{type(update.holder).__name__}.{update.name} aliases "
                        f"{type(owner).__name__}.{name}"
                    )


@torch.no_grad()
def _apply_oft_merge_plan(plan: _OFTWrapperMergePlan) -> nn.Module:
    """Copy a fully prepared plan and return the wrapper replacement."""
    for update in plan.updates:
        update.weight.copy_(update.merged_weight)
    return plan.replacement


def _replace_oft_merge_wrappers(model, wrapper_types: tuple[type[nn.Module], ...], merge_wrapper):
    """Replace adapter wrappers while preserving aliases and avoiding wrapper-child grafting.

    The generic PEFT walker iterates the *old* wrapper's children after a
    transform returns a new base module, then attaches those children to the
    replacement. A destructive merge needs a replacement-aware traversal that
    treats each recognized wrapper as a leaf.
    """
    replacements: dict[int, nn.Module] = {}

    def replace(module: nn.Module) -> nn.Module:
        module_id = id(module)
        if module_id in replacements:
            return replacements[module_id]
        if isinstance(module, wrapper_types):
            replacement = merge_wrapper(module)
            replacements[module_id] = replacement
            return replacement

        replacements[module_id] = module
        for name, child in list(module._modules.items()):
            if child is None:
                continue
            replacement = replace(child)
            if replacement is not child:
                module._modules[name] = replacement
        return module

    if isinstance(model, list):
        for index, model_chunk in enumerate(model):
            model[index] = replace(model_chunk)
        return model
    return replace(model)


def _set_oft_merged_model_mode(model, training: bool) -> None:
    """Set train/eval mode on a single merged model or every pipeline chunk."""
    model_chunks = model if isinstance(model, list) else [model]
    for model_chunk in model_chunks:
        model_chunk.train(mode=training)


@dataclass
class OFTMerge(OrbitPEFTMixin, PEFT):
    """
    Merges the learned OFT rotation into the base weight: W_merged = W @ R.T.

    Tensor-parallelism handling:
        Unlike LoRA merge which requires all-gather to reconstruct full-rank matrices,
        OFT merge works locally on each TP rank without communication:

        - ColumnParallelLinear (linear_qkv, linear_fc1):
            W: [out/TP, in], R: [in, in]
            W_merged = W @ R.T operates on the full input dimension — no gather needed.

        - RowParallelLinear (linear_proj, linear_fc2):
            W: [out, in/TP], R: [in/TP, in/TP]
            R is already sized for the local input shard (set in OFT.transform),
            so W_merged = W @ R.T operates on the local shard — no gather needed.
    """

    _WRAPPER_TYPES = (OFTLinear, OFTTopKRouter, _SplitLNOFTLinear, TEOFTLayerNormLinear)

    @staticmethod
    def _merge_weight_or_raise(weight_holder: nn.Module, name: str, wrapper: nn.Module) -> torch.Tensor:
        """Return a mergeable dense weight or fail before any model mutation."""
        weight = getattr(weight_holder, name, None)
        suffix = name[len("weight") :] if name.startswith("weight") else ""
        quantized_markers = (
            f"{name}_packed",
            f"{name}_w_packed",
            f"{name}_v_packed",
            f"{name}_scale",
            f"{name}_shape",
            f"{name}_scale_inv",
            f"weight_scale{suffix}",
            f"weight_double_scale{suffix}",
        )
        has_quantized_state = getattr(weight_holder, "weight_quantizer", None) is not None or any(
            getattr(weight_holder, marker, None) is not None for marker in quantized_markers
        )
        dtype_name = str(getattr(weight, "dtype", ""))
        if (
            has_quantized_state
            or dtype_name.startswith("torch.float8")
            or getattr(weight, "dtype", None) == torch.uint8
        ):
            raise ValueError(
                f"OFT merge does not support quantized weight {name!r} on {type(wrapper).__name__}; "
                "dequantize the base model before merging"
            )
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or not weight.is_floating_point():
            raise ValueError(
                f"OFT merge requires a floating 2-D {name!r} on {type(wrapper).__name__}, "
                f"got {type(weight).__name__} with shape {getattr(weight, 'shape', None)}"
            )
        if weight.device.type == "meta":
            raise ValueError(f"OFT merge cannot mutate meta-device weight {name!r} on {type(wrapper).__name__}")
        return weight

    @staticmethod
    def _validate_rotation_shape(adapter: nn.Module, weight: torch.Tensor, wrapper: nn.Module, name: str) -> None:
        """Require the adapter rotation to span the base weight's local input axis."""
        in_features = getattr(adapter, "in_features", None)
        if in_features != weight.shape[1]:
            raise ValueError(
                f"OFT merge shape mismatch for {type(wrapper).__name__}.{name}: "
                f"adapter input is {in_features}, weight input is {weight.shape[1]}"
            )

    @classmethod
    def _merge_parts(cls, wrapper: nn.Module):
        """Resolve the replacement base, weight owner/names, and rotation module."""
        if isinstance(wrapper, OFTTopKRouter):
            return wrapper.to_wrap, wrapper.to_wrap.gating, ["weight"], wrapper.adapter
        if isinstance(wrapper, (_SplitLNOFTLinear, TEOFTLayerNormLinear)):
            return wrapper._orig_module, wrapper._orig_module, ["weight"], wrapper.adapter
        weight_names = list(getattr(wrapper, "_weight_names", ()))
        if not weight_names:
            raise ValueError(f"OFT merge found no base weights on {type(wrapper).__name__}")
        return wrapper.to_wrap, wrapper.to_wrap, weight_names, wrapper.adapter

    @classmethod
    def _validate_wrapper(cls, wrapper: nn.Module) -> None:
        """Validate every weight owned by one supported legacy wrapper."""
        _, weight_holder, weight_names, adapter = cls._merge_parts(wrapper)
        for name in weight_names:
            weight = cls._merge_weight_or_raise(weight_holder, name, wrapper)
            cls._validate_rotation_shape(adapter, weight, wrapper, name)

    @classmethod
    @torch.no_grad()
    def _prepare_wrapper(cls, wrapper: nn.Module) -> _OFTWrapperMergePlan:
        """Compute every legacy merged weight without mutating the model."""
        base_module, weight_holder, weight_names, adapter = cls._merge_parts(wrapper)
        cls._validate_wrapper(wrapper)
        rotations: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        updates: list[_OFTMergeUpdate] = []
        for name in weight_names:
            base_weight = getattr(weight_holder, name)
            cache_key = (base_weight.device, base_weight.dtype)
            if cache_key not in rotations:
                rotation = adapter.get_delta_weight().to(
                    device=base_weight.device,
                    dtype=base_weight.dtype,
                )
                expected_shape = (base_weight.shape[1], base_weight.shape[1])
                if rotation.ndim != 2 or tuple(rotation.shape) != expected_shape:
                    raise ValueError(
                        f"OFT merge rotation shape mismatch for {type(wrapper).__name__}.{name}: "
                        f"expected {expected_shape}, got {tuple(rotation.shape)}"
                    )
                rotations[cache_key] = rotation
            merged_weight = base_weight @ rotations[cache_key].transpose(-1, -2)
            updates.append(
                _OFTMergeUpdate(
                    holder=weight_holder,
                    name=name,
                    weight=base_weight,
                    merged_weight=merged_weight,
                )
            )
        return _OFTWrapperMergePlan(wrapper=wrapper, replacement=base_module, updates=tuple(updates))

    @classmethod
    def _preflight_model(cls, model) -> tuple[_OFTWrapperMergePlan, ...]:
        """Prepare the complete merge set and reject aliases before any write."""
        wrappers = _collect_oft_merge_wrappers(model, cls._WRAPPER_TYPES)
        plans = tuple(cls._prepare_wrapper(wrapper) for wrapper in wrappers)
        _validate_unaliased_merge_weights(model, plans)
        return plans

    @classmethod
    @torch.no_grad()
    def _merge_wrapper(cls, wrapper: nn.Module) -> nn.Module:
        """Prepare and fold one wrapper for direct-call compatibility."""
        return _apply_oft_merge_plan(cls._prepare_wrapper(wrapper))

    def __call__(self, model, training: bool = True):
        """Atomically preflight, merge, unwrap, freeze, and return the model."""
        plans = self._preflight_model(model)
        plan_by_wrapper = {id(plan.wrapper): plan for plan in plans}
        merged = _replace_oft_merge_wrappers(
            model,
            self._WRAPPER_TYPES,
            lambda wrapper: _apply_oft_merge_plan(plan_by_wrapper[id(wrapper)]),
        )
        self.freeze_model(merged, training=training)
        _set_oft_merged_model_mode(merged, training)
        return merged

    @torch.no_grad()
    def transform(self, module: nn.Module, name: str | None = None, prefix: str | None = None) -> nn.Module:
        """Merge and unwrap one supported wrapper for direct-call compatibility."""
        if not isinstance(module, self._WRAPPER_TYPES):
            return module
        logging.debug(f"merging OFT {(prefix if prefix else '') + '.' + (name if name else '')}")
        self._validate_wrapper(module)
        return self._merge_wrapper(module)
