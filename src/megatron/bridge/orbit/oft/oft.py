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

"""OFT (Orthogonal Fine-Tuning) PEFT method for megatron-bridge.

OFT fine-tunes models by learning orthogonal rotations applied to the input
of linear layers. Unlike LoRA which adds a low-rank residual (W' = W + BA),
OFT applies a multiplicative orthogonal transform (y = W @ R @ x).

Reference: https://arxiv.org/abs/2306.07280
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.utils import unwrap_model

from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.module_matcher import ModuleMatcher
from megatron.bridge.orbit.oft.oft_layers import (
    OFTLinear,
    OFTRotationModule,
    OFTTopKRouter,
    _clear_disabled_bias_parameters,
    _fp8_activation_qdq_per_token_group_ste,
    _get_active_bias_tensor,
    _get_oft_fp8_activation_quant_mode,
    _is_direct_fp8_runtime_weight,
    _module_bias_enabled,
    _oft_fp8_debug_log,
)
from megatron.bridge.orbit.quant.qwen3_fp8_gemm import (
    maybe_qwen3_native_block_fp8_linear,
    should_attempt_qwen3_native_fp8_gemm,
)
from megatron.bridge.peft.utils import get_adapter_attributes_from_linear, is_expert_linear
from megatron.bridge.utils.import_utils import safe_import_from


logger = logging.getLogger(__name__)

TELayerNormColumnParallelLinear, HAVE_TE_LN_COL_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine",
    "TELayerNormColumnParallelLinear",
)

TEOFTLayerNormColumnParallelLinear, HAVE_TE_OFT_LN = safe_import_from(
    "megatron.bridge.orbit.oft.te_oft",
    "TEOFTLayerNormColumnParallelLinear",
)


class _SplitLNOFTLinear(nn.Module):
    """Splits a fused TELayerNormColumnParallelLinear into LN + OFT + Linear.

    The fused TEOFTLayerNormColumnParallelLinear uses a custom autograd.Function
    that severs gradient flow to OFT parameters. This module splits the fused
    layer into three separate operations:
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
        out_features, in_features = orig_module.weight.shape if orig_module.weight.numel() > 0 else (
            getattr(orig_module, 'out_features', 0),
            getattr(orig_module, 'in_features', 0),
        )
        # Get the actual sizes from config
        tp_size = getattr(orig_module, 'tp_size', 1)
        has_bias = _module_bias_enabled(orig_module)

        self.linear = TEColumnParallelLinear(
            input_size=in_features * tp_size,  # full size before TP split
            output_size=out_features * tp_size,
            config=config,
            init_method=lambda w: None,  # no init, we'll share weights
            gather_output=False,
            bias=has_bias,
            skip_bias_add=getattr(orig_module, 'te_return_bias', False),
            is_expert=False,
            tp_comm_buffer_name=getattr(orig_module, 'ub_name', None),
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
        from megatron.core.tensor_parallel.mappings import (
            gather_from_sequence_parallel_region,
        )

        packed = self._orig_module.weight
        quantizer = self._orig_module.weight_quantizer
        scale = quantizer._scale
        scale_2 = quantizer._double_scale
        shape = self._nvfp4_weight_shape(packed, scale)

        w_compute = dequantize_nvfp4(packed, scale, scale_2, shape, dtype=x.dtype, device=x.device)
        w_compute_ptr = w_compute.data_ptr()
        bias = _get_active_bias_tensor(self._orig_module)
        has_bias = bias is not None

        if getattr(self._orig_module, "sequence_parallel", False):
            x = gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=True)

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
        from megatron.core.tensor_parallel.mappings import (
            gather_from_sequence_parallel_region,
        )

        packed = self._orig_module.weight_packed
        scale = self._orig_module.weight_scale
        shape = self._orig_module.weight_shape

        w_compute = dequantize_int4(packed, scale, shape, device=x.device).to(x.dtype)
        w_compute_ptr = w_compute.data_ptr()
        bias = _get_active_bias_tensor(self._orig_module)
        has_bias = bias is not None

        if getattr(self._orig_module, "sequence_parallel", False):
            x = gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=True)

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
        from megatron.core.tensor_parallel.mappings import (
            gather_from_sequence_parallel_region,
        )

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

        if getattr(self._orig_module, "sequence_parallel", False):
            x = gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=True)

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
class OFT(PEFT, ModuleMatcher):
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

    target_modules: List[str] = field(
        default_factory=lambda: ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    )
    r: int = 0
    block_size: int = 32
    coft: bool = False
    eps: float = 6e-5
    block_share: bool = False
    module_dropout: float = 0.0

    def transform(self, module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None) -> nn.Module:
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
        if HAVE_TE_OFT_LN and isinstance(module, TEOFTLayerNormColumnParallelLinear):
            return module

        if (ans := self.match(module, name, prefix)) is not None:
            (match, full_name) = ans

            if match in ("output_layer", "word_embeddings"):
                raise NotImplementedError(
                    f"--oft-type oft (legacy shared-R OFT) does not support OFT on "
                    f"{match!r} (matched at {full_name}). Use --oft-type canonical_oft "
                    f"(the default) which supports the all-mode targets."
                )

            # Fused LN+Linear layers (TELayerNormColumnParallelLinear):
            # OFT needs to insert rotation between LN and GEMM. The fused
            # TEOFTLayerNormColumnParallelLinear wrapper uses a custom autograd
            # Function that severs gradient flow to OFT parameters (the OFT
            # rotation runs inside the Function boundary, but backward returns
            # None for OFT params). Instead, we de-fuse the module by extracting
            # the underlying Linear and wrapping it with OFTLinear, which is a
            # normal nn.Module where autograd works correctly.
            if HAVE_TE_LN_COL_LINEAR and isinstance(module, TELayerNormColumnParallelLinear):
                logger.warning(
                    f"OFT on fused TELayerNormColumnParallelLinear ({full_name}): "
                    f"de-fusing to separate LN + OFTLinear to ensure correct gradient flow. "
                    f"Consider using `config.model.transformer_layer_spec = local_layer_spec` "
                    f"for a cleaner setup."
                )
                model_parallel_config = getattr(module, "config", None)
                is_expert = is_expert_linear(full_name)
                attrs = get_adapter_attributes_from_linear(module, is_expert=is_expert, adapter_type="oft")
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

            if model_parallel_config is not None:
                # Megatron parallel linear — use get_adapter_attributes_from_linear
                is_expert = is_expert_linear(full_name)
                attrs = get_adapter_attributes_from_linear(module, is_expert=is_expert, adapter_type="oft")

                # attrs.in_features is the FULL (un-sharded) dimension.
                # For RowParallel (input_is_parallel), the rotation operates on
                # the TP-local shard, so we divide by TP size.
                if attrs.input_is_parallel:
                    if is_expert:
                        tp_size = parallel_state.get_expert_tensor_parallel_world_size()
                    else:
                        tp_size = parallel_state.get_tensor_model_parallel_world_size()
                    rotation_in_features = attrs.in_features // tp_size
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
                r=self.r,
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
        modules_to_freeze = []

        model = unwrap_model(model)[0]
        if hasattr(model, "llava_model"):
            model = model.llava_model

        if self.freeze_vision_model and model.vision_model is not None:
            modules_to_freeze.append(model.vision_model)
        if self.freeze_vision_projection and model.vision_projection is not None:
            modules_to_freeze.append(model.vision_projection)
        if self.freeze_language_model and model.language_model is not None:
            modules_to_freeze.append(model.language_model)

        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

        if training:
            if isinstance(model, list):
                for model_chunk in model:
                    model_chunk.train(mode=True)
            elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model.module.train(mode=True)
            else:
                model.train(mode=True)


@dataclass
class OFTMerge(PEFT):
    """
    Merges the learned OFT rotation into the base weight: W_merged = W @ R.

    Tensor-parallelism handling:
        Unlike LoRA merge which requires all-gather to reconstruct full-rank matrices,
        OFT merge works locally on each TP rank without communication:

        - ColumnParallelLinear (linear_qkv, linear_fc1):
            W: [out/TP, in], R: [in, in]
            W_merged = W @ R operates on the full input dimension — no gather needed.

        - RowParallelLinear (linear_proj, linear_fc2):
            W: [out, in/TP], R: [in/TP, in/TP]
            R is already sized for the local input shard (set in OFT.transform),
            so W_merged = W @ R operates on the local shard — no gather needed.
    """

    @torch.no_grad()
    def transform(self, module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None) -> nn.Module:
        if not isinstance(module, (OFTLinear, OFTTopKRouter)):
            return module
        logging.debug(f"merging OFT {(prefix if prefix else '') + '.' + (name if name else '')}")

        # For TopKRouter, the weight lives in the gating sub-module
        if isinstance(module, OFTTopKRouter):
            weight_holder = module.to_wrap.gating
        else:
            weight_holder = module.to_wrap

        if hasattr(weight_holder, "weight"):
            base_weight = weight_holder.weight
            R = module.adapter.get_delta_weight().to(base_weight.device, base_weight.dtype)

            # Validate shapes: R should be [in, in] matching weight's input dim
            assert R.shape[0] == base_weight.shape[1], (
                f"Shape mismatch: R is {R.shape} but weight input dim is {base_weight.shape[1]}. "
                f"This may indicate a tensor-parallelism configuration issue."
            )

            # W_merged = W @ R  (weight is [out, in], R is [in, in])
            weight_holder.weight.data = base_weight @ R
        return module
