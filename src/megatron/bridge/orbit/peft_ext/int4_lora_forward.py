# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""INT4-dequant base forward for LoRA-wrapped linears (orbit fork).

Extracted from ``megatron.bridge.peft.lora_layers``; ``LoRALinear.forward``
calls :func:`_base_linear_forward_int4` (the first parameter is the
``LoRALinear`` instance) as its only seam.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _module_bias_enabled(module: nn.Module) -> bool:
    if hasattr(module, "use_bias"):
        return bool(getattr(module, "use_bias"))
    if hasattr(module, "apply_bias"):
        return bool(getattr(module, "apply_bias"))

    bias = getattr(module, "bias", None)
    if bias is None:
        return False
    if isinstance(bias, torch.Tensor) and bias.numel() == 0:
        return False
    return True


def _get_active_bias_tensor(module: nn.Module, name: str = "bias") -> Optional[torch.Tensor]:
    if not _module_bias_enabled(module):
        return None

    bias = getattr(module, name, None)
    if bias is None:
        return None
    if isinstance(bias, torch.Tensor) and bias.numel() == 0:
        return None
    return bias


def _apply_layernorm_if_present(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    ln_weight = getattr(module, "layer_norm_weight", None)
    if ln_weight is None:
        return x

    hidden_size = ln_weight.shape[0]
    ln_bias = getattr(module, "layer_norm_bias", None)
    if isinstance(ln_bias, torch.Tensor) and ln_bias.numel() == 0:
        ln_bias = None

    eps = getattr(module, "eps", 1e-5)
    normalization = getattr(module, "normalization", "LayerNorm")
    if normalization == "RMSNorm":
        return F.rms_norm(x, (hidden_size,), ln_weight, eps)
    return F.layer_norm(x, (hidden_size,), ln_weight, ln_bias, eps)


def _base_linear_forward_int4(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

    from megatron.bridge.orbit.low_precision.int4 import dequantize_int4

    layernorm_output = _apply_layernorm_if_present(self.to_wrap, x)
    linear_input = layernorm_output
    if getattr(self.to_wrap, "sequence_parallel", False):
        linear_input = gather_from_sequence_parallel_region(linear_input, tensor_parallel_output_grad=True)

    packed = self.to_wrap.weight_packed
    scale = self.to_wrap.weight_scale
    shape = self.to_wrap.weight_shape
    w_compute = dequantize_int4(packed, scale, shape, device=linear_input.device).to(linear_input.dtype)
    w_compute_ptr = w_compute.data_ptr()
    bias = _get_active_bias_tensor(self.to_wrap)
    has_bias = bias is not None

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
            if getattr(self.to_wrap, "te_return_bias", False) and has_bias:
                return F.linear(linear_input.to(w_compute.dtype), w_compute, None), bias, layernorm_output
            return (
                F.linear(linear_input.to(w_compute.dtype), w_compute, bias if has_bias else None),
                None,
                layernorm_output,
            )
    finally:
        del w_compute
