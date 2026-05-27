# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Qwen3 native block-FP8 GEMM helpers for OFT."""

from __future__ import annotations

import functools
import math
import os
from typing import Callable

import torch
import torch.nn.functional as F

from megatron.bridge.peft.fp8_utils import dequant_fp8


QWEN3_FP8_BLOCK_SIZE = (128, 128)
_VALID_BACKENDS = {"auto", "qdq", "sglang_native"}
_BACKEND_ENV = "MEGATRON_QWEN3_FP8_GEMM_BACKEND"
_AUTO_UNAVAILABLE_SENTINEL = "__QWEN3_NATIVE_FP8_AUTO_UNAVAILABLE__"
_SGLANG_NATIVE_LINEAR_IMPORT_ERROR: BaseException | None = None


def get_qwen3_fp8_gemm_backend() -> str:
    """Return the configured Qwen3 FP8 GEMM backend."""
    backend = os.environ.get(_BACKEND_ENV, "auto").strip().lower()
    if backend in {"", "0", "false", "off"}:
        return "qdq"
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"Unsupported {_BACKEND_ENV}={backend!r}. "
            "Expected one of: auto, qdq, sglang_native."
        )
    return backend


def should_attempt_qwen3_native_fp8_gemm() -> bool:
    """Return whether callers should try the native block-FP8 path."""
    return get_qwen3_fp8_gemm_backend() != "qdq"


@functools.lru_cache(maxsize=1)
def _load_sglang_native_linear() -> Callable[..., torch.Tensor] | None:
    """Import SGLang's native block-FP8 linear helper if available."""
    global _SGLANG_NATIVE_LINEAR_IMPORT_ERROR
    try:
        from sglang.srt.layers.quantization.fp8_utils import (
            triton_w8a8_block_fp8_linear,
        )
    except Exception as exc:
        _SGLANG_NATIVE_LINEAR_IMPORT_ERROR = exc
        return None
    _SGLANG_NATIVE_LINEAR_IMPORT_ERROR = None
    return triton_w8a8_block_fp8_linear


def _block_scale_shape(
    weight: torch.Tensor,
    block_size: tuple[int, int],
) -> tuple[int, int]:
    """Return the block-scale matrix shape for a 2D weight."""
    block_n, block_k = block_size
    return (
        max(1, math.ceil(int(weight.shape[0]) / block_n)),
        max(1, math.ceil(int(weight.shape[1]) / block_k)),
    )


def _normalize_weight_scale(
    weight_scale_inv: torch.Tensor,
    weight: torch.Tensor,
    block_size: tuple[int, int],
) -> torch.Tensor:
    """Normalize scalar or block-wise scale tensors for the native kernel."""
    expected = _block_scale_shape(weight, block_size)
    scale = weight_scale_inv.detach()

    if scale.ndim == 0 or scale.numel() == 1:
        return (
            scale.reshape(1)
            .to(device=weight.device, dtype=torch.float32)
            .expand(expected)
            .contiguous()
        )

    if scale.ndim != 2:
        raise ValueError(
            "Qwen3 native FP8 GEMM expects scalar or 2D weight_scale_inv, "
            f"got shape={tuple(scale.shape)}."
        )

    if tuple(scale.shape) != expected:
        raise ValueError(
            "Qwen3 native FP8 GEMM weight_scale_inv shape mismatch: "
            f"expected={expected} got={tuple(scale.shape)} "
            f"weight_shape={tuple(weight.shape)}."
        )

    return scale.to(device=weight.device, dtype=torch.float32).contiguous()


def _validate_inputs(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    module_name: str,
    block_size: tuple[int, int],
) -> None:
    """Validate the subset of shapes supported by Qwen3 native block-FP8 GEMM."""
    if block_size != QWEN3_FP8_BLOCK_SIZE:
        raise ValueError(
            f"{module_name}: only block size {QWEN3_FP8_BLOCK_SIZE} is supported, "
            f"got {block_size}."
        )
    if not input.is_floating_point():
        raise ValueError(f"{module_name}: input must be a floating point tensor.")
    if input.dim() < 1:
        raise ValueError(f"{module_name}: input must have at least one dimension.")
    if weight.dim() != 2:
        raise ValueError(f"{module_name}: weight must be 2D, got shape={tuple(weight.shape)}.")
    if weight.numel() == 0 or weight.shape[0] == 0 or weight.shape[1] == 0:
        raise ValueError(f"{module_name}: weight must be non-empty, got shape={tuple(weight.shape)}.")
    if input.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"{module_name}: input last dim {input.shape[-1]} must match "
            f"weight input dim {weight.shape[1]}."
        )
    if input.shape[-1] % QWEN3_FP8_BLOCK_SIZE[1] != 0:
        raise ValueError(
            f"{module_name}: input last dim {input.shape[-1]} must be divisible by "
            f"{QWEN3_FP8_BLOCK_SIZE[1]}."
        )
    if not isinstance(weight_scale_inv, torch.Tensor):
        raise ValueError(f"{module_name}: weight_scale_inv must be a tensor.")


def _as_fp8_weight(weight: torch.Tensor) -> torch.Tensor:
    """Return a contiguous ``float8_e4m3fn`` weight tensor."""
    if weight.dtype == torch.float8_e4m3fn:
        return weight.contiguous()
    return weight.to(torch.float8_e4m3fn).contiguous()


def _dequant_fp8_qwen3(
    weight: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize Qwen3 FP8 weights, including ceil-shaped tail block scales."""
    if scale.numel() == 1:
        return dequant_fp8(weight, scale, out_dtype=out_dtype)

    block_n, block_k = QWEN3_FP8_BLOCK_SIZE
    if (
        weight.dim() == 2
        and scale.dim() == 2
        and weight.shape[0] == scale.shape[0] * block_n
        and weight.shape[1] == scale.shape[1] * block_k
    ):
        return dequant_fp8(weight, scale, out_dtype=out_dtype)

    if weight.dim() != 2 or scale.dim() != 2:
        raise ValueError(
            "Qwen3 FP8 dequant expects scalar scale or 2D block scales for a "
            f"2D weight, got weight_shape={tuple(weight.shape)} "
            f"scale_shape={tuple(scale.shape)}."
        )

    expected = _block_scale_shape(weight, QWEN3_FP8_BLOCK_SIZE)
    if tuple(scale.shape) != expected:
        raise ValueError(
            "Qwen3 FP8 dequant block scale shape mismatch: "
            f"expected={expected} got={tuple(scale.shape)} "
            f"weight_shape={tuple(weight.shape)}."
        )

    out_features, in_features = weight.shape
    weight_float = weight.float()
    scale_float = scale.to(device=weight.device, dtype=torch.float32)
    output = torch.empty_like(weight_float, dtype=torch.float32)

    for block_i in range(scale.shape[0]):
        row_start = block_i * block_n
        row_end = min(row_start + block_n, out_features)
        for block_j in range(scale.shape[1]):
            col_start = block_j * block_k
            col_end = min(col_start + block_k, in_features)
            output[row_start:row_end, col_start:col_end] = (
                weight_float[row_start:row_end, col_start:col_end]
                * scale_float[block_i, block_j]
            )

    return output.to(out_dtype)


class _Qwen3NativeBlockFp8LinearFn(torch.autograd.Function):
    """Autograd wrapper for native forward with local frozen-weight backward."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale_inv: torch.Tensor,
        bias: torch.Tensor | None,
        module_name: str,
    ) -> torch.Tensor:
        block_size = QWEN3_FP8_BLOCK_SIZE
        _validate_inputs(input, weight, weight_scale_inv, module_name, block_size)

        native_linear = _load_sglang_native_linear()
        if native_linear is None:
            backend = get_qwen3_fp8_gemm_backend()
            if backend == "auto":
                raise RuntimeError(_AUTO_UNAVAILABLE_SENTINEL)
            error = RuntimeError(
                f"{module_name}: {_BACKEND_ENV}=sglang_native requested, but "
                "SGLang triton_w8a8_block_fp8_linear is unavailable."
            )
            if _SGLANG_NATIVE_LINEAR_IMPORT_ERROR is not None:
                raise error from _SGLANG_NATIVE_LINEAR_IMPORT_ERROR
            raise error

        weight_fp8 = _as_fp8_weight(weight)
        scale = _normalize_weight_scale(weight_scale_inv, weight_fp8, block_size)
        output = native_linear(
            input=input.contiguous(),
            weight=weight_fp8,
            block_size=list(block_size),
            weight_scale=scale,
            input_scale=None,
            bias=bias,
        )

        ctx.save_for_backward(weight_fp8, scale)
        ctx.input_shape = tuple(input.shape)
        ctx.input_dtype = input.dtype
        ctx.has_bias = bias is not None
        ctx.bias_dtype = bias.dtype if bias is not None else None

        return output.to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        weight_fp8, scale = ctx.saved_tensors
        grad_input = None
        grad_bias = None

        grad_2d = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        if ctx.needs_input_grad[0]:
            weight_compute = _dequant_fp8_qwen3(weight_fp8, scale, out_dtype=grad_2d.dtype)
            grad_input = grad_2d.matmul(weight_compute).view(ctx.input_shape).to(ctx.input_dtype)

        if ctx.has_bias and ctx.needs_input_grad[3]:
            reduce_dims = tuple(range(grad_output.dim() - 1))
            grad_bias = grad_output.sum(dim=reduce_dims) if reduce_dims else grad_output
            grad_bias = grad_bias.to(ctx.bias_dtype)

        return grad_input, None, None, grad_bias, None


def maybe_qwen3_native_block_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    module_name: str = "qwen3_fp8_linear",
) -> torch.Tensor | None:
    """Run native block-FP8 linear when enabled, otherwise signal fallback."""
    backend = get_qwen3_fp8_gemm_backend()
    if backend == "qdq":
        return None

    try:
        return _Qwen3NativeBlockFp8LinearFn.apply(
            input,
            weight,
            weight_scale_inv,
            bias,
            module_name,
        )
    except RuntimeError:
        if backend == "auto":
            return None
        raise


def qdq_fp8_linear_fallback(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the existing dequantize-then-linear FP8 fallback."""
    weight_compute = _dequant_fp8_qwen3(weight, weight_scale_inv, out_dtype=input.dtype)
    return F.linear(input.to(weight_compute.dtype), weight_compute, bias)
