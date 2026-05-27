"""Fused INT4 dense dequant Triton kernel.

This keeps the Megatron INT4 packed-weight contract unchanged:
``weight_packed`` stores 8 signed INT4 lanes per int32 and ``weight_scale``
stores one scale per row/group.  The kernel materializes the same BF16 dense
weight as the PyTorch reference, but without building the intermediate
``[out, packed_in, 8]`` unpacked tensor.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    triton = None  # type: ignore
    tl = None  # type: ignore
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _int4_dequant_kernel(
        w_packed_ptr,
        w_scale_ptr,
        out_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_wn: tl.constexpr,
        stride_wkpack: tl.constexpr,
        stride_sn: tl.constexpr,
        stride_sg: tl.constexpr,
        stride_on: tl.constexpr,
        stride_ok: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        out_type: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

        packed = tl.load(
            w_packed_ptr
            + offs_n[:, None] * stride_wn
            + (offs_k[None, :] // 8) * stride_wkpack,
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0,
        )
        scale = tl.load(
            w_scale_ptr
            + offs_n[:, None] * stride_sn
            + (offs_k[None, :] // GROUP_SIZE) * stride_sg,
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        shift = (offs_k[None, :] % 8) * 4
        weight = (((packed >> shift) & 0xF).to(tl.float32) - 8.0) * scale
        out = weight.to(out_type)

        out_ptrs = out_ptr + offs_n[:, None] * stride_on + offs_k[None, :] * stride_ok
        tl.store(out_ptrs, out, mask=(offs_n[:, None] < N) & (offs_k[None, :] < K))


def dequantize_int4_triton(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_shape: torch.Tensor,
    *,
    group_size: int = 32,
    out_dtype: torch.dtype = torch.bfloat16,
    block_n: int = 16,
    block_k: int = 128,
) -> torch.Tensor:
    """Materialize a dense INT4 weight with Triton.

    ``weight_shape`` is accepted for API symmetry with ``dequantize_int4``.
    The current Megatron reference derives the local shape from
    ``weight_packed``; this function intentionally mirrors that behavior.
    """

    del weight_shape
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not available")
    if not weight_packed.is_cuda:
        raise ValueError("dequantize_int4_triton requires CUDA weight_packed")
    if weight_packed.dim() != 2:
        raise ValueError(f"expected 2-D weight_packed, got {tuple(weight_packed.shape)}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if block_k % 8 != 0:
        raise ValueError(f"block_k must be divisible by 8, got {block_k}")
    if out_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"out_dtype must be bfloat16 or float16, got {out_dtype}")

    out_features, packed_in = weight_packed.shape
    in_features = packed_in * 8
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features={in_features} must be divisible by group_size={group_size}"
        )

    if weight_scale.dim() == 1:
        scale = weight_scale.view(out_features, -1)
    else:
        scale = weight_scale.view(out_features, -1)
    expected_groups = in_features // group_size
    if scale.shape[1] != expected_groups:
        raise ValueError(
            f"expected {expected_groups} scale groups for shape "
            f"({out_features}, {in_features}), got {scale.shape[1]}"
        )
    if not scale.is_cuda:
        scale = scale.to(weight_packed.device)

    out = torch.empty((out_features, in_features), device=weight_packed.device, dtype=out_dtype)
    out_type = tl.bfloat16 if out_dtype is torch.bfloat16 else tl.float16
    grid = (triton.cdiv(out_features, block_n), triton.cdiv(in_features, block_k))
    _int4_dequant_kernel[grid](
        weight_packed,
        scale,
        out,
        out_features,
        in_features,
        weight_packed.stride(0),
        weight_packed.stride(1),
        scale.stride(0),
        scale.stride(1),
        out.stride(0),
        out.stride(1),
        GROUP_SIZE=group_size,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        out_type=out_type,
        num_warps=4,
        num_stages=3,
    )
    return out
