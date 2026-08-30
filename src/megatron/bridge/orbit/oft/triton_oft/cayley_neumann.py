"""Triton-accelerated Cayley-Neumann transform for OFT rotation matrices.

R = I + 2Q + 2Q² + 2Q³ + Q⁴  from skew-symmetric Q.

Single-program-per-block approach: each triton program handles one
(block_size x block_size) block entirely in registers, fusing all
matrix multiplications and accumulations.

For small blocks (fp16: <= 128, fp32: <= 32), uses triton kernels (3-4x fwd, 14-25x bwd).
For larger blocks, falls back to torch (register pressure makes triton slower).

Hardcoded for NUM_TERMS=5 (the standard OFT configuration).
"""

import torch
import triton
import triton.language as tl


NUM_TERMS = 5  # R = I + 2Q + 2Q² + 2Q³ + Q⁴
# fp32 needs 2x registers per element, so lower threshold
_TRITON_MAX_BLOCK_SIZE_FP32 = 32
_TRITON_MAX_BLOCK_SIZE_FP16 = 128

# ─────────────────────────────────────────────────────────────────────────────
# Naive
# ─────────────────────────────────────────────────────────────────────────────


def _torch_cayley_neumann(
    Q_skew: torch.Tensor,
    num_terms: int = 5,
) -> torch.Tensor:
    """Compute Cayley transform via Neumann series approximation.

    Args:
        Q_skew: (num_blocks, block_size, block_size) skew-symmetric matrices
        num_terms: number of terms in Neumann series

    Returns:
        (num_blocks, block_size, block_size) orthogonal matrices
    """
    b, block_size, _ = Q_skew.shape
    R = torch.eye(block_size, device=Q_skew.device, dtype=Q_skew.dtype).repeat(b, 1, 1)
    if num_terms > 1:
        R.add_(Q_skew, alpha=2.0)
        if num_terms > 2:
            Q_squared = torch.bmm(Q_skew, Q_skew)
            R.add_(Q_squared, alpha=2.0)
            Q_power = Q_squared
            for _ in range(3, num_terms - 1):
                Q_power = torch.bmm(Q_power, Q_skew)
                R.add_(Q_power, alpha=2.0)
            Q_power = torch.bmm(Q_power, Q_skew)
            R.add_(Q_power)
    return R


# ─────────────────────────────────────────────────────────────────────────────
# Forward kernel
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _cayley_fwd_kernel(
    Q_ptr,
    R_ptr,
    stride_b,
    stride_r,
    stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute R = I + 2Q + 2Q² + 2Q³ + Q⁴ for one block. Grid: (num_blocks,)"""
    bid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_SIZE)
    cols = tl.arange(0, BLOCK_SIZE)

    Q_base = Q_ptr + bid * stride_b
    Q = tl.load(Q_base + rows[:, None] * stride_r + cols[None, :] * stride_c)

    # Accumulate R
    eye = (rows[:, None] == cols[None, :]).to(Q.dtype)
    R = eye + 2.0 * Q

    # Q² = Q @ Q  (cast dot output back to input dtype for next dot)
    Q2 = tl.dot(Q, Q, input_precision="ieee").to(Q.dtype)
    R = R + 2.0 * Q2

    # Q³ = Q² @ Q
    Q3 = tl.dot(Q2, Q, input_precision="ieee").to(Q.dtype)
    R = R + 2.0 * Q3

    # Q⁴ = Q³ @ Q
    Q4 = tl.dot(Q3, Q, input_precision="ieee").to(Q.dtype)
    R = R + Q4

    R_base = R_ptr + bid * stride_b
    tl.store(
        R_base + rows[:, None] * stride_r + cols[None, :] * stride_c,
        R.to(Q.dtype),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Backward kernel
# ─────────────────────────────────────────────────────────────────────────────
#
# R = I + 2P₁ + 2P₂ + 2P₃ + P₄   where Pₖ = Pₖ₋₁ @ Q, P₀ = I
#
# Adjoint recurrence (backward through the polynomial):
#   G₄ = grad_R                          (coeff of P₄ is 1)
#   Gₖ = cₖ·grad_R + Gₖ₊₁ @ Qᵀ         for k = 3,2,1  (cₖ = 2)
#
# Gradient via Horner's method:
#   grad_Q = G₁ + Qᵀ(G₂ + Qᵀ(G₃ + Qᵀ·G₄))
#
# Single backward sweep, O(n) matmuls:
#   acc = G₄;  G_prev = G₄
#   for k = n-1 down to 1:
#     Gₖ = 2·grad_R + G_prev @ Qᵀ
#     acc = Gₖ + Qᵀ @ acc
#     G_prev = Gₖ
#   grad_Q = acc


@triton.jit
def _cayley_bwd_kernel(
    grad_R_ptr,
    Q_ptr,
    grad_Q_ptr,
    stride_b,
    stride_r,
    stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute grad_Q for one block. Grid: (num_blocks,)"""
    bid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_SIZE)
    cols = tl.arange(0, BLOCK_SIZE)

    Q_base = Q_ptr + bid * stride_b
    Q = tl.load(Q_base + rows[:, None] * stride_r + cols[None, :] * stride_c)
    gR_base = grad_R_ptr + bid * stride_b
    gR = tl.load(gR_base + rows[:, None] * stride_r + cols[None, :] * stride_c)

    Q_T = tl.trans(Q)
    # G_prev = G₄ = grad_R;  acc = G₄ (Horner innermost)
    G_prev = gR
    acc = gR

    # Sweep k = 3, 2, 1 (3 iterations for NUM_TERMS=5)
    for _ in range(3):
        # Gₖ = 2·grad_R + G_prev @ Qᵀ  (cast to input dtype for next dot)
        G_k = (2.0 * gR + tl.dot(G_prev, Q_T, input_precision="ieee")).to(gR.dtype)
        G_prev = G_k

        # Horner: acc = Gₖ + Qᵀ @ acc
        acc = (G_k + tl.dot(Q_T, acc, input_precision="ieee")).to(gR.dtype)

    # Store grad_Q
    gQ_base = grad_Q_ptr + bid * stride_b
    tl.store(
        gQ_base + rows[:, None] * stride_r + cols[None, :] * stride_c,
        acc.to(gR.dtype),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Python launchers
# ─────────────────────────────────────────────────────────────────────────────


def cayley_neumann_fwd(Q_skew: torch.Tensor, num_terms: int = 5) -> torch.Tensor:
    """Triton Cayley-Neumann forward (block_size must be <= _TRITON_MAX_BLOCK_SIZE)."""
    assert num_terms == NUM_TERMS, f"Only num_terms={NUM_TERMS} supported, got {num_terms}"
    num_blocks, block_size, _ = Q_skew.shape
    R = torch.empty_like(Q_skew)
    _cayley_fwd_kernel[(num_blocks,)](
        Q_skew,
        R,
        Q_skew.stride(0),
        Q_skew.stride(1),
        Q_skew.stride(2),
        BLOCK_SIZE=block_size,
    )
    return R


def cayley_neumann_bwd(
    grad_R: torch.Tensor,
    Q_skew: torch.Tensor,
    num_terms: int = 5,
) -> torch.Tensor:
    """Triton Cayley-Neumann backward (block_size must be <= _TRITON_MAX_BLOCK_SIZE)."""
    assert num_terms == NUM_TERMS, f"Only num_terms={NUM_TERMS} supported, got {num_terms}"
    num_blocks, block_size, _ = Q_skew.shape
    grad_Q = torch.empty_like(Q_skew)
    _cayley_bwd_kernel[(num_blocks,)](
        grad_R,
        Q_skew,
        grad_Q,
        Q_skew.stride(0),
        Q_skew.stride(1),
        Q_skew.stride(2),
        BLOCK_SIZE=block_size,
        num_warps=4 if block_size <= 32 else 8,
        num_stages=1,
    )
    return grad_Q


# ─────────────────────────────────────────────────────────────────────────────
# Autograd Function
# ─────────────────────────────────────────────────────────────────────────────


class CayleyNeumannFunction(torch.autograd.Function):
    """Autograd wrapper for triton Cayley-Neumann transform."""

    @staticmethod
    def forward(ctx, Q_skew, num_terms):
        ctx.save_for_backward(Q_skew)
        ctx.num_terms = num_terms
        return cayley_neumann_fwd(Q_skew, num_terms)

    @staticmethod
    def backward(ctx, grad_R):
        (Q_skew,) = ctx.saved_tensors
        grad_Q = cayley_neumann_bwd(grad_R.contiguous(), Q_skew, ctx.num_terms)
        return grad_Q, None


def cayley_neumann(Q_skew: torch.Tensor, num_terms: int = 5) -> torch.Tensor:
    """Functional API: triton Cayley-Neumann with autograd support.

    Uses triton for small blocks (fp16: <= 64, fp32: <= 32).
    Falls back to torch for larger blocks (register pressure makes triton slower).
    """
    block_size = Q_skew.shape[-1]
    max_bs = _TRITON_MAX_BLOCK_SIZE_FP32 if Q_skew.dtype == torch.float32 else _TRITON_MAX_BLOCK_SIZE_FP16
    if block_size > max_bs:
        return _torch_cayley_neumann(Q_skew, num_terms)
    return CayleyNeumannFunction.apply(Q_skew, num_terms)
