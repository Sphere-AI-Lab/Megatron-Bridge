"""Single-adapter block-diagonal OFT rotation kernel for training.

Simplified from sglang's multi-adapter sgemm_oft_r by removing:
  - seg_indptr, weight_indices, oft_block_sizes (no multi-adapter dispatch)
  - num_slices (always 1 in megatron-bridge OFTRotationModule)
  - Identity passthrough branch (adapter always active during training)

Computes: output = x @ R  where R is block-diagonal.

Grid: (num_blocks * num_token_tiles,)
Each program handles one (BLOCK_S, BLOCK_SIZE) tile of one block.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sgemm_oft_r_single_fwd_kernel(
    x_ptr,
    output_ptr,
    R_ptr,
    total_tokens,
    input_dim,
    R_stride_0,  # stride for block dimension
    R_stride_1,  # stride for row within block
    R_stride_2,  # stride for col within block
    x_stride_0,
    output_stride_0,
    num_blocks: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute output[t, b*BS:(b+1)*BS] = x[t, b*BS:(b+1)*BS] @ R[b, :, :].

    Grid: (num_blocks * num_token_tiles,)
    """
    pid = tl.program_id(0)
    num_token_tiles = tl.cdiv(total_tokens, BLOCK_S)
    block_idx = pid // num_token_tiles
    token_tile_idx = pid % num_token_tiles

    if block_idx >= num_blocks:
        return

    token_offset = token_tile_idx * BLOCK_S
    if token_offset >= total_tokens:
        return

    s_offsets = tl.arange(0, BLOCK_S)
    actual_s = token_offset + s_offsets
    s_mask = actual_s < total_tokens

    k_offsets = tl.arange(0, BLOCK_SIZE)
    c_offsets = tl.arange(0, BLOCK_SIZE)

    x_base = x_ptr + block_idx * BLOCK_SIZE
    out_base = output_ptr + block_idx * BLOCK_SIZE
    R_base = R_ptr + block_idx * R_stride_0

    if BLOCK_SIZE >= 16:
        x_tile = tl.load(
            x_base + actual_s[:, None] * x_stride_0 + k_offsets[None, :],
            mask=s_mask[:, None],
            other=0.0,
        )
        R_block = tl.load(
            R_base + k_offsets[:, None] * R_stride_1 + c_offsets[None, :] * R_stride_2,
        )
        out = tl.dot(x_tile, R_block, input_precision="ieee")
    else:
        acc = tl.zeros((BLOCK_S, BLOCK_SIZE), dtype=tl.float32)
        for k in range(BLOCK_SIZE):
            x_col = tl.load(
                x_base + actual_s * x_stride_0 + k,
                mask=s_mask,
                other=0.0,
            ).to(tl.float32)
            R_row = tl.load(
                R_base + k * R_stride_1 + c_offsets * R_stride_2,
            ).to(tl.float32)
            acc += x_col[:, None] * R_row[None, :]
        out = acc

    tl.store(
        out_base + actual_s[:, None] * output_stride_0 + c_offsets[None, :],
        out.to(x_ptr.dtype.element_ty),
        mask=s_mask[:, None],
    )


@triton.jit
def _sgemm_oft_r_single_grad_R_kernel(
    x_ptr,
    grad_y_ptr,
    grad_R_ptr,
    total_tokens,
    input_dim,
    BLOCK_SIZE: tl.constexpr,
    TILE_T: tl.constexpr,
):
    """Compute grad_R[b, k, c] = sum_t x[t, b*BS+k] * grad_y[t, b*BS+c].

    Grid: (num_blocks,)
    """
    block_idx = tl.program_id(0)
    col_base = block_idx * BLOCK_SIZE

    k_offsets = tl.arange(0, BLOCK_SIZE)
    c_offsets = tl.arange(0, BLOCK_SIZE)

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for t_start in range(0, total_tokens, TILE_T):
        t_offsets = t_start + tl.arange(0, TILE_T)
        t_mask = t_offsets < total_tokens

        x_tile = tl.load(
            x_ptr + t_offsets[:, None] * input_dim + col_base + k_offsets[None, :],
            mask=t_mask[:, None],
            other=0.0,
        )
        gy_tile = tl.load(
            grad_y_ptr + t_offsets[:, None] * input_dim + col_base + c_offsets[None, :],
            mask=t_mask[:, None],
            other=0.0,
        )

        acc += tl.dot(tl.trans(x_tile), gy_tile, input_precision="ieee")

    out_base = grad_R_ptr + block_idx * BLOCK_SIZE * BLOCK_SIZE
    tl.store(
        out_base + k_offsets[:, None] * BLOCK_SIZE + c_offsets[None, :],
        acc,  # fp32 output to avoid overflow
    )


# ─────────────────────────────────────────────────────────────────────────────
# Python launchers
# ─────────────────────────────────────────────────────────────────────────────


def oft_r_single_fwd(
    x: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:
    """Single-adapter block-diagonal OFT rotation forward.

    Args:
        x: (total_tokens, input_dim)
        R: (num_blocks, block_size, block_size) precomputed rotation matrices

    Returns:
        (total_tokens, input_dim)
    """
    total_tokens, input_dim = x.shape
    num_blocks, block_size, _ = R.shape
    output = torch.empty_like(x)

    BLOCK_S = 16
    num_token_tiles = triton.cdiv(total_tokens, BLOCK_S)
    grid = (num_blocks * num_token_tiles,)

    _sgemm_oft_r_single_fwd_kernel[grid](
        x,
        output,
        R,
        total_tokens,
        input_dim,
        R.stride(0),
        R.stride(1),
        R.stride(2),
        x.stride(0),
        output.stride(0),
        num_blocks=num_blocks,
        BLOCK_S=BLOCK_S,
        BLOCK_SIZE=block_size,
    )
    return output


def oft_r_single_grad_R(
    x: torch.Tensor,
    grad_y: torch.Tensor,
    num_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Single-adapter grad_R computation.

    Args:
        x: (total_tokens, input_dim)
        grad_y: (total_tokens, input_dim)
        num_blocks: input_dim // block_size
        block_size: size of each block

    Returns:
        (num_blocks, block_size, block_size) in fp32
    """
    total_tokens, input_dim = x.shape
    grad_R = torch.empty(num_blocks, block_size, block_size, device=x.device, dtype=torch.float32)

    # TILE_T capped at 64: with BLOCK_SIZE=128 the kernel's two ieee-precision
    # tl.dot tiles (x_tile + gy_tile + fp32 staging from input_precision="ieee")
    # at TILE_T=128 require 256 KiB shared mem, exceeding the 232 KiB H100 cap.
    TILE_T = max(16, min(64, triton.next_power_of_2(total_tokens)))

    _sgemm_oft_r_single_grad_R_kernel[(num_blocks,)](
        x,
        grad_y,
        grad_R,
        total_tokens,
        input_dim,
        BLOCK_SIZE=block_size,
        TILE_T=TILE_T,
    )
    return grad_R


# ─────────────────────────────────────────────────────────────────────────────
# Autograd Function
# ─────────────────────────────────────────────────────────────────────────────


class OFTRotationSingleFunction(torch.autograd.Function):
    """Single-adapter OFT rotation with triton fwd + bwd.

    Usage:
        y = OFTRotationSingleFunction.apply(x, R)
    where:
        x: (total_tokens, input_dim)
        R: (num_blocks, block_size, block_size) — must have requires_grad if training
    """

    @staticmethod
    def forward(ctx, x, R):
        num_blocks, block_size, _ = R.shape
        ctx.save_for_backward(x, R)
        ctx.num_blocks = num_blocks
        ctx.block_size = block_size
        return oft_r_single_fwd(x, R)

    @staticmethod
    def backward(ctx, grad_output):
        x, R = ctx.saved_tensors
        num_blocks = ctx.num_blocks
        block_size = ctx.block_size

        grad_output = grad_output.contiguous()

        # grad_x = grad_y @ R^T
        R_T = R.transpose(-1, -2).contiguous()
        grad_x = oft_r_single_fwd(grad_output, R_T)

        # grad_R[b] = x[:, b, :].T @ grad_y[:, b, :]
        # Computed in fp32, cast back to R's dtype for autograd compatibility
        grad_R = oft_r_single_grad_R(x, grad_output, num_blocks, block_size)
        grad_R = grad_R.to(R.dtype)

        return grad_x, grad_R


def oft_r_single(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Functional API: triton OFT Rotation Single with autograd support."""
    return OFTRotationSingleFunction.apply(x, R)
