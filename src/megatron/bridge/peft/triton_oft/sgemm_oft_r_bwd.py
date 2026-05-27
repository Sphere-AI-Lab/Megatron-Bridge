"""Triton backward kernels for block-diagonal OFT rotation.

Forward:  y = x @ R  (block-diagonal, via sgemm_oft_r_fwd)
Backward:
    grad_x = grad_y @ R^T   (reuse sgemm_oft_r_fwd with transposed R)
    grad_R[b] = x[:, b, :].T @ grad_y[:, b, :]  (dedicated triton reduction kernel)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _grad_R_kernel(
    x_ptr,
    grad_y_ptr,
    grad_R_ptr,
    total_tokens,
    input_dim,
    BLOCK_SIZE: tl.constexpr,
    TILE_T: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """Compute grad_R[b, k, c] = sum_t x[t, b*BS+k] * grad_y[t, b*BS+c].

    Grid: (num_blocks,)
    Each program computes one (BLOCK_SIZE, BLOCK_SIZE) output block
    by tiling over the token dimension and accumulating in fp32,
    then storing in OUT_DTYPE.
    """
    block_idx = tl.program_id(0)
    col_base = block_idx * BLOCK_SIZE

    k_offsets = tl.arange(0, BLOCK_SIZE)
    c_offsets = tl.arange(0, BLOCK_SIZE)

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for t_start in range(0, total_tokens, TILE_T):
        t_offsets = t_start + tl.arange(0, TILE_T)
        t_mask = t_offsets < total_tokens

        # x_tile: (TILE_T, BLOCK_SIZE)
        x_tile = tl.load(
            x_ptr + t_offsets[:, None] * input_dim + col_base + k_offsets[None, :],
            mask=t_mask[:, None],
            other=0.0,
        )
        # gy_tile: (TILE_T, BLOCK_SIZE)
        gy_tile = tl.load(
            grad_y_ptr + t_offsets[:, None] * input_dim + col_base + c_offsets[None, :],
            mask=t_mask[:, None],
            other=0.0,
        )

        # x.T @ grad_y: (BLOCK_SIZE, TILE_T) @ (TILE_T, BLOCK_SIZE) -> (BLOCK_SIZE, BLOCK_SIZE)
        acc += tl.dot(tl.trans(x_tile), gy_tile, input_precision="ieee")

    # Store grad_R[block_idx, :, :] cast to output dtype
    out_base = grad_R_ptr + block_idx * BLOCK_SIZE * BLOCK_SIZE
    tl.store(
        out_base + k_offsets[:, None] * BLOCK_SIZE + c_offsets[None, :],
        acc.to(OUT_DTYPE),
    )


def sgemm_oft_r_grad_R(
    x: torch.Tensor,
    grad_y: torch.Tensor,
    num_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Compute grad_R for block-diagonal OFT rotation via triton kernel.

    grad_R[b] = x_blocked[:, b, :].T @ grad_y_blocked[:, b, :]

    Accumulates in fp32 internally, stores in the input dtype.

    Args:
        x: (total_tokens, input_dim) — saved input from forward
        grad_y: (total_tokens, input_dim) — upstream gradient
        num_blocks: number of orthogonal blocks (input_dim // block_size)
        block_size: size of each orthogonal block

    Returns:
        (num_blocks, block_size, block_size) — gradient w.r.t. R blocks
    """
    total_tokens, input_dim = x.shape
    out_dtype = x.dtype
    grad_R = torch.empty(
        num_blocks, block_size, block_size, device=x.device, dtype=out_dtype
    )

    # TILE_T must be >= 16 for tl.dot, and power of 2
    TILE_T = max(16, min(128, triton.next_power_of_2(total_tokens)))

    # Map torch dtype to triton constexpr dtype
    DTYPE_MAP = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }
    OUT_DTYPE = DTYPE_MAP[out_dtype]

    _grad_R_kernel[(num_blocks,)](
        x, grad_y, grad_R,
        total_tokens, input_dim,
        BLOCK_SIZE=block_size,
        TILE_T=TILE_T,
        OUT_DTYPE=OUT_DTYPE,
    )
    return grad_R
