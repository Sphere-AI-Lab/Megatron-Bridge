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

"""Triton backward kernels for block-diagonal OFT rotation.

Forward:  y = x @ R  (block-diagonal, via sgemm_oft_r_fwd)
Backward:
    grad_x = grad_y @ R^T   (reuse sgemm_oft_r_fwd with transposed R)
    grad_R[b] = x[:, b, :].T @ grad_y[:, b, :]  (dedicated triton reduction kernel)
"""

import torch
import triton
import triton.language as tl

from megatron.bridge.orbit.oft.triton_oft.sgemm_oft_r import OFT_SMEM_BUDGET, _PIPELINE_STAGES


# Same H100 shared-memory concern as the forward kernel (sgemm_oft_r.py): the
# accumulator and the staged x/grad_y tiles all grow with BLOCK_SIZE, so an
# untiled kernel overflows shared memory once block_size is large enough.
# Tile the two BLOCK_SIZE axes of the (BLOCK_SIZE, BLOCK_SIZE) output block
# instead, mirroring the forward kernel's TILE_K/TILE_N split of its own
# BLOCK_SIZE axis.
BWD_UNTILED_MAX_BS = 128
BWD_TILE_K = 128
BWD_TILE_C = 128


def _pick_bwd_tiles(block_size: int, tile_t: int, itemsize: int = 2) -> tuple[int, int]:
    """Choose the largest (tile_k, tile_c) output tiles within the H100 smem budget."""

    if block_size <= BWD_UNTILED_MAX_BS:
        return block_size, block_size
    tile_k = min(BWD_TILE_K, block_size)
    tile_c = min(BWD_TILE_C, block_size)
    while tile_k > 16 and _PIPELINE_STAGES * itemsize * tile_t * (tile_k + tile_c) > OFT_SMEM_BUDGET:
        tile_k //= 2
        tile_c //= 2
    return tile_k, tile_c


@triton.jit
def _grad_R_kernel(
    x_ptr,
    grad_y_ptr,
    grad_R_ptr,
    total_tokens,
    input_dim,
    BLOCK_SIZE: tl.constexpr,
    TILE_K: tl.constexpr,
    TILE_C: tl.constexpr,
    TILE_T: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """Compute grad_R[b, k, c] = sum_t x[t, b*BS+k] * grad_y[t, b*BS+c].

    Grid: (num_blocks, cdiv(BLOCK_SIZE, TILE_K), cdiv(BLOCK_SIZE, TILE_C))
    Each program computes one (TILE_K, TILE_C) tile of the block's
    (BLOCK_SIZE, BLOCK_SIZE) output by tiling over the token dimension and
    accumulating in fp32, then storing in OUT_DTYPE. For BLOCK_SIZE <=
    BWD_UNTILED_MAX_BS, TILE_K == TILE_C == BLOCK_SIZE and this reduces to
    one program per block, same as the original untiled kernel.
    """
    block_idx = tl.program_id(0)
    k_tile_idx = tl.program_id(1)
    c_tile_idx = tl.program_id(2)
    col_base = block_idx * BLOCK_SIZE

    k_offsets = k_tile_idx * TILE_K + tl.arange(0, TILE_K)
    c_offsets = c_tile_idx * TILE_C + tl.arange(0, TILE_C)
    k_mask = k_offsets < BLOCK_SIZE
    c_mask = c_offsets < BLOCK_SIZE

    acc = tl.zeros((TILE_K, TILE_C), dtype=tl.float32)

    for t_start in range(0, total_tokens, TILE_T):
        t_offsets = t_start + tl.arange(0, TILE_T)
        t_mask = t_offsets < total_tokens

        # x_tile: (TILE_T, TILE_K)
        x_tile = tl.load(
            x_ptr + t_offsets[:, None] * input_dim + col_base + k_offsets[None, :],
            mask=t_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        # gy_tile: (TILE_T, TILE_C)
        gy_tile = tl.load(
            grad_y_ptr + t_offsets[:, None] * input_dim + col_base + c_offsets[None, :],
            mask=t_mask[:, None] & c_mask[None, :],
            other=0.0,
        )

        # x.T @ grad_y: (TILE_K, TILE_T) @ (TILE_T, TILE_C) -> (TILE_K, TILE_C)
        acc += tl.dot(tl.trans(x_tile), gy_tile, input_precision="ieee")

    # Store grad_R[block_idx, k_offsets, c_offsets] cast to output dtype
    out_base = grad_R_ptr + block_idx * BLOCK_SIZE * BLOCK_SIZE
    tl.store(
        out_base + k_offsets[:, None] * BLOCK_SIZE + c_offsets[None, :],
        acc.to(OUT_DTYPE),
        mask=k_mask[:, None] & c_mask[None, :],
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
    grad_R = torch.empty(num_blocks, block_size, block_size, device=x.device, dtype=out_dtype)

    # TILE_T must be >= 16 for tl.dot, and power of 2
    TILE_T = max(16, min(128, triton.next_power_of_2(total_tokens)))

    TILE_K, TILE_C = _pick_bwd_tiles(block_size, TILE_T)

    # Map torch dtype to triton constexpr dtype
    DTYPE_MAP = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }
    OUT_DTYPE = DTYPE_MAP[out_dtype]

    grid = (num_blocks, triton.cdiv(block_size, TILE_K), triton.cdiv(block_size, TILE_C))
    _grad_R_kernel[grid](
        x,
        grad_y,
        grad_R,
        total_tokens,
        input_dim,
        BLOCK_SIZE=block_size,
        TILE_K=TILE_K,
        TILE_C=TILE_C,
        TILE_T=TILE_T,
        OUT_DTYPE=OUT_DTYPE,
    )
    return grad_R
