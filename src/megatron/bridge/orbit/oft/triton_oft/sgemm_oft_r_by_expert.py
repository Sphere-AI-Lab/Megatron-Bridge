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

"""Segmented OFT rotation for grouped expert inputs.

Input rows are contiguous by expert. ``tokens_per_expert`` supplies segment
lengths. The kernel applies:

    out[offset_e + t, block] = x[offset_e + t, block] @ R[e, block]

without materializing per-token rotation matrices. Grad-x reuses the fwd
kernel with swapped R i/j strides (so R is read transposed in-place, no
``.contiguous()`` alloc). Grad-R accumulates ``x.T @ grad_y`` per (expert,
block) tile.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _oft_r_by_expert_fwd_kernel(
    x_ptr,
    out_ptr,
    R_ptr,
    offsets_ptr,
    counts_ptr,
    input_dim: tl.constexpr,
    x_stride_0,
    out_stride_0,
    R_stride_e,
    R_stride_b,
    R_stride_i,
    R_stride_j,
    BLOCK_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_K: tl.constexpr,
):
    expert = tl.program_id(0)
    token_tile = tl.program_id(1)
    block_idx = tl.program_id(2)

    token_count = tl.load(counts_ptr + expert).to(tl.int64)
    segment_start = tl.load(offsets_ptr + expert).to(tl.int64)
    local_offsets = token_tile * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    row_ids = segment_start + local_offsets
    row_mask = local_offsets < token_count

    k_base = block_idx * BLOCK_SIZE
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_SIZE), dtype=tl.float32)

    for k_off in range(0, BLOCK_SIZE, TILE_K):
        k_offsets = (k_base + k_off + tl.arange(0, TILE_K)).to(tl.int64)
        r_rows = (k_off + tl.arange(0, TILE_K)).to(tl.int64)

        x_ptrs = x_ptr + row_ids[:, None] * x_stride_0 + k_offsets[None, :]
        x_tile = tl.load(x_ptrs, mask=row_mask[:, None], other=0.0)

        r_ptrs = (
            R_ptr
            + expert * R_stride_e
            + block_idx * R_stride_b
            + r_rows[:, None] * R_stride_i
            + col_offsets[None, :] * R_stride_j
        )
        r_tile = tl.load(r_ptrs)
        acc += tl.dot(x_tile, r_tile, input_precision="ieee")

    out_ptrs = out_ptr + row_ids[:, None] * out_stride_0 + (k_base + col_offsets)[None, :]
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=row_mask[:, None])


@triton.jit
def _oft_r_by_expert_grad_R_kernel(
    x_ptr,
    grad_y_ptr,
    grad_R_ptr,
    offsets_ptr,
    counts_ptr,
    input_dim: tl.constexpr,
    x_stride_0,
    gy_stride_0,
    grad_R_stride_e,
    grad_R_stride_b,
    grad_R_stride_i,
    grad_R_stride_j,
    BLOCK_SIZE: tl.constexpr,
    TILE_T: tl.constexpr,
):
    expert = tl.program_id(0)
    block_idx = tl.program_id(1)

    token_count = tl.load(counts_ptr + expert).to(tl.int64)
    segment_start = tl.load(offsets_ptr + expert).to(tl.int64)
    k_base = block_idx * BLOCK_SIZE

    k_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    c_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for t_start in range(0, token_count, TILE_T):
        t_offsets = t_start + tl.arange(0, TILE_T).to(tl.int64)
        row_ids = segment_start + t_offsets
        row_mask = t_offsets < token_count

        x_ptrs = x_ptr + row_ids[:, None] * x_stride_0 + (k_base + k_offsets)[None, :]
        gy_ptrs = grad_y_ptr + row_ids[:, None] * gy_stride_0 + (k_base + c_offsets)[None, :]
        x_tile = tl.load(x_ptrs, mask=row_mask[:, None], other=0.0)
        gy_tile = tl.load(gy_ptrs, mask=row_mask[:, None], other=0.0)
        acc += tl.dot(tl.trans(x_tile), gy_tile, input_precision="ieee")

    out_ptrs = (
        grad_R_ptr
        + expert * grad_R_stride_e
        + block_idx * grad_R_stride_b
        + k_offsets[:, None] * grad_R_stride_i
        + c_offsets[None, :] * grad_R_stride_j
    )
    tl.store(out_ptrs, acc)


def _offsets_counts_max(
    tokens_per_expert: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    counts = tokens_per_expert.to(dtype=torch.int64)
    if counts.numel() == 0:
        offsets = counts.new_empty(0)
        return offsets, counts, 0
    cumsum = torch.cumsum(counts, dim=0)
    offsets = torch.empty(counts.numel(), device=counts.device, dtype=torch.int64)
    offsets[0] = 0
    offsets[1:] = cumsum[:-1]
    max_count = int(counts.max().item())
    return offsets, counts, max_count


def _fwd_kernel_launch(
    x: torch.Tensor,
    R: torch.Tensor,
    offsets: torch.Tensor,
    counts: torch.Tensor,
    max_count: int,
    R_stride_i: int,
    R_stride_j: int,
) -> torch.Tensor:
    total_tokens, input_dim = x.shape
    num_experts, num_blocks, block_size, _ = R.shape
    assert input_dim == num_blocks * block_size
    output = torch.empty_like(x)
    if total_tokens == 0 or max_count == 0:
        return output
    block_m = 16
    tile_k = min(64, block_size)
    grid = (num_experts, triton.cdiv(max_count, block_m), num_blocks)
    _oft_r_by_expert_fwd_kernel[grid](
        x,
        output,
        R,
        offsets,
        counts,
        input_dim,
        x.stride(0),
        output.stride(0),
        R.stride(0),
        R.stride(1),
        R_stride_i,
        R_stride_j,
        BLOCK_M=block_m,
        BLOCK_SIZE=block_size,
        TILE_K=tile_k,
    )
    return output


def oft_r_by_expert_fwd(
    x: torch.Tensor,
    R: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Apply per-expert OFT rotations to segmented expert inputs."""

    offsets, counts, max_count = _offsets_counts_max(tokens_per_expert)
    return _fwd_kernel_launch(x, R, offsets, counts, max_count, R.stride(2), R.stride(3))


def oft_r_by_expert_grad_R(
    x: torch.Tensor,
    grad_y: torch.Tensor,
    R_shape: tuple[int, int, int, int],
    tokens_per_expert: torch.Tensor,
    offsets: torch.Tensor | None = None,
    counts: torch.Tensor | None = None,
    max_count: int | None = None,
) -> torch.Tensor:
    """Compute the per-expert rotation gradient for segmented OFT inputs."""

    num_experts, num_blocks, block_size, _ = R_shape
    if offsets is None or counts is None or max_count is None:
        offsets, counts, max_count = _offsets_counts_max(tokens_per_expert)
    grad_R = torch.empty(R_shape, device=x.device, dtype=torch.float32)
    if x.shape[0] == 0 or counts.numel() == 0 or max_count == 0:
        grad_R.zero_()
        return grad_R

    tile_t = max(16, min(64, triton.next_power_of_2(max_count)))
    _oft_r_by_expert_grad_R_kernel[(num_experts, num_blocks)](
        x,
        grad_y,
        grad_R,
        offsets,
        counts,
        x.shape[1],
        x.stride(0),
        grad_y.stride(0),
        grad_R.stride(0),
        grad_R.stride(1),
        grad_R.stride(2),
        grad_R.stride(3),
        BLOCK_SIZE=block_size,
        TILE_T=tile_t,
    )
    return grad_R


class OFTRotationByExpertFunction(torch.autograd.Function):
    """Autograd wrapper for segmented per-expert OFT rotation."""

    @staticmethod
    def forward(ctx, x, R, tokens_per_expert):
        tokens_per_expert = tokens_per_expert.to(device=x.device, dtype=torch.int64)
        offsets, counts, max_count = _offsets_counts_max(tokens_per_expert)
        ctx.save_for_backward(x, R, offsets, counts)
        ctx.max_count = max_count
        return _fwd_kernel_launch(x, R, offsets, counts, max_count, R.stride(2), R.stride(3))

    @staticmethod
    def backward(ctx, grad_output):
        x, R, offsets, counts = ctx.saved_tensors
        max_count = ctx.max_count
        grad_output = grad_output.contiguous()
        # grad_x = grad_y @ R^T — reuse fwd kernel with swapped R i/j strides so R is
        # read transposed in-place; saves a (E, blocks, bs, bs) alloc + copy per backward.
        grad_x = _fwd_kernel_launch(
            grad_output,
            R,
            offsets,
            counts,
            max_count,
            R.stride(3),
            R.stride(2),
        )
        grad_R = oft_r_by_expert_grad_R(
            x,
            grad_output,
            tuple(R.shape),
            tokens_per_expert=None,
            offsets=offsets,
            counts=counts,
            max_count=max_count,
        ).to(R.dtype)
        return grad_x, grad_R, None


def oft_r_by_expert(
    x: torch.Tensor,
    R: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Apply per-expert OFT rotations with a custom Triton backward."""

    return OFTRotationByExpertFunction.apply(x, R, tokens_per_expert)
