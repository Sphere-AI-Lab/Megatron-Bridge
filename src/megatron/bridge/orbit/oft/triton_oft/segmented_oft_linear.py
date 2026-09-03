# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Segment-aware OFT rotation followed by a fused output projection."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _segmented_oft_linear_fwd_kernel(
    x_ptr,
    weight_ptr,
    rotations_ptr,
    segment_offsets_ptr,
    rotation_ids_ptr,
    output_ptr,
    M,
    K,
    N,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fuse one routed block rotation and projection per output segment."""
    pid_m = tl.program_id(0)
    pid_segment = tl.program_id(1)
    pid_n = tl.program_id(2)

    segment_start = tl.load(segment_offsets_ptr + pid_segment)
    segment_end = tl.load(segment_offsets_ptr + pid_segment + 1)
    rotation_id = tl.load(rotation_ids_ptr + pid_segment)
    if segment_start + pid_n * BLOCK_N >= segment_end:
        return

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = segment_start + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offsets_m < M
    mask_n = offsets_n < segment_end
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offsets_block = tl.arange(0, BLOCK_SIZE)
    for block_idx in range(NUM_BLOCKS):
        offsets_k = block_idx * BLOCK_SIZE + offsets_block
        x = tl.load(
            x_ptr + offsets_m[:, None] * K + offsets_k[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        rotation_base = (rotation_id * NUM_BLOCKS + block_idx) * BLOCK_SIZE * BLOCK_SIZE
        rotation = tl.load(
            rotations_ptr
            + rotation_base
            + offsets_block[:, None] * BLOCK_SIZE
            + offsets_block[None, :]
        )
        rotated = tl.dot(x, rotation, input_precision="ieee", out_dtype=tl.float32).to(tl.bfloat16)
        weight = tl.load(
            weight_ptr + offsets_n[:, None] * K + offsets_k[None, :],
            mask=mask_n[:, None],
            other=0.0,
        )
        accumulator += tl.dot(rotated, tl.trans(weight), out_dtype=tl.float32)

    tl.store(
        output_ptr + offsets_m[:, None] * N + offsets_n[None, :],
        accumulator,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _validate_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    rotations: torch.Tensor,
    segment_offsets: torch.Tensor,
    rotation_ids: torch.Tensor,
) -> tuple[int, int, int]:
    """Validate the shared segmented OFT linear tensor contract."""
    if x.ndim < 2:
        raise ValueError(f"x must have at least two dimensions, got shape {tuple(x.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be two-dimensional, got shape {tuple(weight.shape)}")
    if rotations.ndim != 4 or rotations.shape[-1] != rotations.shape[-2]:
        raise ValueError(
            "rotations must have shape (num_adapters, num_blocks, block_size, block_size), "
            f"got {tuple(rotations.shape)}"
        )

    input_dim = x.shape[-1]
    output_dim, weight_input_dim = weight.shape
    block_size = rotations.shape[-1]
    if input_dim != weight_input_dim:
        raise ValueError(f"x input dim {input_dim} does not match weight input dim {weight_input_dim}")
    if rotations.shape[1] * block_size != input_dim:
        raise ValueError(
            f"rotation shape covers input dim {rotations.shape[1] * block_size}, expected {input_dim}"
        )
    if segment_offsets.ndim != 1 or rotation_ids.ndim != 1:
        raise ValueError("segment_offsets and rotation_ids must be one-dimensional")
    if segment_offsets.numel() != rotation_ids.numel() + 1:
        raise ValueError("segment_offsets must contain exactly one more element than rotation_ids")
    if segment_offsets.dtype not in (torch.int32, torch.int64):
        raise TypeError("segment_offsets must use an integer dtype")
    if rotation_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("rotation_ids must use an integer dtype")

    if segment_offsets.is_cuda and rotation_ids.is_cuda:
        return input_dim, output_dim, block_size

    offsets = segment_offsets.detach().cpu().tolist()
    ids = rotation_ids.detach().cpu().tolist()
    if not offsets or offsets[0] != 0 or offsets[-1] != output_dim:
        raise ValueError(f"segment_offsets must cover [0, {output_dim}], got {offsets}")
    if any(start >= end for start, end in zip(offsets, offsets[1:])):
        raise ValueError(f"segment_offsets must be strictly increasing, got {offsets}")
    if any(rotation_id < 0 or rotation_id >= rotations.shape[0] for rotation_id in ids):
        raise ValueError(f"rotation_ids must be in [0, {rotations.shape[0]}), got {ids}")
    return input_dim, output_dim, block_size


def segmented_oft_linear_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    rotations: torch.Tensor,
    segment_offsets: torch.Tensor,
    rotation_ids: torch.Tensor,
) -> torch.Tensor:
    """Apply independently routed OFT rotations with ordinary PyTorch ops."""
    input_dim, output_dim, block_size = _validate_inputs(
        x, weight, rotations, segment_offsets, rotation_ids
    )
    leading_shape = x.shape[:-1]
    x_2d = x.reshape(-1, input_dim)
    x_blocks = x_2d.to(rotations.dtype).reshape(x_2d.shape[0], -1, block_size)
    offsets = segment_offsets.detach().cpu().tolist()
    ids = rotation_ids.detach().cpu().tolist()

    outputs: list[torch.Tensor] = []
    for start, end, rotation_id in zip(offsets, offsets[1:], ids):
        rotated = torch.einsum("mbi,bij->mbj", x_blocks, rotations[rotation_id]).reshape_as(x_2d)
        rotated = rotated.to(x.dtype)
        outputs.append(F.linear(rotated.to(weight.dtype), weight[start:end]))

    return torch.cat(outputs, dim=-1).reshape(*leading_shape, output_dim)


def _segmented_oft_linear_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    rotations: torch.Tensor,
    segment_offsets: torch.Tensor,
    rotation_ids: torch.Tensor,
) -> torch.Tensor:
    """Launch the fused forward kernel for validated BF16 CUDA tensors."""
    input_dim = x.shape[-1]
    output_dim = weight.shape[0]
    leading_shape = x.shape[:-1]
    x_2d = x.reshape(-1, input_dim).contiguous()
    weight = weight.contiguous()
    rotations = rotations.contiguous()
    segment_offsets = segment_offsets.to(device=x.device, dtype=torch.int32).contiguous()
    rotation_ids = rotation_ids.to(device=x.device, dtype=torch.int32).contiguous()
    output = torch.empty((x_2d.shape[0], output_dim), device=x.device, dtype=x.dtype)

    block_size = rotations.shape[-1]
    num_blocks = rotations.shape[1]
    block_m = 16 if x_2d.shape[0] < 32 else 32
    block_n = 128
    grid = (
        triton.cdiv(x_2d.shape[0], block_m),
        rotation_ids.numel(),
        triton.cdiv(output_dim, block_n),
    )
    _segmented_oft_linear_fwd_kernel[grid](
        x_2d,
        weight,
        rotations,
        segment_offsets,
        rotation_ids,
        output,
        x_2d.shape[0],
        input_dim,
        output_dim,
        NUM_BLOCKS=num_blocks,
        BLOCK_SIZE=block_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8 if block_m == 32 else 4,
    )
    return output.reshape(*leading_shape, output_dim)


class _SegmentedOFTLinearFunction(torch.autograd.Function):
    """Autograd boundary for fused forward with explicit segmented backward."""

    @staticmethod
    def forward(ctx, x, weight, rotations, segment_offsets, rotation_ids):
        ctx.save_for_backward(x, weight, rotations, segment_offsets, rotation_ids)
        return _segmented_oft_linear_forward(x, weight, rotations, segment_offsets, rotation_ids)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, rotations, segment_offsets, rotation_ids = ctx.saved_tensors
        input_dim = x.shape[-1]
        block_size = rotations.shape[-1]
        x_2d = x.reshape(-1, input_dim)
        grad_output_2d = grad_output.reshape(-1, weight.shape[0]).contiguous()
        x_blocks = x_2d.to(rotations.dtype).reshape(x_2d.shape[0], -1, block_size)
        grad_x = torch.zeros_like(x_2d)
        grad_rotations = torch.zeros_like(rotations)
        offsets = segment_offsets.detach().cpu().tolist()
        ids = rotation_ids.detach().cpu().tolist()

        for start, end, rotation_id in zip(offsets, offsets[1:], ids):
            grad_rotated = torch.matmul(grad_output_2d[:, start:end], weight[start:end])
            grad_rotated_blocks = grad_rotated.to(rotations.dtype).reshape_as(x_blocks)
            grad_x_contribution = torch.einsum(
                "mbj,bij->mbi", grad_rotated_blocks, rotations[rotation_id]
            ).reshape_as(x_2d)
            grad_x.add_(grad_x_contribution.to(grad_x.dtype))
            grad_rotation = torch.einsum(
                "mbi,mbj->bij", x_blocks.float(), grad_rotated_blocks.float()
            ).to(rotations.dtype)
            grad_rotations[rotation_id].add_(grad_rotation)

        return grad_x.reshape_as(x), None, grad_rotations, None, None


def _can_use_fused_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    rotations: torch.Tensor,
    segment_offsets: torch.Tensor,
    rotation_ids: torch.Tensor,
) -> bool:
    return (
        x.is_cuda
        and weight.is_cuda
        and rotations.is_cuda
        and segment_offsets.is_cuda
        and rotation_ids.is_cuda
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and rotations.dtype == torch.bfloat16
        and rotations.shape[-1] in (16, 32)
    )


def segmented_oft_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    rotations: torch.Tensor,
    segment_offsets: torch.Tensor,
    rotation_ids: torch.Tensor,
) -> torch.Tensor:
    """Apply segment-specific OFT rotations and one packed linear projection."""
    _validate_inputs(x, weight, rotations, segment_offsets, rotation_ids)
    if not _can_use_fused_kernel(x, weight, rotations, segment_offsets, rotation_ids):
        return segmented_oft_linear_reference(x, weight, rotations, segment_offsets, rotation_ids)
    return _SegmentedOFTLinearFunction.apply(x, weight, rotations, segment_offsets, rotation_ids)
