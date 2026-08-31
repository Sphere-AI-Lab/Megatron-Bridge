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

import torch
import triton
import triton.language as tl


# Keep these launch-policy values aligned with SGLang's unsegmented and
# segmented OFT rotation kernels. Tiling large rotation blocks prevents one
# Triton program from exceeding H100's shared-memory limit.
OFT_SMEM_BUDGET = 232448
OFT_UNTILED_MAX_BS = 128
OFT_TILE_K = 128
OFT_TILE_N = 128
_PIPELINE_STAGES = 3


def _tiled_smem_bytes(block_s: int, tile_k: int, tile_n: int, itemsize: int = 2) -> int:
    """Estimate the staged operand bytes for one pipelined Triton program."""

    return _PIPELINE_STAGES * itemsize * (block_s * tile_k + tile_k * tile_n)


def _pick_tiles(block_size: int, block_s: int) -> tuple[int, int]:
    """Choose the largest contraction and output tiles within the H100 budget."""

    if block_size <= OFT_UNTILED_MAX_BS:
        return block_size, block_size
    tile_n = min(OFT_TILE_N, block_size)
    tile_k = min(OFT_TILE_K, block_size)
    while tile_k > 16 and _tiled_smem_bytes(block_s, tile_k, tile_n) > OFT_SMEM_BUDGET:
        tile_k //= 2
    return tile_k, tile_n


@triton.jit
def _sgemm_oft_r_kernel(
    x_ptr,
    output_ptr,
    weights_ptr,
    seg_indptr_ptr,
    weight_indices_ptr,
    oft_block_sizes_ptr,
    input_dim,
    max_seg_len,
    weights_stride_0,
    weights_stride_1,
    weights_stride_2,
    weights_stride_3,
    x_stride_0,
    output_stride_0,
    num_blocks: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_K: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """Unified Triton kernel for segmented block-diagonal OFT rotation.

    Handles all cases: standard (num_slices=1), QKV (num_slices=3),
    gate_up (num_slices=2) via the grid's axis 1.

    Grid: (num_blocks * cdiv(max_seg_len, BLOCK_S) * num_col_tiles,
           num_slices, num_segments)
      axis 0: block_idx * token/column tiles + token_tile_idx * num_col_tiles + col_tile_idx
      axis 1: slice_id (0 for standard; 0/1/2 for QKV; 0/1 for gate_up)
      axis 2: segment index

    Computes: for each slice s in [0, num_slices):
      output[:, s*input_dim:(s+1)*input_dim] = x @ R_s  (block-diagonal)
    where R_s is weights[adapter, s*num_blocks:(s+1)*num_blocks, :, :].

    When num_slices=1, the same x is rotated once (standard linear).
    When num_slices=3, the same x is rotated by 3 different R matrices (QKV).
    When num_slices=2, the same x is rotated by 2 different R matrices (gate_up).
    """
    pid_0 = tl.program_id(0)
    slice_id = tl.program_id(1)
    seg_idx = tl.program_id(2)

    num_col_tiles: tl.constexpr = tl.cdiv(BLOCK_SIZE, TILE_N)
    num_token_tiles = tl.cdiv(max_seg_len, BLOCK_S)
    tiles_per_block = num_token_tiles * num_col_tiles

    block_idx = pid_0 // tiles_per_block
    rem = pid_0 % tiles_per_block
    token_tile_idx = rem // num_col_tiles
    col_tile_idx = rem % num_col_tiles

    if block_idx >= num_blocks:
        return

    seg_start = tl.load(seg_indptr_ptr + seg_idx)
    seg_end = tl.load(seg_indptr_ptr + seg_idx + 1)
    seg_len = seg_end - seg_start

    token_offset = token_tile_idx * BLOCK_S
    if token_offset >= seg_len:
        return

    adapter_idx = tl.load(weight_indices_ptr + seg_idx)
    block_size_val = tl.load(oft_block_sizes_ptr + adapter_idx)

    s_offsets = tl.arange(0, BLOCK_S)
    actual_s = token_offset + s_offsets
    s_mask = actual_s < seg_len

    col_offset = col_tile_idx * TILE_N
    c_offsets = col_offset + tl.arange(0, TILE_N)
    c_mask = c_offsets < BLOCK_SIZE

    # Input: always read from the same x (all slices share the same input)
    x_base = x_ptr + seg_start * x_stride_0 + block_idx * BLOCK_SIZE
    # Output: offset by slice_id * input_dim to write into the correct slice
    out_base = output_ptr + seg_start * output_stride_0 + slice_id * input_dim + block_idx * BLOCK_SIZE

    if block_size_val == 0:
        # Base model passthrough: copy x to the corresponding output slice
        x_vals = tl.load(
            x_base + actual_s[:, None] * x_stride_0 + c_offsets[None, :],
            mask=s_mask[:, None] & c_mask[None, :],
            other=0.0,
        )
        tl.store(
            out_base + actual_s[:, None] * output_stride_0 + c_offsets[None, :],
            x_vals,
            mask=s_mask[:, None] & c_mask[None, :],
        )
        return

    # Weight block: weights[adapter_idx, slice_id * num_blocks + block_idx, :, :]
    weight_block_idx = slice_id * num_blocks + block_idx
    R_base = weights_ptr + adapter_idx * weights_stride_0 + weight_block_idx * weights_stride_1

    if BLOCK_SIZE >= 16:
        # Walk large blocks in fixed-width tiles so staged operands do not grow
        # with the full OFT block size. For blocks <= 128 both loops execute
        # once and retain the original kernel shape.
        acc = tl.zeros((BLOCK_S, TILE_N), dtype=tl.float32)
        for k0 in range(0, BLOCK_SIZE, TILE_K):
            k_offsets = k0 + tl.arange(0, TILE_K)
            k_mask = k_offsets < BLOCK_SIZE
            x_tile = tl.load(
                x_base + actual_s[:, None] * x_stride_0 + k_offsets[None, :],
                mask=s_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            R_tile = tl.load(
                R_base + k_offsets[:, None] * weights_stride_2 + c_offsets[None, :] * weights_stride_3,
                mask=k_mask[:, None] & c_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(x_tile.to(R_tile.dtype), R_tile, input_precision="ieee")
        out = acc
    else:
        # Element-wise loop for small block sizes (block_size < 16)
        acc = tl.zeros((BLOCK_S, TILE_N), dtype=tl.float32)
        for k in range(BLOCK_SIZE):
            x_col = tl.load(
                x_base + actual_s * x_stride_0 + k,
                mask=s_mask,
                other=0.0,
            ).to(tl.float32)
            R_row = tl.load(
                R_base + k * weights_stride_2 + c_offsets * weights_stride_3,
                mask=c_mask,
                other=0.0,
            ).to(tl.float32)
            acc += x_col[:, None] * R_row[None, :]
        out = acc

    tl.store(
        out_base + actual_s[:, None] * output_stride_0 + c_offsets[None, :],
        out.to(x_ptr.dtype.element_ty),
        mask=s_mask[:, None] & c_mask[None, :],
    )


def sgemm_oft_r_fwd(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info,
    num_slices: int = 1,
) -> torch.Tensor:
    """Unified launcher for segmented block-diagonal OFT rotation.

    Handles standard (num_slices=1), QKV (num_slices=3), and
    gate_up (num_slices=2) in a single kernel.

    Args:
        x: (total_tokens, input_dim) input tensor
        weights: (num_ofts, num_slices * num_blocks, block_size, block_size) precomputed R
        batch_info: OFTBatchInfo containing:
            - seg_indptr (num_segments+1,): CSR-style boundaries into the token dimension.
                seg_indptr[i]:seg_indptr[i+1] gives the token range for segment i.
                Example: [0, 5, 8, 15] means seg0=tokens 0-4, seg1=tokens 5-7, seg2=tokens 8-14.
            - seg_lens (num_segments,): length of each segment. Example: [5, 3, 7].
            - weight_indices (num_segments,): which adapter slot each segment uses.
                Example: [1, 0, 0] — seg0 uses adapter 1, seg1 uses adapter 0, etc.
                The actual "no adapter" check is done via oft_block_sizes, not this array.
            - oft_block_sizes (num_ofts,): block size of each adapter slot.
                0 means no adapter (triggers identity/copy passthrough).
                Example: [64, 64] — adapter 0 has block_size=64, adapter 1 has block_size=64.
            - max_len (int): max(seg_lens), used to size the grid's token tile dimension.
            - num_segments (int): number of segments (= batch size for triton backend).
            - bs (int): batch size.
            - permutation: (chunked backend only, not used in this kernel).
        num_slices: 1 for standard, 2 for gate_up, 3 for QKV

    Returns:
        (total_tokens, num_slices * input_dim) rotated output

    Example (QKV, num_slices=3):
        3 sequences in the batch:
          Seq 0: 5 tokens, uses adapter 1
          Seq 1: 3 tokens, no adapter
          Seq 2: 7 tokens, uses adapter 0

        input_dim = 128, block_size = 64, num_slices = 3

        Derived values:
          seg_indptr      = [0, 5, 8, 15]   (5+3+7 = 15 total tokens)
          weight_indices  = [1, X, 0]        (adapter slot per segment)
          oft_block_sizes = [64, 64]         (block_size per adapter; 0 = no adapter)
          num_blocks      = 128 / 64 = 2
          max_len         = 7
          BLOCK_S         = 16
          num_token_tiles = cdiv(7, 16) = 1

        Grid = (num_blocks * num_token_tiles, num_slices, num_segments)
             = (2 * 1, 3, 3) = 18 programs

        Each program is identified by (pid_0, slice_id, seg_idx):
          pid_0    → unpacked into block_idx and token_tile_idx
          slice_id → 0=Q, 1=K, 2=V
          seg_idx  → which sequence/segment

        All 18 programs:
          pid_0 | slice_id | seg_idx | block_idx | What it computes
          ------|----------|---------|-----------|------------------------------------------
            0   |  0 (Q)   |  0      |     0     | Seq0, Q slice, cols 0-63:   x @ R_Q_blk0
            1   |  0 (Q)   |  0      |     1     | Seq0, Q slice, cols 64-127: x @ R_Q_blk1
            0   |  1 (K)   |  0      |     0     | Seq0, K slice, cols 0-63:   x @ R_K_blk0
            1   |  1 (K)   |  0      |     1     | Seq0, K slice, cols 64-127: x @ R_K_blk1
            0   |  2 (V)   |  0      |     0     | Seq0, V slice, cols 0-63:   x @ R_V_blk0
            1   |  2 (V)   |  0      |     1     | Seq0, V slice, cols 64-127: x @ R_V_blk1
            0   |  0 (Q)   |  1      |     0     | Seq1, Q slice, cols 0-63:   COPY x (no adapter)
            1   |  0 (Q)   |  1      |     1     | Seq1, Q slice, cols 64-127: COPY x (no adapter)
            0   |  1 (K)   |  1      |     0     | Seq1, K slice, cols 0-63:   COPY x (no adapter)
            1   |  1 (K)   |  1      |     1     | Seq1, K slice, cols 64-127: COPY x (no adapter)
            0   |  2 (V)   |  1      |     0     | Seq1, V slice, cols 0-63:   COPY x (no adapter)
            1   |  2 (V)   |  1      |     1     | Seq1, V slice, cols 64-127: COPY x (no adapter)
            0   |  0 (Q)   |  2      |     0     | Seq2, Q slice, cols 0-63:   x @ R_Q_blk0
            1   |  0 (Q)   |  2      |     1     | Seq2, Q slice, cols 64-127: x @ R_Q_blk1
            0   |  1 (K)   |  2      |     0     | Seq2, K slice, cols 0-63:   x @ R_K_blk0
            1   |  1 (K)   |  2      |     1     | Seq2, K slice, cols 64-127: x @ R_K_blk1
            0   |  2 (V)   |  2      |     0     | Seq2, V slice, cols 0-63:   x @ R_V_blk0
            1   |  2 (V)   |  2      |     1     | Seq2, V slice, cols 64-127: x @ R_V_blk1

        For program (pid_0=0, slice_id=1(K), seg_idx=0(Seq0, adapter 1)):

          Step 1 — Unpack pid_0:
            block_idx      = pid_0 // num_token_tiles  = 0 // 1 = 0
            token_tile_idx = pid_0 %  num_token_tiles  = 0 %  1 = 0

          Step 2 — Segment boundaries (from seg_indptr):
            seg_start    = seg_indptr[seg_idx]     = seg_indptr[0] = 0
            seg_end      = seg_indptr[seg_idx + 1] = seg_indptr[1] = 5
            seg_len      = seg_end - seg_start     = 5
            token_offset = token_tile_idx * BLOCK_S = 0 * 16 = 0

          Step 3 — Adapter lookup:
            adapter_idx    = weight_indices[seg_idx]      = weight_indices[0] = 1
            block_size_val = oft_block_sizes[adapter_idx] = oft_block_sizes[1] = 64 (nonzero → has adapter)

          Step 4 — Input address (same x for all slices):
            x_base = x_ptr + seg_start * x_stride_0 + block_idx * BLOCK_SIZE
                   = x[0:, 0:]    →  x[0:5, 0:64]

          Step 5 — Output address (offset by slice_id into the correct region):
            out_base = output_ptr + seg_start * output_stride_0 + slice_id * input_dim + block_idx * BLOCK_SIZE
                     = output[0:, 1*128 + 0:]    →  output[0:5, 128:192]

          Step 6 — Weight block address:
            weight_block_idx = slice_id * num_blocks + block_idx = 1 * 2 + 0 = 2
            R_base = weights[adapter_idx, weight_block_idx, :, :] = weights[1, 2, :, :]

          Step 7 — Compute and store:
            out = x[0:5, 0:64] @ weights[1, 2, :, :]    →  (5, 64) @ (64, 64) = (5, 64)
            store → output[0:5, 128:192]

        Output memory layout per token:
          output[tok, :] = [ Q rotation (0-127) | K rotation (128-255) | V rotation (256-383) ]
                             slice_id=0           slice_id=1             slice_id=2

        Weight tensor layout per adapter:
          weights[adapter, :, :, :] =
            [ block0_Q, block1_Q, block0_K, block1_K, block0_V, block1_V ]
              idx 0     idx 1     idx 2     idx 3     idx 4     idx 5
            indexed as: slice_id * num_blocks + block_idx

        No-adapter case (Seq 1, block_size_val=0):
          Kernel copies x directly to output → identity (passthrough)
    """
    total_tokens, input_dim = x.shape

    if weights.numel() == 0:
        return x.repeat(1, num_slices)

    num_ofts, total_blocks_buf, block_size, _ = weights.shape

    if block_size == 0 or total_blocks_buf == 0:
        return x.repeat(1, num_slices)

    if input_dim % block_size != 0:
        raise ValueError(f"OFT input_dim ({input_dim}) must be divisible by block_size ({block_size})")

    num_blocks = input_dim // block_size
    output = torch.empty((total_tokens, num_slices * input_dim), device=x.device, dtype=x.dtype)

    BLOCK_S = 16
    BLOCK_SIZE = block_size

    TILE_K, TILE_N = _pick_tiles(BLOCK_SIZE, BLOCK_S)
    num_token_tiles = triton.cdiv(batch_info.max_len, BLOCK_S)
    num_col_tiles = triton.cdiv(BLOCK_SIZE, TILE_N)
    grid = (num_blocks * num_token_tiles * num_col_tiles, num_slices, batch_info.num_segments)

    _sgemm_oft_r_kernel[grid](
        x,
        output,
        weights,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.oft_block_sizes,
        input_dim,
        batch_info.max_len,
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        weights.stride(3),
        x.stride(0),
        output.stride(0),
        num_blocks=num_blocks,
        BLOCK_S=BLOCK_S,
        BLOCK_SIZE=BLOCK_SIZE,
        TILE_K=TILE_K,
        TILE_N=TILE_N,
    )

    return output
