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

"""Fused FP8 block-wise dequant triton kernel.

Mirror of ``sglang/srt/oft/triton_ops/parity_dequant_fp8.py`` — keep the
two files in sync so training (Bridge) and inference (sglang parity
mode) run the same dequant math.

Semantics
---------
    out[..., m, n] = fp32(w_fp8[..., m, n]) * scale[..., m//BH, n//BW]
    cast to out_dtype (bf16 / fp16)

The PyTorch reference (``quant/fp8_utils.py::dequant_fp8``) materializes a
full fp32 copy of the weight before multiplying. For a 200B-parameter
expert tensor that's 800GB of transient fp32 and three full memory
passes. This kernel reads FP8 → fp32 in-register, multiplies by one
scale per tile, stores the cast result. One memory pass, no fp32
intermediate.

Consumed by ``dequant_fp8`` in ``quant/fp8_utils.py`` on CUDA inputs;
falls back to the PyTorch path otherwise.
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
    def _dequant_fp8_block_kernel(
        w_ptr,
        scale_ptr,
        out_ptr,
        E,
        M,
        N,
        SR,
        SC,
        w_stride_e,
        w_stride_m,
        w_stride_n,
        s_stride_e,
        s_stride_m,
        s_stride_n,
        o_stride_e,
        o_stride_m,
        o_stride_n,
        BH: tl.constexpr,
        BW: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Grid: (E, cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))."""
        pid_e = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)

        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = m_offs < M
        n_mask = n_offs < N
        tile_mask = m_mask[:, None] & n_mask[None, :]

        w_fp8 = tl.load(
            w_ptr + pid_e * w_stride_e + m_offs[:, None] * w_stride_m + n_offs[None, :] * w_stride_n,
            mask=tile_mask,
            other=0.0,
        )
        w_f32 = w_fp8.to(tl.float32)

        sm = m_offs // BH
        sn = n_offs // BW
        s_mask_m = sm < SR
        s_mask_n = sn < SC
        scale_mask = s_mask_m[:, None] & s_mask_n[None, :]
        scale = tl.load(
            scale_ptr + pid_e * s_stride_e + sm[:, None] * s_stride_m + sn[None, :] * s_stride_n,
            mask=scale_mask,
            other=1.0,
        )

        out = (w_f32 * scale).to(out_ptr.dtype.element_ty)

        tl.store(
            out_ptr + pid_e * o_stride_e + m_offs[:, None] * o_stride_m + n_offs[None, :] * o_stride_n,
            out,
            mask=tile_mask,
        )


def dequant_fp8_block_triton(
    w_fp8: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype,
    *,
    block_size: int | None = None,
) -> torch.Tensor:
    """Fused FP8 block-wise dequant. CUDA-only; caller must gate on device.

    Accepts 2-D ``[M, N]`` or 3-D ``[E, M, N]`` inputs; returns same rank.
    By default, block dimensions are inferred from an evenly divisible scale
    grid. Passing ``block_size`` supports fixed-size blocks with partial tails.
    """
    assert _HAS_TRITON, "triton not available"
    assert w_fp8.is_cuda, "dequant_fp8_block_triton requires CUDA"

    squeeze_out = False
    if w_fp8.dim() == 2:
        w_fp8 = w_fp8.unsqueeze(0)
        scale = scale.unsqueeze(0)
        squeeze_out = True
    assert w_fp8.dim() == 3 and scale.dim() == 3, (
        f"expected [E,M,N] and [E,sr,sc], got {w_fp8.shape} and {scale.shape}"
    )
    E, M, N = w_fp8.shape
    E_s, SR, SC = scale.shape
    assert E_s == E, (E, E_s)
    if block_size is None:
        assert M % SR == 0 and N % SC == 0, (M, SR, N, SC)
        BH = M // SR
        BW = N // SC
    else:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        expected_scale_shape = (
            triton.cdiv(M, block_size),
            triton.cdiv(N, block_size),
        )
        if (SR, SC) != expected_scale_shape:
            raise ValueError(
                f"Expected scale grid {expected_scale_shape} for weight shape {(M, N)} "
                f"with block_size={block_size}, got {(SR, SC)}"
            )
        BH = block_size
        BW = block_size

    out = torch.empty((E, M, N), dtype=out_dtype, device=w_fp8.device)

    BLOCK_M = min(128, max(16, triton.next_power_of_2(BH)))
    BLOCK_N = min(128, max(16, triton.next_power_of_2(BW)))

    grid = (E, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _dequant_fp8_block_kernel[grid](
        w_fp8,
        scale,
        out,
        E,
        M,
        N,
        SR,
        SC,
        w_fp8.stride(0),
        w_fp8.stride(1),
        w_fp8.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BH=BH,
        BW=BW,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return out.squeeze(0) if squeeze_out else out
