# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Triton-accelerated block-diagonal OFT rotation kernels with autograd support.

Provides fwd/bwd triton kernels and an autograd Function for OFT rotation:
    Forward:  y = x @ R          (block-diagonal)
    Backward: grad_x = grad_y @ R^T
              grad_R = x^T @ grad_y  (per-block reduction)

Ported from sglang.srt.oft.triton_ops.
"""

from megatron.bridge.orbit.oft.triton_oft.cayley_neumann import (
    CayleyNeumannFunction,
    cayley_neumann,
    cayley_neumann_bwd,
    cayley_neumann_fwd,
)
from megatron.bridge.orbit.oft.triton_oft.dequant_fp8 import dequant_fp8_block_triton
from megatron.bridge.orbit.oft.triton_oft.oft_rotation import OFTRotationFunction, oft_rotation
from megatron.bridge.orbit.oft.triton_oft.sgemm_oft_r import sgemm_oft_r_fwd
from megatron.bridge.orbit.oft.triton_oft.sgemm_oft_r_bwd import sgemm_oft_r_grad_R
from megatron.bridge.orbit.oft.triton_oft.sgemm_oft_r_by_expert import (
    OFTRotationByExpertFunction,
    oft_r_by_expert,
    oft_r_by_expert_fwd,
    oft_r_by_expert_grad_R,
)
from megatron.bridge.orbit.oft.triton_oft.sgemm_oft_r_single import (
    OFTRotationSingleFunction,
    oft_r_single,
    oft_r_single_fwd,
    oft_r_single_grad_R,
)
