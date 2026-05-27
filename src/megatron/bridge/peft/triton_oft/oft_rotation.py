"""Triton-accelerated block-diagonal OFT rotation with autograd support.

Provides OFTRotationFunction (torch.autograd.Function) that uses triton kernels
for both forward and backward passes:

    Forward:  y = x @ R          (sgemm_oft_r_fwd)
    Backward: grad_x = grad_y @ R^T  (sgemm_oft_r_fwd with transposed R)
              grad_R = x^T @ grad_y   (sgemm_oft_r_grad_R)
"""

import torch

from megatron.bridge.peft.triton_oft.sgemm_oft_r import sgemm_oft_r_fwd
from megatron.bridge.peft.triton_oft.sgemm_oft_r_bwd import sgemm_oft_r_grad_R


class OFTRotationFunction(torch.autograd.Function):
    """Autograd-compatible block-diagonal OFT rotation using triton kernels.

    All three passes (forward, grad_x, grad_R) use triton kernels.

    Usage:
        y = OFTRotationFunction.apply(x, R_blocks, weights, weights_T,
                                       batch_info, num_blocks, block_size)
    where:
        x:          (total_tokens, input_dim) input tensor
        R_blocks:   (num_blocks, block_size, block_size) rotation matrices
        weights:    (1, num_blocks, block_size, block_size) = R_blocks.unsqueeze(0)
        weights_T:  (1, num_blocks, block_size, block_size) = R_blocks^T.unsqueeze(0)
        batch_info: OFTBatchInfo with segment metadata
        num_blocks: int
        block_size: int
    """

    @staticmethod
    def forward(ctx, x, R_blocks, weights, weights_T, batch_info, num_blocks, block_size):
        ctx.save_for_backward(x, R_blocks, weights_T)
        ctx.batch_info = batch_info
        ctx.num_blocks = num_blocks
        ctx.block_size = block_size
        return sgemm_oft_r_fwd(x, weights, batch_info, num_slices=1)

    @staticmethod
    def backward(ctx, grad_output):
        x, R_blocks, weights_T = ctx.saved_tensors
        batch_info = ctx.batch_info
        num_blocks = ctx.num_blocks
        block_size = ctx.block_size

        # grad_x = grad_y @ R^T  (triton fwd kernel with transposed R)
        grad_x = sgemm_oft_r_fwd(grad_output, weights_T, batch_info, num_slices=1)

        # grad_R[b] = x[:, b, :].T @ grad_y[:, b, :]  (triton reduction kernel)
        grad_R = sgemm_oft_r_grad_R(x, grad_output, num_blocks, block_size)

        return grad_x, grad_R, None, None, None, None, None


def oft_rotation(x, R_blocks, weights, weights_T, batch_info, num_blocks, block_size):
    """Functional API for triton OFT rotation with autograd support.

    Args:
        x:          (total_tokens, input_dim) input tensor
        R_blocks:   (num_blocks, block_size, block_size) rotation matrices
        weights:    (1, num_blocks, block_size, block_size) = R_blocks.unsqueeze(0)
        weights_T:  (1, num_blocks, block_size, block_size) = R_blocks^T.unsqueeze(0)
        batch_info: OFTBatchInfo with segment metadata
        num_blocks: int
        block_size: int

    Returns:
        (total_tokens, input_dim) rotated output
    """
    return OFTRotationFunction.apply(
        x, R_blocks, weights, weights_T, batch_info, num_blocks, block_size
    )
