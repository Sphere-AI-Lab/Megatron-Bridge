# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.bridge.orbit.oft.triton_oft.segmented_oft_linear import (
    segmented_oft_linear,
    segmented_oft_linear_reference,
)


def _dense_rotation(blocks: torch.Tensor) -> torch.Tensor:
    return torch.block_diag(*blocks.unbind(0))


@pytest.mark.unit
def test_reference_routes_repeated_rotation_ids() -> None:
    torch.manual_seed(7)
    x = torch.randn(3, 8)
    weight = torch.randn(7, 8)
    rotations = torch.randn(3, 2, 4, 4)
    offsets = torch.tensor([0, 2, 5, 7], dtype=torch.int32)
    rotation_ids = torch.tensor([0, 2, 0], dtype=torch.int32)

    expected = torch.cat(
        [
            (x @ _dense_rotation(rotations[0])) @ weight[0:2].T,
            (x @ _dense_rotation(rotations[2])) @ weight[2:5].T,
            (x @ _dense_rotation(rotations[0])) @ weight[5:7].T,
        ],
        dim=-1,
    )

    actual = segmented_oft_linear_reference(x, weight, rotations, offsets, rotation_ids)

    torch.testing.assert_close(actual, expected)


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("shape", "block_size"),
    [((1, 32), 16), ((2, 17, 64), 32)],
)
def test_segmented_oft_linear_matches_reference(
    shape: tuple[int, ...], block_size: int
) -> None:
    torch.manual_seed(11)
    device = torch.device("cuda")
    input_dim = shape[-1]
    output_dim = 23
    x = torch.randn(*shape, device=device, dtype=torch.bfloat16)
    weight = torch.randn(output_dim, input_dim, device=device, dtype=torch.bfloat16)
    rotations = torch.randn(
        3,
        input_dim // block_size,
        block_size,
        block_size,
        device=device,
        dtype=torch.bfloat16,
    )
    offsets = torch.tensor([0, 7, 19, 23], device=device, dtype=torch.int32)
    rotation_ids = torch.tensor([0, 2, 0], device=device, dtype=torch.int32)

    expected = segmented_oft_linear_reference(x, weight, rotations, offsets, rotation_ids)
    actual = segmented_oft_linear(x, weight, rotations, offsets, rotation_ids)

    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=1e-2)
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    )
    assert cosine > 0.9999


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_segmented_oft_linear_backward_accumulates_repeated_rotation_ids() -> None:
    torch.manual_seed(19)
    device = torch.device("cuda")
    x = torch.randn(5, 32, device=device, dtype=torch.bfloat16)
    weight = torch.randn(13, 32, device=device, dtype=torch.bfloat16)
    rotations = torch.randn(3, 2, 16, 16, device=device, dtype=torch.bfloat16)
    offsets = torch.tensor([0, 3, 9, 13], device=device, dtype=torch.int32)
    rotation_ids = torch.tensor([0, 2, 0], device=device, dtype=torch.int32)
    upstream = torch.randn(5, 13, device=device, dtype=torch.bfloat16)

    x_ref = x.detach().clone().requires_grad_(True)
    rotations_ref = rotations.detach().clone().requires_grad_(True)
    expected = segmented_oft_linear_reference(x_ref, weight, rotations_ref, offsets, rotation_ids)
    expected.backward(upstream)

    x_actual = x.detach().clone().requires_grad_(True)
    rotations_actual = rotations.detach().clone().requires_grad_(True)
    actual = segmented_oft_linear(x_actual, weight, rotations_actual, offsets, rotation_ids)
    actual.backward(upstream)

    for actual_grad, reference_grad in (
        (x_actual.grad, x_ref.grad),
        (rotations_actual.grad, rotations_ref.grad),
    ):
        torch.testing.assert_close(
            actual_grad.float(), reference_grad.float(), rtol=5e-2, atol=6.25e-2
        )
        cosine = torch.nn.functional.cosine_similarity(
            actual_grad.float().flatten(), reference_grad.float().flatten(), dim=0
        )
        assert cosine > 0.9999
    assert torch.count_nonzero(rotations_actual.grad[1]) == 0
