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

from types import SimpleNamespace

import pytest
import torch
from megatron.core import parallel_state

from megatron.bridge.orbit.oft.oft_layers import OFTRotationModule, TEOFTLayerNormLinear
from megatron.bridge.orbit.oft.triton_oft.cayley_neumann import _torch_cayley_neumann, cayley_neumann
from megatron.bridge.orbit.oft.triton_oft.dequant_fp8 import dequant_fp8_block_triton
from megatron.bridge.orbit.oft.triton_oft.int4_dequant import dequantize_int4_triton
from megatron.bridge.orbit.oft.triton_oft.oft_rotation import oft_rotation
from megatron.bridge.orbit.oft.triton_oft.sgemm_oft_r import sgemm_oft_r_fwd
from megatron.bridge.orbit.oft.triton_oft.sgemm_oft_r_by_expert import oft_r_by_expert
from megatron.bridge.orbit.oft.triton_oft.sgemm_oft_r_single import oft_r_single


pytestmark = [pytest.mark.unit, pytest.mark.run_only_on("gpu")]

# B200 Triton SGEMM exhibits this amount of forward and backward rounding on
# the deterministic inputs below. The bounds remain tight enough to catch
# indexing or layout regressions.
_SGEMM_ATOL = 6e-3
_SGEMM_RTOL = 1e-3


@pytest.fixture(autouse=True)
def mock_tensor_parallel_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_group", lambda: object())


def _rotation_reference(x: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    tokens = x.shape[0]
    blocks, block_size, _ = rotation.shape
    return torch.einsum("tbk,bkc->tbc", x.reshape(tokens, blocks, block_size), rotation).reshape_as(x)


def _batch_info(segment_lengths: list[int], weight_indices: list[int], block_sizes: list[int]):
    device = torch.device("cuda")
    lengths = torch.tensor(segment_lengths, device=device, dtype=torch.int64)
    boundaries = torch.cat((torch.zeros(1, device=device, dtype=torch.int64), lengths.cumsum(0)))
    return SimpleNamespace(
        seg_indptr=boundaries,
        weight_indices=torch.tensor(weight_indices, device=device, dtype=torch.int64),
        oft_block_sizes=torch.tensor(block_sizes, device=device, dtype=torch.int64),
        max_len=max(segment_lengths),
        num_segments=len(segment_lengths),
    )


@pytest.mark.parametrize("num_slices", [1, 2, 3])
def test_segmented_multi_adapter_rotation_matches_torch(num_slices: int) -> None:
    torch.manual_seed(5 + num_slices)
    segment_lengths = [2, 3, 1]
    adapter_indices = [0, 1, 2]
    batch_info = _batch_info(segment_lengths, adapter_indices, [16, 16, 0])
    x = torch.randn(sum(segment_lengths), 32, device="cuda", dtype=torch.float32)
    weights = torch.randn(3, num_slices * 2, 16, 16, device="cuda", dtype=torch.float32)

    result = sgemm_oft_r_fwd(x, weights, batch_info, num_slices=num_slices)
    reference_segments = []
    offset = 0
    for length, adapter_idx in zip(segment_lengths, adapter_indices):
        segment = x[offset : offset + length]
        if batch_info.oft_block_sizes[adapter_idx] == 0:
            reference_segments.append(segment.repeat(1, num_slices))
        else:
            slices = []
            for slice_idx in range(num_slices):
                start = slice_idx * 2
                slices.append(_rotation_reference(segment, weights[adapter_idx, start : start + 2]))
            reference_segments.append(torch.cat(slices, dim=-1))
        offset += length

    expected = torch.cat(reference_segments)
    torch.testing.assert_close(result, expected, atol=_SGEMM_ATOL, rtol=_SGEMM_RTOL)


def test_generic_rotation_forward_and_backward_match_torch() -> None:
    torch.manual_seed(9)
    batch_info = _batch_info([7], [0], [16])
    x_triton = torch.randn(7, 32, device="cuda", dtype=torch.float32, requires_grad=True)
    r_triton = torch.randn(2, 16, 16, device="cuda", dtype=torch.float32, requires_grad=True)
    x_torch = x_triton.detach().clone().requires_grad_(True)
    r_torch = r_triton.detach().clone().requires_grad_(True)
    grad = torch.randn_like(x_triton)

    weights = r_triton.detach().unsqueeze(0)
    weights_t = r_triton.detach().transpose(-1, -2).contiguous().unsqueeze(0)
    out_triton = oft_rotation(x_triton, r_triton, weights, weights_t, batch_info, 2, 16)
    out_torch = _rotation_reference(x_torch, r_torch)
    grads_triton = torch.autograd.grad(out_triton, (x_triton, r_triton), grad)
    grads_torch = torch.autograd.grad(out_torch, (x_torch, r_torch), grad)

    torch.testing.assert_close(out_triton, out_torch, atol=_SGEMM_ATOL, rtol=_SGEMM_RTOL)
    torch.testing.assert_close(grads_triton[0], grads_torch[0], atol=_SGEMM_ATOL, rtol=_SGEMM_RTOL)
    torch.testing.assert_close(grads_triton[1], grads_torch[1], atol=_SGEMM_ATOL, rtol=_SGEMM_RTOL)


@pytest.mark.parametrize("block_size", [192, 256])
def test_segmented_rotation_large_block_matches_torch(block_size: int) -> None:
    torch.manual_seed(10)
    batch_info = _batch_info([2], [0], [block_size])
    x = torch.randn(2, block_size, device="cuda", dtype=torch.float16) * 0.01
    weights = torch.randn(1, 1, block_size, block_size, device="cuda", dtype=torch.float16) * 0.01

    result = sgemm_oft_r_fwd(x, weights, batch_info)
    expected = x @ weights[0, 0]

    torch.testing.assert_close(result, expected, atol=2e-3, rtol=2e-3)


def test_cayley_triton_forward_and_backward_match_torch() -> None:
    torch.manual_seed(1)
    raw = torch.randn(2, 16, 16, device="cuda", dtype=torch.float32) * 0.01
    q_triton = (raw - raw.transpose(-1, -2)).detach().requires_grad_(True)
    q_torch = q_triton.detach().clone().requires_grad_(True)
    grad = torch.randn_like(q_triton)

    out_triton = cayley_neumann(q_triton)
    out_torch = _torch_cayley_neumann(q_torch)
    grad_triton = torch.autograd.grad(out_triton, q_triton, grad)[0]
    grad_torch = torch.autograd.grad(out_torch, q_torch, grad)[0]

    torch.testing.assert_close(out_triton, out_torch, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(grad_triton, grad_torch, atol=3e-4, rtol=3e-4)


def test_single_rotation_triton_forward_and_backward_match_torch() -> None:
    torch.manual_seed(2)
    x_triton = torch.randn(7, 32, device="cuda", dtype=torch.float32, requires_grad=True)
    r_triton = torch.randn(2, 16, 16, device="cuda", dtype=torch.float32, requires_grad=True)
    x_torch = x_triton.detach().clone().requires_grad_(True)
    r_torch = r_triton.detach().clone().requires_grad_(True)
    grad = torch.randn_like(x_triton)

    out_triton = oft_r_single(x_triton, r_triton)
    out_torch = _rotation_reference(x_torch, r_torch)
    grads_triton = torch.autograd.grad(out_triton, (x_triton, r_triton), grad)
    grads_torch = torch.autograd.grad(out_torch, (x_torch, r_torch), grad)

    torch.testing.assert_close(out_triton, out_torch, atol=_SGEMM_ATOL, rtol=_SGEMM_RTOL)
    torch.testing.assert_close(grads_triton[0], grads_torch[0], atol=_SGEMM_ATOL, rtol=_SGEMM_RTOL)
    torch.testing.assert_close(grads_triton[1], grads_torch[1], atol=_SGEMM_ATOL, rtol=_SGEMM_RTOL)


def test_segmented_expert_rotation_forward_and_backward_match_torch() -> None:
    torch.manual_seed(3)
    tokens_per_expert = torch.tensor([3, 0, 4], device="cuda", dtype=torch.int64)
    x_triton = torch.randn(7, 32, device="cuda", dtype=torch.float32, requires_grad=True)
    r_triton = torch.randn(3, 2, 16, 16, device="cuda", dtype=torch.float32, requires_grad=True)
    x_torch = x_triton.detach().clone().requires_grad_(True)
    r_torch = r_triton.detach().clone().requires_grad_(True)
    grad = torch.randn_like(x_triton)

    reference_chunks = []
    offset = 0
    for expert_idx, count in enumerate(tokens_per_expert.tolist()):
        if count:
            reference_chunks.append(_rotation_reference(x_torch[offset : offset + count], r_torch[expert_idx]))
        offset += count
    out_torch = torch.cat(reference_chunks)
    out_triton = oft_r_by_expert(x_triton, r_triton, tokens_per_expert)
    grads_triton = torch.autograd.grad(out_triton, (x_triton, r_triton), grad)
    grads_torch = torch.autograd.grad(out_torch, (x_torch, r_torch), grad)

    torch.testing.assert_close(out_triton, out_torch, atol=_SGEMM_ATOL, rtol=_SGEMM_RTOL)
    torch.testing.assert_close(grads_triton[0], grads_torch[0], atol=_SGEMM_ATOL, rtol=_SGEMM_RTOL)
    torch.testing.assert_close(grads_triton[1], grads_torch[1], atol=_SGEMM_ATOL, rtol=_SGEMM_RTOL)


def test_int4_dequant_triton_matches_packed_reference() -> None:
    q = torch.arange(4 * 64, device="cuda", dtype=torch.int32).reshape(4, 64) % 16 - 8
    unsigned = (q + 8).reshape(4, 8, 8)
    packed = torch.zeros(4, 8, device="cuda", dtype=torch.int32)
    for lane in range(8):
        packed |= (unsigned[:, :, lane] & 0xF) << (lane * 4)
    scale = torch.tensor(
        [[0.25, 0.5], [0.75, 1.0], [1.25, 1.5], [1.75, 2.0]],
        device="cuda",
        dtype=torch.float32,
    )
    shape = torch.tensor([4, 64], device="cuda", dtype=torch.int32)

    result = dequantize_int4_triton(packed, scale, shape, group_size=32, out_dtype=torch.float16)
    expected = (q.float() * scale.repeat_interleave(32, dim=1)).to(torch.float16)

    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def test_fp8_block_dequant_triton_matches_torch() -> None:
    values = torch.linspace(-2, 2, 32 * 64, device="cuda", dtype=torch.float32).reshape(32, 64)
    weight = values.to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.5, 1.0], [1.5, 2.0]], device="cuda", dtype=torch.float32)

    result = dequant_fp8_block_triton(weight, scale, torch.bfloat16)
    expanded_scale = scale.repeat_interleave(16, dim=0).repeat_interleave(32, dim=1)
    expected = (weight.float() * expanded_scale).to(torch.bfloat16)

    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def test_te_layernorm_linear_oft_identity_matches_transformer_engine() -> None:
    te = pytest.importorskip("transformer_engine.pytorch")
    torch.manual_seed(4)
    layer = te.LayerNormLinear(
        in_features=16,
        out_features=8,
        bias=True,
        normalization="LayerNorm",
        device="cuda",
    )
    adapter = OFTRotationModule(
        in_features=16,
        block_size=4,
        input_is_parallel=True,
        dtype=layer.weight.dtype,
        device=torch.device("cuda"),
    )
    wrapped = TEOFTLayerNormLinear(layer, adapter)
    x_reference = torch.randn(5, 16, device="cuda", dtype=layer.weight.dtype, requires_grad=True)
    x_wrapped = x_reference.detach().clone().requires_grad_(True)

    expected = layer(x_reference)
    expected = expected[0] if isinstance(expected, tuple) else expected
    result, bias = wrapped(x_wrapped)
    grad = torch.randn_like(result)
    grad_reference = torch.autograd.grad(expected, x_reference, grad)[0]
    grad_wrapped = torch.autograd.grad(result, x_wrapped, grad)[0]

    assert bias is None
    torch.testing.assert_close(result, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(grad_wrapped, grad_reference, atol=6e-5, rtol=5e-4)
