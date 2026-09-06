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
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

from megatron.bridge.orbit.oft import canonical_oft, oft_layers
from megatron.bridge.orbit.oft.canonical_oft import OFTLinearSplitFC1UpGate, OFTLinearSplitQKV
from megatron.bridge.orbit.oft.oft_layers import OFTLinear, OFTRotationModule


pytestmark = pytest.mark.unit


def _reference_oft_linear_bank(x, weight, rotations, output_sizes):
    block_size = rotations.shape[-1]
    num_blocks = rotations.shape[1]
    x_blocks = x.reshape(*x.shape[:-1], num_blocks, block_size)
    outputs = []
    for weight_slice, rotation in zip(weight.split(output_sizes, dim=0), rotations):
        rotated = torch.einsum("...rk,rkc->...rc", x_blocks, rotation).reshape_as(x)
        outputs.append(F.linear(rotated, weight_slice))
    return torch.cat(outputs, dim=-1)


def test_materialize_oft_weight_matches_explicit_block_diagonal_rotation():
    torch.manual_seed(7)
    weight = torch.randn(5, 8, dtype=torch.float64)
    rotations = torch.randn(2, 4, 4, dtype=torch.float64)

    actual = oft_layers._materialize_oft_weight(weight, rotations)
    expected = weight @ torch.block_diag(*rotations).T

    torch.testing.assert_close(actual, expected)


def test_materialized_oft_linear_bank_matches_uneven_slice_forward_and_gradients():
    torch.manual_seed(11)
    output_sizes = (7, 3, 3)
    x_ref = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    rotations_ref = torch.randn(3, 2, 4, 4, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(sum(output_sizes), 8, dtype=torch.float64)
    output_grad = torch.randn(2, 5, sum(output_sizes), dtype=torch.float64)

    expected = _reference_oft_linear_bank(x_ref, weight, rotations_ref, output_sizes)
    expected.backward(output_grad)
    expected_x_grad = x_ref.grad.detach().clone()
    expected_rotation_grad = rotations_ref.grad.detach().clone()

    x_actual = x_ref.detach().clone().requires_grad_()
    rotations_actual = rotations_ref.detach().clone().requires_grad_()
    actual = oft_layers._materialized_oft_linear_bank(
        x_actual,
        weight,
        rotations_actual,
        output_sizes,
    )
    actual.backward(output_grad)

    torch.testing.assert_close(actual, expected.detach())
    torch.testing.assert_close(x_actual.grad, expected_x_grad)
    torch.testing.assert_close(rotations_actual.grad, expected_rotation_grad)


def test_materialized_oft_linear_bank_supports_a_single_projection():
    torch.manual_seed(13)
    output_sizes = (6,)
    x = torch.randn(9, 8, dtype=torch.float64, requires_grad=True)
    rotations = torch.randn(1, 2, 4, 4, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(6, 8, dtype=torch.float64)

    actual = oft_layers._materialized_oft_linear_bank(x, weight, rotations, output_sizes)
    expected = _reference_oft_linear_bank(x, weight, rotations, output_sizes)

    torch.testing.assert_close(actual, expected)


class _StubRotation(nn.Module):
    def __init__(self):
        super().__init__()
        self.rotation = nn.Parameter(torch.eye(4, dtype=torch.float64).repeat(2, 1, 1))
        self.coft = False
        self.block_share = False
        self.dropout = nn.Dropout(0.0)
        self.is_expert = False
        self.raise_in_forward = False

    def _compute_rotation(self):
        return self.rotation

    def forward(self, x):
        if self.raise_in_forward:
            raise AssertionError("activation-rotation fallback was used")
        blocks = x.reshape(*x.shape[:-1], 2, 4)
        return torch.einsum("...rk,rkc->...rc", blocks, self.rotation).reshape_as(x)


def test_oft_linear_uses_opt_in_materialized_path_without_base_weight_gradient(monkeypatch):
    torch.manual_seed(17)
    monkeypatch.setenv("MEGATRON_OFT_MATERIALIZE_DENSE", "1")
    base = nn.Linear(8, 6, bias=False, dtype=torch.float64)
    base.weight.requires_grad_(False)
    adapter = _StubRotation()
    adapter.raise_in_forward = True
    wrapper = OFTLinear(base, adapter)
    x = torch.randn(5, 8, dtype=torch.float64, requires_grad=True)

    actual = wrapper(x)
    expected = F.linear(x, oft_layers._materialize_oft_weight(base.weight, adapter.rotation))
    assert isinstance(actual, torch.Tensor)
    actual.sum().backward()

    assert base.weight.grad is None
    assert adapter.rotation.grad is not None
    torch.testing.assert_close(actual, expected)


def _fail_activation_path(*args, **kwargs):
    raise AssertionError("activation-rotation fallback was used")


def test_dense_materialization_defaults_off(monkeypatch):
    monkeypatch.delenv("MEGATRON_OFT_MATERIALIZE_DENSE", raising=False)
    base = nn.Linear(8, 6, bias=False, dtype=torch.float64).requires_grad_(False)
    adapter = _StubRotation()
    wrapper = OFTLinear(base, adapter)
    x = torch.randn(5, 8, dtype=torch.float64)
    monkeypatch.setattr(oft_layers, "_materialized_oft_linear_bank", _fail_activation_path, raising=False)

    actual = wrapper(x)

    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual, base(adapter(x)))


@pytest.fixture
def cpu_tp(monkeypatch, tmp_path):
    """Use a real one-rank Gloo group; all tensor math remains on CPU."""
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_group", lambda **kwargs: dist.group.WORLD)
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("enabled", [True, False])
def test_dense_plain_linear_preserves_tensor_contract_through_sequential(cpu_tp, monkeypatch, enabled):
    torch.manual_seed(19)
    monkeypatch.setenv("MEGATRON_OFT_MATERIALIZE_DENSE", "1")
    base = nn.Linear(8, 6, bias=True, dtype=torch.float64).requires_grad_(False)
    adapter = OFTRotationModule(in_features=8, block_size=4, dtype=torch.float64)
    with torch.no_grad():
        adapter.oft_r.normal_(std=0.02)
    wrapper = OFTLinear(base, adapter)
    if not enabled:
        wrapper.disable_adapter_layers()
    x = torch.randn(5, 8, dtype=torch.float64)
    expected = F.relu(base(adapter(x) if enabled else x))

    actual = nn.Sequential(wrapper, nn.ReLU())(x)

    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual, expected)


def _make_core_linear(kind, output_size, dtype, skip_bias_add, tp_group):
    config = ModelParallelConfig(
        use_cpu_initialization=True,
        params_dtype=dtype,
        gradient_accumulation_fusion=False,
    )
    linear_cls = RowParallelLinear if kind == "row" else ColumnParallelLinear
    kwargs = {"input_is_parallel": True} if kind == "row" else {}
    base = linear_cls(
        8,
        output_size,
        config=config,
        init_method=lambda tensor: nn.init.normal_(tensor, std=0.1),
        bias=True,
        skip_bias_add=skip_bias_add,
        tp_group=tp_group,
        **kwargs,
    )
    base.requires_grad_(False)
    with torch.no_grad():
        base.bias.normal_(std=0.1)
    return base


def _make_wrapper(kind, dtype, skip_bias_add, tp_group, qkv_output_size=24):
    output_size = qkv_output_size if kind == "qkv" else (12 if kind == "gate_up" else 6)
    base = _make_core_linear(kind, output_size, dtype, skip_bias_add, tp_group)
    if kind == "qkv":
        # Preserve Qwen3 GQA's four query heads per KV group, at tiny dimensions.
        provider = SimpleNamespace(hidden_size=8, num_attention_heads=8, num_query_groups=2, kv_channels=2)
        wrapper = OFTLinearSplitQKV(base, in_features=8, block_size=4, provider=provider)
    elif kind == "gate_up":
        wrapper = OFTLinearSplitFC1UpGate(base, in_features=8, block_size=4)
    else:
        adapter = OFTRotationModule(in_features=8, block_size=4, input_is_parallel=kind == "row", dtype=dtype)
        wrapper = OFTLinear(base, adapter)
    wrapper.to(dtype=dtype)
    with torch.no_grad():
        for module in wrapper.modules():
            if isinstance(module, OFTRotationModule):
                module.oft_r.normal_(std=0.02)
    return wrapper


def _assert_wrapper_parity(wrapper, monkeypatch, dtype):
    monkeypatch.delenv("MEGATRON_OFT_MATERIALIZE_DENSE", raising=False)
    trainable = {name: id(param) for name, param in wrapper.named_parameters() if param.requires_grad}
    assert trainable and all(name.endswith("oft_r") for name in trainable)
    optimizer = torch.optim.SGD((param for param in wrapper.parameters() if param.requires_grad), lr=0.1)
    original_weight = wrapper.to_wrap.weight.detach().clone()
    x_ref = torch.randn(5, 2, 8, dtype=dtype, requires_grad=True)
    expected, expected_bias = wrapper(x_ref)
    output_grad = torch.randn_like(expected)
    expected.backward(output_grad)
    expected_grads = {
        name: param.grad.detach().clone() for name, param in wrapper.named_parameters() if name in trainable
    }
    expected_x_grad = x_ref.grad.detach().clone()
    wrapper.zero_grad(set_to_none=True)

    monkeypatch.setenv("MEGATRON_OFT_MATERIALIZE_DENSE", "1")
    for module in wrapper.modules():
        if isinstance(module, OFTRotationModule):
            monkeypatch.setattr(module, "forward", _fail_activation_path)
    monkeypatch.setattr(canonical_oft, "_apply_precomputed_oft_rotation_to_x", _fail_activation_path)
    monkeypatch.setattr(canonical_oft, "segmented_oft_linear", _fail_activation_path)
    x_actual = x_ref.detach().clone().requires_grad_()
    actual, actual_bias = wrapper(x_actual)
    actual.backward(output_grad)

    # Folding reassociates BF16 products; FP64 tests establish the algebraic parity.
    tolerances = {"rtol": 0.03, "atol": 0.01} if dtype == torch.bfloat16 else {}
    torch.testing.assert_close(actual, expected.detach(), **tolerances)
    torch.testing.assert_close(x_actual.grad, expected_x_grad, **tolerances)
    assert actual_bias is expected_bias
    for name, param in wrapper.named_parameters():
        if name in trainable:
            torch.testing.assert_close(param.grad, expected_grads[name], **tolerances)
    assert {name: id(param) for name, param in wrapper.named_parameters() if param.requires_grad} == trainable
    assert {id(param) for group in optimizer.param_groups for param in group["params"]} == set(trainable.values())
    assert not wrapper.to_wrap.weight.requires_grad
    assert wrapper.to_wrap.weight.grad is None
    torch.testing.assert_close(wrapper.to_wrap.weight, original_weight, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["column", "row", "qkv", "gate_up"])
@pytest.mark.parametrize("skip_bias_add", [False, True])
def test_dense_wrapper_forward_and_adapter_input_gradients_match(cpu_tp, monkeypatch, kind, skip_bias_add):
    torch.manual_seed(23)
    wrapper = _make_wrapper(kind, torch.float64, skip_bias_add, cpu_tp)
    _assert_wrapper_parity(wrapper, monkeypatch, torch.float64)


@pytest.mark.parametrize("kind", ["column", "row", "qkv", "gate_up"])
def test_bf16_dense_wrapper_forward_and_gradients_match(cpu_tp, monkeypatch, kind):
    torch.manual_seed(29)
    wrapper = _make_wrapper(kind, torch.bfloat16, False, cpu_tp)
    _assert_wrapper_parity(wrapper, monkeypatch, torch.bfloat16)


@pytest.mark.parametrize("tp_rank", range(4))
def test_qkv_materialization_preserves_partial_tp_shard_routes(cpu_tp, monkeypatch, tp_rank):
    torch.manual_seed(31)
    # Exercise metadata for shards cutting a GQA group; tensor collectives stay TP1.
    monkeypatch.setattr(canonical_oft, "get_pg_size", lambda group: 4)
    monkeypatch.setattr(canonical_oft, "get_pg_rank", lambda group: tp_rank)
    wrapper = _make_wrapper("qkv", torch.float64, False, cpu_tp, qkv_output_size=6)
    expected_segments = (("q", 0, 6),) if tp_rank % 2 == 0 else (("q", 0, 2), ("k", 2, 4), ("v", 4, 6))
    assert wrapper._segments == expected_segments
    _assert_wrapper_parity(wrapper, monkeypatch, torch.float64)


def test_trainable_base_keeps_activation_path(monkeypatch):
    monkeypatch.setenv("MEGATRON_OFT_MATERIALIZE_DENSE", "1")
    base = nn.Linear(8, 6, bias=False, dtype=torch.float64)
    adapter = _StubRotation()
    wrapper = OFTLinear(base, adapter)
    monkeypatch.setattr(oft_layers, "_materialized_oft_linear_bank", _fail_activation_path, raising=False)

    actual = wrapper(torch.randn(5, 8, dtype=torch.float64))
    actual.sum().backward()

    assert base.weight.grad is not None
    assert adapter.rotation.grad is not None


def test_dense_selector_excludes_fp8_scales_attached_after_wrapping(monkeypatch):
    monkeypatch.setenv("MEGATRON_OFT_MATERIALIZE_DENSE", "1")
    base = nn.Linear(8, 6, bias=False, dtype=torch.float64).requires_grad_(False)
    wrapper = OFTLinear(base, _StubRotation())
    base.register_buffer("weight_scale_inv", torch.ones(1, dtype=torch.float64))

    assert not wrapper._can_use_dense_train_materialization(torch.randn(5, 8, dtype=torch.float64), (), {})


def test_mixed_adapter_dtype_keeps_activation_path(cpu_tp, monkeypatch):
    monkeypatch.setenv("MEGATRON_OFT_MATERIALIZE_DENSE", "1")
    base = nn.Linear(8, 6, bias=False, dtype=torch.float32).requires_grad_(False)
    adapter = OFTRotationModule(in_features=8, block_size=4, dtype=torch.float64)
    with torch.no_grad():
        adapter.oft_r.normal_(std=0.02)
    wrapper = OFTLinear(base, adapter)
    x = torch.randn(5, 8, dtype=torch.float32, requires_grad=True)
    expected = base(adapter(x))
    monkeypatch.setattr(oft_layers, "_materialized_oft_linear_bank", _fail_activation_path)

    actual = wrapper(x)
    actual.sum().backward()

    torch.testing.assert_close(actual, expected)
    assert x.grad is not None
    assert adapter.oft_r.grad is not None
    assert base.weight.grad is None
