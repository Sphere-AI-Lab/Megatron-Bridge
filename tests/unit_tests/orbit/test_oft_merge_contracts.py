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

"""Destructive OFT merge must be complete, deterministic, and atomic."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state

from megatron.bridge.orbit.oft.canonical_oft import (
    CanonicalOFTMerge,
    OFTLinearGroupedSplitFC1UpGate,
    OFTLinearSplitFC1UpGate,
    OFTLinearSplitQKV,
    OFTVocabParallelEmbedding,
    _SplitLNCanonicalOFTFC1,
)
from megatron.bridge.orbit.oft.oft import OFTMerge, _SplitLNOFTLinear
from megatron.bridge.orbit.oft.oft_layers import OFTLinear, OFTRotationModule, OFTTopKRouter


@pytest.fixture(autouse=True)
def _stub_parallel_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_group", lambda: object())
    monkeypatch.setattr(parallel_state, "get_expert_tensor_parallel_group", lambda: object())


def _rotation(in_features: int = 4) -> OFTRotationModule:
    adapter = OFTRotationModule(
        in_features=in_features,
        block_size=2,
        input_is_parallel=True,
        dtype=torch.float64,
    )
    with torch.no_grad():
        adapter.oft_r.copy_(torch.tensor([[0.25], [-0.4]], dtype=torch.float64))
    return adapter


def _coft_rotation(in_features: int = 4) -> OFTRotationModule:
    adapter = OFTRotationModule(
        in_features=in_features,
        block_size=2,
        coft=True,
        eps=0.1,
        input_is_parallel=True,
        dtype=torch.float64,
    )
    with torch.no_grad():
        adapter.oft_r.fill_(1.0)
    return adapter


class _GroupedLinear(nn.Module):
    def __init__(self, *, experts: int = 2, out_features: int = 4, in_features: int = 4) -> None:
        super().__init__()
        self.num_gemms = experts
        self.in_features = in_features
        self.out_features = out_features
        self.config = SimpleNamespace(sequence_parallel=False, gated_linear_unit=True)
        for index in range(experts):
            self.register_parameter(
                f"weight{index}",
                nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float64)),
            )

    def forward(self, x: torch.Tensor, tokens_per_expert):
        chunks = torch.split(x, [int(value) for value in tokens_per_expert], dim=0)
        return torch.cat(
            [F.linear(chunk, getattr(self, f"weight{index}")) for index, chunk in enumerate(chunks)],
            dim=0,
        ), None


@pytest.mark.unit
def test_legacy_merge_unwraps_and_applies_rotation_exactly_once() -> None:
    base = nn.Linear(4, 3, bias=False, dtype=torch.float64)
    wrapper = OFTLinear(base, _rotation())
    model = nn.Sequential(wrapper)
    x = torch.randn(3, 4, dtype=torch.float64)
    expected = base(wrapper.adapter(x)).detach()

    merged = OFTMerge()(model, training=False)

    assert merged is model
    assert model[0] is base
    assert not any(isinstance(module, OFTLinear) for module in model.modules())
    torch.testing.assert_close(model(x), expected)


@pytest.mark.unit
def test_legacy_merge_unwraps_router_and_folds_gating_weight() -> None:
    class Router(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gating = nn.Linear(4, 3, bias=False, dtype=torch.float64)

        def forward(self, x):
            return self.gating(x)

    router = Router()
    adapter = _rotation()
    wrapper = OFTTopKRouter(router, adapter)
    model = nn.Sequential(wrapper)
    x = torch.randn(2, 4, dtype=torch.float64)
    expected = router(adapter(x)).detach()

    OFTMerge()(model, training=False)

    assert model[0] is router
    torch.testing.assert_close(model(x), expected)


@pytest.mark.unit
def test_legacy_merge_unwraps_split_layernorm_wrapper() -> None:
    base = nn.Linear(4, 3, bias=False, dtype=torch.float64)
    adapter = _rotation()
    wrapper = _SplitLNOFTLinear.__new__(_SplitLNOFTLinear)
    nn.Module.__init__(wrapper)
    wrapper._orig_module = base
    wrapper.adapter = adapter
    wrapper._adapter_enabled = True
    model = nn.Sequential(wrapper)
    original_weight = base.weight.detach().clone()
    expected_weight = original_weight @ adapter.get_delta_weight().T

    OFTMerge()(model, training=False)

    assert model[0] is base
    torch.testing.assert_close(base.weight, expected_weight)


@pytest.mark.unit
def test_legacy_merge_folds_every_grouped_expert_weight() -> None:
    base = _GroupedLinear()
    adapter = _rotation()
    wrapper = OFTLinear(base, adapter)
    model = nn.ModuleList([wrapper])
    x = torch.randn(4, 4, dtype=torch.float64)
    tokens = torch.tensor([2, 2])
    expected, _ = base(adapter(x), tokens)

    OFTMerge()(model, training=False)

    assert model[0] is base
    actual, bias = model[0](x, tokens)
    assert bias is None
    torch.testing.assert_close(actual, expected)


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["quantized", "shape"], ids=["quantized", "shape-mismatch"])
def test_legacy_merge_preflights_all_wrappers_before_mutating(failure: str) -> None:
    first_base = nn.Linear(4, 3, bias=False, dtype=torch.float64)
    first = OFTLinear(first_base, _rotation())
    if failure == "quantized":
        second_base = nn.Linear(4, 3, bias=False, dtype=torch.float64)
        second_base.register_buffer("weight_packed", torch.zeros(3, 2, dtype=torch.uint8))
    else:
        second_base = nn.Linear(3, 3, bias=False, dtype=torch.float64)
    second = OFTLinear(second_base, _rotation())
    model = nn.Sequential(first, second)
    before = first_base.weight.detach().clone()

    with pytest.raises(ValueError, match="quantized|shape mismatch"):
        OFTMerge()(model, training=False)

    torch.testing.assert_close(first_base.weight, before)
    assert model[0] is first
    assert model[1] is second


@pytest.mark.unit
def test_canonical_merge_unwraps_plain_and_split_fc1_wrappers() -> None:
    plain_base = nn.Linear(4, 3, bias=False, dtype=torch.float64)
    plain = OFTLinear(plain_base, _rotation())
    split_base = nn.Linear(4, 6, bias=False, dtype=torch.float64)
    split_base.config = SimpleNamespace(sequence_parallel=False, gated_linear_unit=True)
    split = OFTLinearSplitFC1UpGate(split_base, in_features=4, block_size=2, input_is_parallel=True)
    split.to(torch.float64)
    with torch.no_grad():
        split.adapter_gate.oft_r.copy_(torch.tensor([[0.2], [0.1]], dtype=torch.float64))
        split.adapter_up.oft_r.copy_(torch.tensor([[-0.3], [0.4]], dtype=torch.float64))
    model = nn.Sequential(plain, split)
    x = torch.randn(2, 4, dtype=torch.float64)
    expected_plain = plain_base(plain.adapter(x)).detach()
    expected_split, _ = split(x)

    CanonicalOFTMerge()(model, training=False)

    assert model[0] is plain_base
    assert model[1] is split_base
    torch.testing.assert_close(model[0](x), expected_plain)
    torch.testing.assert_close(model[1](x), expected_split)


@pytest.mark.unit
def test_canonical_merge_unwraps_qkv_and_fused_layernorm_split() -> None:
    provider = SimpleNamespace(
        num_attention_heads=2,
        num_query_groups=2,
        kv_channels=1,
        attention_output_gate=False,
        sequence_parallel=False,
    )
    qkv_base = nn.Linear(4, 6, bias=False, dtype=torch.float64)
    qkv_base.config = provider
    qkv = OFTLinearSplitQKV(qkv_base, in_features=4, provider=provider, block_size=2, input_is_parallel=True)
    qkv.to(torch.float64)

    fc1_base = nn.Linear(4, 6, bias=False, dtype=torch.float64)
    fc1_base.config = SimpleNamespace(sequence_parallel=False, gated_linear_unit=True)
    fc1 = OFTLinearSplitFC1UpGate(fc1_base, in_features=4, block_size=2, input_is_parallel=True).to(torch.float64)
    fused = _SplitLNCanonicalOFTFC1.__new__(_SplitLNCanonicalOFTFC1)
    nn.Module.__init__(fused)
    fused._orig_module = fc1_base
    fused._fc1 = fc1
    model = nn.ModuleList([qkv, fused])

    CanonicalOFTMerge()(model, training=False)

    assert model[0] is qkv_base
    assert model[1] is fc1_base


@pytest.mark.unit
@pytest.mark.parametrize(
    ("active_adapters", "inactive_names"),
    [
        (("q",), {"k", "v"}),
        (("k", "v"), {"q"}),
    ],
    ids=["q-only", "kv-only"],
)
def test_canonical_qkv_subset_forward_and_merge_leave_inactive_rows_bit_identical(
    active_adapters: tuple[str, ...],
    inactive_names: set[str],
) -> None:
    provider = SimpleNamespace(
        num_attention_heads=2,
        num_query_groups=2,
        kv_channels=1,
        attention_output_gate=False,
        sequence_parallel=False,
    )
    base = nn.Linear(4, 6, bias=False, dtype=torch.float64)
    base.config = provider
    wrapper = OFTLinearSplitQKV(
        base,
        in_features=4,
        provider=provider,
        block_size=2,
        input_is_parallel=True,
        active_adapters=active_adapters,
    ).to(torch.float64)
    with torch.no_grad():
        for index, name in enumerate(active_adapters, start=1):
            getattr(wrapper, f"adapter_{name}").oft_r.copy_(
                torch.tensor([[0.1 * index], [-0.2 * index]], dtype=torch.float64)
            )
    x = torch.randn(5, 4, dtype=torch.float64)
    weight_before = base.weight.detach().clone()
    base_out = F.linear(x, weight_before)
    expected, bias = wrapper(x)

    assert bias is None
    assert wrapper._adapter_names == active_adapters
    for name, start, end in wrapper._segments:
        if name in inactive_names:
            assert torch.equal(expected[..., start:end], base_out[..., start:end])

    model = nn.Sequential(wrapper)
    CanonicalOFTMerge()(model, training=False)

    assert model[0] is base
    torch.testing.assert_close(base(x), expected)
    for name, start, end in wrapper._segments:
        if name in inactive_names:
            assert torch.equal(base.weight[start:end], weight_before[start:end])


@pytest.mark.unit
def test_canonical_q_only_copy_slices_matches_eager_gradients_without_inactive_leakage() -> None:
    provider = SimpleNamespace(
        num_attention_heads=2,
        num_query_groups=2,
        kv_channels=1,
        attention_output_gate=False,
        sequence_parallel=False,
    )
    base = nn.Linear(4, 6, bias=False, dtype=torch.float64)
    base.weight.requires_grad_(False)
    base.config = provider
    wrapper = OFTLinearSplitQKV(
        base,
        in_features=4,
        provider=provider,
        block_size=2,
        input_is_parallel=True,
        active_adapters=("q",),
    ).to(torch.float64)
    with torch.no_grad():
        wrapper.adapter_q.oft_r.copy_(torch.tensor([[0.2], [-0.15]], dtype=torch.float64))

    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    probe = torch.randn(3, 6, dtype=torch.float64)
    actual, _ = wrapper(x)
    (actual * probe).sum().backward()
    actual_x_grad = x.grad.detach().clone()
    actual_adapter_grad = wrapper.adapter_q.oft_r.grad.detach().clone()

    wrapper.adapter_q.oft_r.grad = None
    x_ref = x.detach().clone().requires_grad_(True)
    q_input = wrapper.adapter_q(x_ref)
    reference = torch.cat(
        [
            F.linear(q_input if name == "q" else x_ref, base.weight[start:end])
            for name, start, end in wrapper._segments
        ],
        dim=-1,
    )
    (reference * probe).sum().backward()

    torch.testing.assert_close(x_ref.grad, actual_x_grad)
    torch.testing.assert_close(wrapper.adapter_q.oft_r.grad, actual_adapter_grad)
    assert torch.count_nonzero(actual_adapter_grad) > 0

    wrapper.adapter_q.oft_r.grad = None
    inactive_probe = torch.zeros_like(probe)
    for name, start, end in wrapper._segments:
        if name != "q":
            inactive_probe[..., start:end] = 1
    inactive_out, _ = wrapper(x.detach().clone().requires_grad_(True))
    (inactive_out * inactive_probe).sum().backward()

    assert wrapper.adapter_q.oft_r.grad is None or torch.count_nonzero(wrapper.adapter_q.oft_r.grad) == 0


@pytest.mark.unit
def test_canonical_q_only_does_not_create_or_modify_attention_output_gate_adapter() -> None:
    provider = SimpleNamespace(
        num_attention_heads=2,
        num_query_groups=2,
        kv_channels=1,
        attention_output_gate=True,
        sequence_parallel=False,
    )
    base = nn.Linear(4, 8, bias=False, dtype=torch.float64)
    base.config = provider
    wrapper = OFTLinearSplitQKV(
        base,
        in_features=4,
        provider=provider,
        block_size=2,
        input_is_parallel=True,
        active_adapters=("q",),
    ).to(torch.float64)
    with torch.no_grad():
        wrapper.adapter_q.oft_r.copy_(torch.tensor([[0.2], [-0.15]], dtype=torch.float64))

    x = torch.randn(3, 4, dtype=torch.float64)
    weight_before = base.weight.detach().clone()
    base_out = F.linear(x, weight_before)
    expected, bias = wrapper(x)

    assert bias is None
    assert wrapper._adapter_names == ("q",)
    assert not hasattr(wrapper, "adapter_gate")
    assert "adapter_gate.oft_r" not in wrapper.state_dict()
    for name, start, end in wrapper._segments:
        if name != "q":
            assert torch.equal(expected[..., start:end], base_out[..., start:end])

    model = nn.Sequential(wrapper)
    CanonicalOFTMerge()(model, training=False)

    torch.testing.assert_close(base(x), expected)
    for name, start, end in wrapper._segments:
        if name != "q":
            assert torch.equal(base.weight[start:end], weight_before[start:end])


@pytest.mark.unit
@pytest.mark.parametrize("active_adapter", ["gate", "up"])
def test_canonical_fc1_subset_forward_and_merge_leave_inactive_half_bit_identical(
    active_adapter: str,
) -> None:
    base = nn.Linear(4, 6, bias=False, dtype=torch.float64)
    base.config = SimpleNamespace(sequence_parallel=False, gated_linear_unit=True)
    wrapper = OFTLinearSplitFC1UpGate(
        base,
        in_features=4,
        block_size=2,
        input_is_parallel=True,
        active_adapters=(active_adapter,),
    ).to(torch.float64)
    with torch.no_grad():
        getattr(wrapper, f"adapter_{active_adapter}").oft_r.copy_(torch.tensor([[0.3], [-0.25]], dtype=torch.float64))
    x = torch.randn(5, 4, dtype=torch.float64)
    weight_before = base.weight.detach().clone()
    base_out = F.linear(x, weight_before)
    expected, bias = wrapper(x)
    half = base.weight.shape[0] // 2
    inactive_slice = slice(half, None) if active_adapter == "gate" else slice(None, half)

    assert bias is None
    assert wrapper._adapter_names == (active_adapter,)
    assert torch.equal(expected[..., inactive_slice], base_out[..., inactive_slice])

    model = nn.Sequential(wrapper)
    CanonicalOFTMerge()(model, training=False)

    assert model[0] is base
    torch.testing.assert_close(base(x), expected)
    assert torch.equal(base.weight[inactive_slice], weight_before[inactive_slice])


@pytest.mark.unit
def test_canonical_merge_folds_embedding_output_rotation() -> None:
    base = nn.Embedding(5, 4, dtype=torch.float64)
    adapter = _rotation()
    wrapper = OFTVocabParallelEmbedding(base, adapter)
    model = nn.Sequential(wrapper)
    ids = torch.tensor([[0, 3]])
    expected = adapter(base(ids)).detach()

    CanonicalOFTMerge()(model, training=False)

    assert model[0] is base
    torch.testing.assert_close(model(ids), expected)


@pytest.mark.unit
def test_canonical_merge_folds_grouped_fc1_and_unwraps() -> None:
    base = _GroupedLinear(out_features=6)
    wrapper = OFTLinearGroupedSplitFC1UpGate(
        base,
        in_features=4,
        block_size=2,
        input_is_parallel=True,
    ).to(torch.float64)
    with torch.no_grad():
        wrapper.adapter_gate.oft_r.copy_(torch.tensor([[[0.2], [0.1]], [[-0.1], [0.3]]], dtype=torch.float64))
        wrapper.adapter_up.oft_r.copy_(torch.tensor([[[-0.3], [0.4]], [[0.25], [-0.2]]], dtype=torch.float64))
    model = nn.ModuleList([wrapper])
    x = torch.randn(4, 4, dtype=torch.float64)
    tokens = torch.tensor([2, 2])
    expected, _ = wrapper(x, tokens)

    CanonicalOFTMerge()(model, training=False)

    assert model[0] is base
    actual, _ = base(x, tokens)
    torch.testing.assert_close(actual, expected)


@pytest.mark.unit
@pytest.mark.parametrize("active_adapter", ["gate", "up"])
def test_canonical_grouped_fc1_subset_preserves_inactive_half_and_merge_parity(
    active_adapter: str,
) -> None:
    base = _GroupedLinear(out_features=6)
    wrapper = OFTLinearGroupedSplitFC1UpGate(
        base,
        in_features=4,
        block_size=2,
        input_is_parallel=True,
        active_adapters=(active_adapter,),
    ).to(torch.float64)
    with torch.no_grad():
        getattr(wrapper, f"adapter_{active_adapter}").oft_r.copy_(
            torch.tensor([[[0.2], [0.1]], [[-0.1], [0.3]]], dtype=torch.float64)
        )
    x = torch.randn(4, 4, dtype=torch.float64)
    tokens = torch.tensor([2, 2])
    weight_before = [getattr(base, f"weight{index}").detach().clone() for index in range(base.num_gemms)]
    base_out, _ = base(x, tokens)
    expected, bias = wrapper(x, tokens)
    half = base.out_features // 2
    inactive_slice = slice(half, None) if active_adapter == "gate" else slice(None, half)

    assert bias is None
    assert torch.equal(expected[..., inactive_slice], base_out[..., inactive_slice])

    model = nn.ModuleList([wrapper])
    CanonicalOFTMerge()(model, training=False)

    actual, _ = base(x, tokens)
    torch.testing.assert_close(actual, expected)
    for index, before in enumerate(weight_before):
        current = getattr(base, f"weight{index}")
        assert torch.equal(current[inactive_slice], before[inactive_slice])


@pytest.mark.unit
def test_canonical_merge_rejects_quantized_split_before_any_mutation() -> None:
    first_base = nn.Linear(4, 6, bias=False, dtype=torch.float64)
    first_base.config = SimpleNamespace(sequence_parallel=False, gated_linear_unit=True)
    first = OFTLinearSplitFC1UpGate(first_base, in_features=4, block_size=2, input_is_parallel=True)

    second_base = nn.Linear(4, 6, bias=False, dtype=torch.float64)
    second_base.config = SimpleNamespace(sequence_parallel=False, gated_linear_unit=True)
    second_base.register_buffer("weight_packed", torch.zeros(6, 2, dtype=torch.uint8))
    second = OFTLinearSplitFC1UpGate(second_base, in_features=4, block_size=2, input_is_parallel=True)
    model = nn.Sequential(first, second)
    before = first_base.weight.detach().clone()

    with pytest.raises(ValueError, match="quantized"):
        CanonicalOFTMerge()(model, training=False)

    torch.testing.assert_close(first_base.weight, before)
    assert model[0] is first
    assert model[1] is second


@pytest.mark.unit
def test_legacy_merge_precomputes_every_rotation_before_mutating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_base = nn.Linear(4, 3, bias=False, dtype=torch.float64)
    second_base = nn.Linear(4, 3, bias=False, dtype=torch.float64)
    first = OFTLinear(first_base, _coft_rotation())
    second = OFTLinear(second_base, _rotation())
    model = nn.Sequential(first, second)
    first_before = first_base.weight.detach().clone()
    first_adapter_before = first.adapter.oft_r.detach().clone()
    second_before = second_base.weight.detach().clone()

    def malformed_rotation() -> torch.Tensor:
        raise RuntimeError("malformed late adapter")

    monkeypatch.setattr(second.adapter, "get_delta_weight", malformed_rotation)

    with pytest.raises(RuntimeError, match="malformed late adapter"):
        OFTMerge()(model, training=False)

    torch.testing.assert_close(first_base.weight, first_before)
    torch.testing.assert_close(first.adapter.oft_r, first_adapter_before)
    torch.testing.assert_close(second_base.weight, second_before)
    assert model[0] is first
    assert model[1] is second


@pytest.mark.unit
def test_canonical_merge_precomputes_every_rotation_before_mutating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_base = nn.Linear(4, 6, bias=False, dtype=torch.float64)
    first_base.config = SimpleNamespace(sequence_parallel=False, gated_linear_unit=True)
    first = OFTLinearSplitFC1UpGate(
        first_base,
        in_features=4,
        block_size=2,
        coft=True,
        eps=0.1,
        input_is_parallel=True,
    ).to(torch.float64)
    with torch.no_grad():
        first.adapter_gate.oft_r.fill_(1.0)
        first.adapter_up.oft_r.fill_(-1.0)
    second_base = nn.Linear(4, 6, bias=False, dtype=torch.float64)
    second_base.config = SimpleNamespace(sequence_parallel=False, gated_linear_unit=True)
    second = OFTLinearSplitFC1UpGate(
        second_base,
        in_features=4,
        block_size=2,
        input_is_parallel=True,
    ).to(torch.float64)
    model = nn.Sequential(first, second)
    first_before = first_base.weight.detach().clone()
    first_gate_before = first.adapter_gate.oft_r.detach().clone()
    first_up_before = first.adapter_up.oft_r.detach().clone()
    second_before = second_base.weight.detach().clone()

    def malformed_rotation() -> torch.Tensor:
        raise RuntimeError("malformed late adapter")

    monkeypatch.setattr(second.adapter_up, "get_delta_weight", malformed_rotation)

    with pytest.raises(RuntimeError, match="malformed late adapter"):
        CanonicalOFTMerge()(model, training=False)

    torch.testing.assert_close(first_base.weight, first_before)
    torch.testing.assert_close(first.adapter_gate.oft_r, first_gate_before)
    torch.testing.assert_close(first.adapter_up.oft_r, first_up_before)
    torch.testing.assert_close(second_base.weight, second_before)
    assert model[0] is first
    assert model[1] is second


@pytest.mark.unit
def test_canonical_grouped_merge_does_not_project_adapters_before_late_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _GroupedLinear(out_features=6)
    wrapper = OFTLinearGroupedSplitFC1UpGate(
        base,
        in_features=4,
        block_size=2,
        coft=True,
        eps=0.1,
        input_is_parallel=True,
    ).to(torch.float64)
    with torch.no_grad():
        wrapper.adapter_gate.oft_r.fill_(1.0)
        wrapper.adapter_up.oft_r.fill_(-1.0)
    gate_before = wrapper.adapter_gate.oft_r.detach().clone()
    up_before = wrapper.adapter_up.oft_r.detach().clone()
    weight_before = [getattr(base, f"weight{index}").detach().clone() for index in range(base.num_gemms)]
    original_up_delta = wrapper.adapter_up.get_delta_weight

    def fail_on_second_expert(expert_idx: int) -> torch.Tensor:
        if expert_idx == 1:
            raise RuntimeError("malformed late grouped adapter")
        return original_up_delta(expert_idx)

    monkeypatch.setattr(wrapper.adapter_up, "get_delta_weight", fail_on_second_expert)
    model = nn.ModuleList([wrapper])

    with pytest.raises(RuntimeError, match="malformed late grouped adapter"):
        CanonicalOFTMerge()(model, training=False)

    torch.testing.assert_close(wrapper.adapter_gate.oft_r, gate_before)
    torch.testing.assert_close(wrapper.adapter_up.oft_r, up_before)
    for index, before in enumerate(weight_before):
        torch.testing.assert_close(getattr(base, f"weight{index}"), before)
    assert model[0] is wrapper


def _tied_embedding_and_output() -> tuple[nn.Embedding, nn.Linear]:
    embedding = nn.Embedding(5, 4, dtype=torch.float64)
    output = nn.Linear(4, 5, bias=False, dtype=torch.float64)
    output.weight = embedding.weight
    return embedding, output


@pytest.mark.unit
def test_canonical_merge_rejects_weight_tied_to_untargeted_consumer() -> None:
    embedding, output = _tied_embedding_and_output()
    output_wrapper = OFTLinear(output, _rotation())
    model = nn.ModuleDict({"embedding": embedding, "output": output_wrapper})
    before = embedding.weight.detach().clone()

    with pytest.raises(ValueError, match="shared storage|aliased"):
        CanonicalOFTMerge()(model, training=False)

    torch.testing.assert_close(embedding.weight, before)
    assert model["embedding"] is embedding
    assert model["output"] is output_wrapper
    assert output.weight is embedding.weight


@pytest.mark.unit
def test_canonical_merge_rejects_weight_tied_between_two_targeted_consumers() -> None:
    embedding, output = _tied_embedding_and_output()
    embedding_wrapper = OFTVocabParallelEmbedding(embedding, _rotation())
    output_wrapper = OFTLinear(output, _rotation())
    model = nn.ModuleDict({"embedding": embedding_wrapper, "output": output_wrapper})
    before = embedding.weight.detach().clone()

    with pytest.raises(ValueError, match="shared storage|aliased"):
        CanonicalOFTMerge()(model, training=False)

    torch.testing.assert_close(embedding.weight, before)
    assert model["embedding"] is embedding_wrapper
    assert model["output"] is output_wrapper
    assert output.weight is embedding.weight


@pytest.mark.unit
@pytest.mark.parametrize("merge_type", [OFTMerge, CanonicalOFTMerge], ids=["legacy", "canonical"])
def test_merge_rejects_target_base_reachable_outside_its_wrapper(
    merge_type: type[OFTMerge] | type[CanonicalOFTMerge],
) -> None:
    base = nn.Linear(4, 3, bias=False, dtype=torch.float64)
    wrapper = OFTLinear(base, _rotation())
    model = nn.ModuleDict({"wrapped": wrapper, "untargeted": base})
    before = base.weight.detach().clone()

    with pytest.raises(ValueError, match="shared storage|aliased"):
        merge_type()(model, training=False)

    torch.testing.assert_close(base.weight, before)
    assert model["wrapped"] is wrapper
    assert model["untargeted"] is base


@pytest.mark.unit
@pytest.mark.parametrize("merge_type", [OFTMerge, CanonicalOFTMerge], ids=["legacy", "canonical"])
def test_merge_allows_same_wrapper_reachable_through_multiple_paths(
    merge_type: type[OFTMerge] | type[CanonicalOFTMerge],
) -> None:
    base = nn.Linear(4, 3, bias=False, dtype=torch.float64)
    wrapper = OFTLinear(base, _rotation())
    model = nn.ModuleDict({"first": wrapper, "alias": wrapper})
    expected = base.weight.detach().clone() @ wrapper.adapter.get_delta_weight().T

    merge_type()(model, training=False)

    assert model["first"] is base
    assert model["alias"] is base
    torch.testing.assert_close(base.weight, expected)


@pytest.mark.unit
def test_canonical_merge_allows_disjoint_weights_in_one_flat_storage() -> None:
    flat = nn.Parameter(torch.randn(10, 4, dtype=torch.float64))
    embedding = nn.Embedding(5, 4, dtype=torch.float64)
    embedding.weight = nn.Parameter(flat[:5])
    output = nn.Linear(4, 5, bias=False, dtype=torch.float64)
    output.weight = nn.Parameter(flat[5:])
    output_wrapper = OFTLinear(output, _rotation())
    model = nn.ModuleDict({"embedding": embedding, "output": output_wrapper})
    embedding_before = embedding.weight.detach().clone()

    CanonicalOFTMerge()(model, training=False)

    torch.testing.assert_close(embedding.weight, embedding_before)
    assert model["embedding"] is embedding
    assert model["output"] is output
