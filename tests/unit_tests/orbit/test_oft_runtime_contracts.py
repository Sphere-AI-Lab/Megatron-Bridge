# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression contracts for OFT runtime wrappers.

These tests deliberately use real ``nn.Module`` consumers. A wrapper is only
correct if it can replace its base module without changing the base return
contract.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state

import megatron.bridge.orbit.oft.oft_layers as oft_layers
from megatron.bridge.orbit.oft.oft import VLMOFT
from megatron.bridge.orbit.oft.oft_layers import (
    MultiplicativeDropoutLayer,
    OFTLinear,
    OFTRotationModule,
    OFTTopKRouter,
)


@pytest.fixture(autouse=True)
def _stub_parallel_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_group", lambda: object())
    monkeypatch.setattr(parallel_state, "get_expert_tensor_model_parallel_group", lambda: object(), raising=False)
    monkeypatch.setattr(parallel_state, "get_expert_tensor_parallel_group", lambda: object())


def _rotation(in_features: int = 4, *, dropout: float = 0.0) -> OFTRotationModule:
    adapter = OFTRotationModule(
        in_features=in_features,
        block_size=2,
        module_dropout=dropout,
        input_is_parallel=True,
        dtype=torch.float64,
    )
    with torch.no_grad():
        adapter.oft_r.copy_(torch.tensor([[0.25], [-0.4]], dtype=torch.float64))
    return adapter


@pytest.mark.unit
@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "disabled"])
def test_plain_linear_wrapper_preserves_tensor_contract_through_sequential(enabled: bool) -> None:
    base = nn.Linear(4, 3, dtype=torch.float64)
    wrapper = OFTLinear(base, _rotation())
    if not enabled:
        wrapper.disable_adapter_layers()
    consumer = nn.Sequential(wrapper, nn.ReLU())
    x = torch.tensor([[0.5, -1.0, 2.0, 3.0]], dtype=torch.float64)
    rotated = wrapper.adapter(x) if enabled else x
    expected = F.relu(base(rotated))

    actual = consumer(x)

    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual, expected)


@pytest.mark.unit
def test_block_shared_dropout_never_drops_the_single_shared_rotation() -> None:
    dropout = MultiplicativeDropoutLayer(p=1.0, block_share=True).train()
    shared_rotation = torch.tensor([[[0.0, -1.0], [1.0, 0.0]]])

    actual = dropout(shared_rotation)

    torch.testing.assert_close(actual, shared_rotation)


@pytest.mark.unit
def test_single_nonshared_block_is_still_subject_to_module_dropout() -> None:
    adapter = OFTRotationModule(
        in_features=4,
        r=1,
        module_dropout=1.0,
        input_is_parallel=True,
        dtype=torch.float64,
    ).train()
    with torch.no_grad():
        adapter.oft_r.fill_(0.25)
    x = torch.randn(3, 4, dtype=torch.float64)

    actual = adapter(x)

    torch.testing.assert_close(actual, x)


@pytest.mark.unit
def test_block_shared_module_preserves_its_single_rotation_under_dropout() -> None:
    adapter = OFTRotationModule(
        in_features=4,
        block_size=4,
        block_share=True,
        module_dropout=1.0,
        input_is_parallel=True,
        dtype=torch.float64,
    ).train()
    with torch.no_grad():
        adapter.oft_r.fill_(0.25)
    x = torch.randn(3, 4, dtype=torch.float64)
    expected = adapter.eval()(x)
    adapter.train()

    actual = adapter(x)

    torch.testing.assert_close(actual, expected)
    assert not torch.equal(actual, x)


@pytest.mark.unit
def test_delta_weight_ignores_training_dropout() -> None:
    adapter = _rotation(dropout=1.0)
    adapter.eval()
    expected = adapter.get_delta_weight().clone()
    adapter.train()

    actual = adapter.get_delta_weight()

    torch.testing.assert_close(actual, expected)
    assert not torch.equal(actual, torch.eye(4, dtype=torch.float64))


class _RouterBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            moe_router_force_load_balancing=False,
            moe_router_force_biased=0.25,
        )
        self.layer_number = 7
        self.maintained = 0

    def _maintain_float32_expert_bias(self) -> None:
        self.maintained += 1

    def apply_input_jitter(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1

    def gating(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2

    def routing(self, logits: torch.Tensor, *args, **kwargs):
        return logits, (args, kwargs)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return "base-forward", x, args, kwargs


class _SpecializedRouter(_RouterBase):
    def forward(self, x: torch.Tensor, *args, **kwargs):
        return "dense-inference", x, args, kwargs


@pytest.mark.unit
def test_disabled_router_wrapper_delegates_to_specialized_forward() -> None:
    router = _SpecializedRouter()
    wrapper = OFTTopKRouter(router, nn.Identity())
    wrapper.disable_adapter_layers()
    x = torch.randn(2, 3)

    actual = wrapper(x, "positional", padding_mask="mask")

    assert actual[0] == "dense-inference"
    assert actual[1] is x
    assert actual[2:] == (("positional",), {"padding_mask": "mask"})
    assert router.maintained == 0


@pytest.mark.unit
def test_enabled_router_wrapper_rejects_forward_overriding_subclass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oft_layers, "TopKRouter", _RouterBase)
    wrapper = OFTTopKRouter(_SpecializedRouter(), nn.Identity())

    with pytest.raises(NotImplementedError, match="overrides TopKRouter.forward"):
        wrapper(torch.randn(2, 3))


@pytest.mark.unit
def test_supported_router_wrapper_preserves_biased_logits(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def apply_biased_logits(logits, bias, layer_number):
        calls.append((bias, layer_number))
        return logits + 3

    monkeypatch.setattr(oft_layers, "TopKRouter", _RouterBase)
    monkeypatch.setattr(oft_layers, "apply_biased_logits", apply_biased_logits)
    router = _RouterBase()
    wrapper = OFTTopKRouter(router, nn.Identity())
    x = torch.ones(2, 3)

    logits, routed_args = wrapper(x, padding_mask="mask")

    torch.testing.assert_close(logits, torch.full_like(x, 7.0))
    assert routed_args == ((), {"padding_mask": "mask"})
    assert calls == [(0.25, 7)]
    assert router.maintained == 1


@pytest.mark.unit
def test_vlm_oft_freezes_every_pipeline_chunk_with_partial_vlm_components() -> None:
    class Chunk(nn.Module):
        def __init__(self, component_name: str) -> None:
            super().__init__()
            setattr(self, component_name, nn.Linear(2, 2))

    vision_chunk = Chunk("vision_model").eval()
    language_chunk = Chunk("language_model").eval()

    VLMOFT().freeze_model([vision_chunk, language_chunk], training=True)

    assert all(not parameter.requires_grad for parameter in vision_chunk.vision_model.parameters())
    assert all(not parameter.requires_grad for parameter in language_chunk.language_model.parameters())
    assert vision_chunk.training is True
    assert language_chunk.training is True
