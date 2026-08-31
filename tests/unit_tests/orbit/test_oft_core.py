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

import pytest
import torch
import torch.nn as nn
from megatron.core import parallel_state
from megatron.core.transformer import utils as transformer_utils

from megatron.bridge.orbit.oft.canonical_oft import _split_wrapper_sharded_state_dict
from megatron.bridge.orbit.oft.oft_layers import MultiplicativeDropoutLayer, OFTRotationModule
from megatron.bridge.orbit.oft.param_names import is_peft_adapter_param_name, is_trainable_base_param_name


@pytest.fixture(autouse=True)
def mock_tensor_parallel_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_group", lambda: object())


@pytest.mark.unit
def test_oft_rotation_starts_as_identity() -> None:
    adapter = OFTRotationModule(in_features=8, block_size=4, input_is_parallel=True, dtype=torch.float32)
    x = torch.randn(3, 8)

    assert torch.equal(adapter(x), x)
    assert torch.equal(adapter.get_delta_weight(), torch.eye(8))


@pytest.mark.unit
def test_exact_cayley_rotation_is_orthogonal_and_differentiable() -> None:
    adapter = OFTRotationModule(in_features=4, block_size=4, input_is_parallel=True, dtype=torch.float64)
    q = torch.tensor([[0.1, -0.2, 0.05, 0.15, -0.1, 0.2]], dtype=torch.float64, requires_grad=True)

    rotation = adapter._cayley_batch(q, block_size=4, use_cayley_neumann=False)
    identity = torch.eye(4, dtype=torch.float64).unsqueeze(0)

    torch.testing.assert_close(rotation @ rotation.transpose(-1, -2), identity, atol=1e-12, rtol=1e-12)
    rotation.square().sum().backward()
    assert q.grad is not None
    assert torch.isfinite(q.grad).all()


@pytest.mark.unit
def test_oft_rotation_adjusts_invalid_block_size_to_nearest_divisor() -> None:
    adapter = OFTRotationModule(in_features=12, block_size=5, input_is_parallel=True)

    assert adapter.block_size == 6
    assert adapter.r == 2
    assert adapter.oft_r.shape == (2, 15)


@pytest.mark.unit
def test_multiplicative_dropout_replaces_every_block_with_identity() -> None:
    dropout = MultiplicativeDropoutLayer(p=1.0).train()
    rotations = torch.randn(3, 4, 4)

    result = dropout(rotations)

    torch.testing.assert_close(result, torch.eye(4).repeat(3, 1, 1))


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    [
        "decoder.layers.0.linear_qkv.adapter.oft_r",
        "decoder.layers.0.linear_qkv.adapter_q.oft_r",
        "decoder.layers.0.linear_fc1.adapter_gate.oft_r",
        "decoder.layers.0.linear_fc1.lora_a",
    ],
)
def test_parameter_name_predicates_recognize_all_adapter_shapes(name: str) -> None:
    assert is_peft_adapter_param_name(name)
    assert not is_trainable_base_param_name(name)


@pytest.mark.unit
def test_parameter_name_predicates_reject_wrapped_and_adapter_base_parameters() -> None:
    assert not is_trainable_base_param_name("decoder.layers.0.linear_qkv.to_wrap.weight")
    assert is_trainable_base_param_name("decoder.layers.0.linear_qkv.weight")


@pytest.mark.unit
def test_split_wrapper_sharded_state_dict_delegates_each_child(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_sharded_state_dict_default(child, prefix, sharded_offsets, metadata, tp_group):
        calls.append((child, prefix, sharded_offsets, metadata, tp_group))
        return {f"{prefix}weight": child.weight}

    monkeypatch.setattr(transformer_utils, "sharded_state_dict_default", fake_sharded_state_dict_default)
    wrapper = nn.Module()
    wrapper.to_wrap = nn.Linear(4, 4, bias=False)
    wrapper.to_wrap.tp_group = "base-tp"
    wrapper.adapter_q = nn.Linear(4, 4, bias=False)
    wrapper.adapter_q.tp_group = "adapter-tp"

    result = _split_wrapper_sharded_state_dict(
        wrapper,
        prefix="decoder.",
        sharded_offsets=((0, 0, 1),),
        metadata={"dp_cp_group": "dp"},
    )

    assert set(result) == {"decoder.to_wrap.weight", "decoder.adapter_q.weight"}
    assert [(prefix, tp_group) for _, prefix, _, _, tp_group in calls] == [
        ("decoder.to_wrap.", "base-tp"),
        ("decoder.adapter_q.", "adapter-tp"),
    ]
