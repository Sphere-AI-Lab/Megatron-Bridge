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

from megatron.bridge.orbit.peft_ext import adapter_attrs, recompute_ext
from megatron.bridge.orbit.peft_ext.bias_normalization import normalize_disabled_bias_placeholders
from megatron.bridge.orbit.peft_ext.meta_init import to_empty_if_meta_device
from megatron.bridge.orbit.peft_ext.peft_mixin import OrbitPEFTMixin
from megatron.bridge.peft.recompute import PEFT_RECOMPUTE_PATCHED
from megatron.bridge.peft.utils import AdapterAttributes


class _FakeTELayerNormColumnParallelLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(sequence_parallel=True)
        self.return_layernorm_output = True
        self.return_layernorm_output_gathered = True


class _DummyTransformerBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input_requires_grad = None

    def forward(self, hidden_states):
        self.last_input_requires_grad = hidden_states.requires_grad
        return hidden_states


class _OFTOnlyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(recompute_method="uniform")
        self.block = _DummyTransformerBlock()
        self.base = torch.nn.Linear(2, 2, bias=False)
        self.base.weight.requires_grad = False
        self.adapter_q = torch.nn.Module()
        self.adapter_q.register_parameter("oft_r", torch.nn.Parameter(torch.ones(1)))


@pytest.mark.unit
def test_oft_adapter_attributes_restore_te_layernorm_output_state(monkeypatch: pytest.MonkeyPatch) -> None:
    attrs = AdapterAttributes(
        input_is_parallel=False,
        in_features=8,
        out_features=16,
        disable_tensor_parallel_comm=False,
        disable_sequence_parallel_comm=True,
        base_linear_is_parallel=True,
    )
    module = _FakeTELayerNormColumnParallelLinear()
    monkeypatch.setattr(adapter_attrs, "HAVE_TE", True)
    monkeypatch.setattr(adapter_attrs, "TELayerNormColumnParallelLinear", _FakeTELayerNormColumnParallelLinear)
    monkeypatch.setattr(adapter_attrs, "get_adapter_attributes_from_linear", lambda _module, is_expert=False: attrs)

    result = adapter_attrs.get_oft_adapter_attributes_from_linear(module)

    assert not module.return_layernorm_output
    assert not module.return_layernorm_output_gathered
    assert not result.disable_sequence_parallel_comm


@pytest.mark.unit
def test_bias_normalization_removes_disabled_linear_bias() -> None:
    linear = torch.nn.Linear(4, 4, bias=True)
    linear.config = SimpleNamespace(add_bias_linear=False, add_qkv_bias=False)

    normalize_disabled_bias_placeholders(linear, name="projection")

    assert linear.bias is None


@pytest.mark.unit
def test_bias_normalization_preserves_non_linear_router_bias() -> None:
    router = torch.nn.Module()
    router.config = SimpleNamespace(add_bias_linear=False, add_qkv_bias=False)
    router.register_parameter("bias", torch.nn.Parameter(torch.ones(4)))

    normalize_disabled_bias_placeholders(router, name="router")

    assert router.bias is not None


@pytest.mark.unit
def test_meta_materialization_unwraps_modelopt_qtensor_wrapper() -> None:
    class QTensorWrapper(torch.Tensor):
        pass

    wrapped = torch.Tensor._make_subclass(QTensorWrapper, torch.empty(2, 3, device="meta"), False)
    module = torch.nn.Module()
    module.register_buffer("packed_weight", wrapped)

    to_empty_if_meta_device(module, device=torch.device("cpu"))

    assert module.packed_weight.device.type == "cpu"
    assert type(module.packed_weight) is torch.Tensor
    assert module.packed_weight.shape == (2, 3)


@pytest.mark.unit
def test_oft_only_recompute_patches_transformer_block(monkeypatch: pytest.MonkeyPatch) -> None:
    import megatron.core.transformer.transformer_block as transformer_block

    monkeypatch.setattr(transformer_block, "TransformerBlock", _DummyTransformerBlock)
    monkeypatch.setattr(recompute_ext, "print_rank_0", lambda message: None)
    PEFT_RECOMPUTE_PATCHED.clear()
    model = _OFTOnlyModel()

    recompute_ext.maybe_enable_recompute_inputs_grad_orbit(model)
    model.block(torch.zeros(2, 2))

    assert id(model) in PEFT_RECOMPUTE_PATCHED
    assert model.block.last_input_requires_grad is True


@pytest.mark.unit
def test_orbit_adapter_filter_recognizes_oft_names_and_tuple_trainability() -> None:
    peft = OrbitPEFTMixin()
    peft.params_to_save = {"explicitly.saved"}
    trainable = torch.nn.Parameter(torch.ones(1))
    frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    assert peft.adapter_key_filter("decoder.layers.0.linear_qkv.adapter_q.oft_r")
    assert peft.adapter_key_filter("explicitly.saved")
    assert not peft.adapter_key_filter("decoder.layers.0.linear_qkv.weight")
    assert peft.adapter_key_filter(("adapter", trainable))
    assert not peft.adapter_key_filter(("base", frozen))
