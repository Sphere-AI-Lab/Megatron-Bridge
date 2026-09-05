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
import torch.nn as nn

from megatron.bridge.orbit.conversion.compressed_tensors_int4 import CompressedTensorsINT4DequantMixin
from megatron.bridge.orbit.conversion.fp8_preserve import BlockFP8PreserveMixin
from megatron.bridge.orbit.conversion.modelopt_nvfp4 import ModelOptNVFP4DequantMixin
from megatron.bridge.orbit.model_bridges import (
    deepseek_v3_int4_bridge,
    kimi_k25_vl_nvfp4_bridge,
    qwen3_moe_provider_ext,
)
from megatron.bridge.orbit.model_bridges.deepseek_v3_int4_bridge import DeepSeekV3INT4Bridge
from megatron.bridge.orbit.model_bridges.kimi_k25_vl_nvfp4_bridge import KimiK25VLNVFP4Bridge
from megatron.bridge.orbit.model_bridges.llama_int4_bridge import LlamaINT4Bridge
from megatron.bridge.orbit.model_bridges.qwen3_int4_bridge import Qwen3INT4Bridge, Qwen3MoEINT4Bridge
from megatron.bridge.orbit.model_bridges.qwen3_moe_fp8_bridge import Qwen3MoEFP8Bridge
from megatron.bridge.orbit.model_bridges.qwen3_moe_provider_ext import Qwen3MoEOrbitProviderMixin


@pytest.mark.unit
@pytest.mark.parametrize("bridge_cls", [LlamaINT4Bridge, Qwen3INT4Bridge, Qwen3MoEINT4Bridge, DeepSeekV3INT4Bridge])
def test_int4_bridges_put_dequant_mixin_first(bridge_cls: type) -> None:
    assert bridge_cls.__mro__.index(CompressedTensorsINT4DequantMixin) == 1


@pytest.mark.unit
def test_quantized_moe_bridges_include_provider_and_precision_mixins() -> None:
    assert issubclass(Qwen3MoEINT4Bridge, Qwen3MoEOrbitProviderMixin)
    assert Qwen3MoEFP8Bridge.__mro__.index(BlockFP8PreserveMixin) == 1
    assert issubclass(Qwen3MoEFP8Bridge, Qwen3MoEOrbitProviderMixin)
    assert KimiK25VLNVFP4Bridge.__mro__.index(ModelOptNVFP4DequantMixin) == 1


@pytest.mark.unit
def test_qwen3_moe_provider_derives_sparse_layer_pattern() -> None:
    class _Base:
        def provider_bridge(self, hf_pretrained):
            return SimpleNamespace()

    class _Bridge(Qwen3MoEOrbitProviderMixin, _Base):
        pass

    hf_pretrained = SimpleNamespace(
        config=SimpleNamespace(
            decoder_sparse_step=2,
            mlp_only_layers=[3],
            num_experts=8,
            num_hidden_layers=6,
        )
    )

    provider = _Bridge().provider_bridge(hf_pretrained)

    assert provider.moe_router_dtype == "fp32"
    assert provider.moe_layer_freq == [0, 1, 0, 0, 0, 1]


@pytest.mark.unit
def test_qwen3_moe_provider_mixin_delegates_to_shared_settings_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = SimpleNamespace()
    hf_config = SimpleNamespace()

    class _Base:
        def provider_bridge(self, hf_pretrained):
            return provider

    class _Bridge(Qwen3MoEOrbitProviderMixin, _Base):
        pass

    calls = []

    def apply_settings(actual_provider, actual_config):
        calls.append((actual_provider, actual_config))
        return actual_provider

    monkeypatch.setattr(qwen3_moe_provider_ext, "apply_qwen3_moe_orbit_provider_settings", apply_settings)

    assert _Bridge().provider_bridge(SimpleNamespace(config=hf_config)) is provider
    assert calls == [(provider, hf_config)]


@pytest.mark.unit
def test_deepseek_requantize_registers_source_scale_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    packed = torch.tensor([[1, 2]], dtype=torch.int32)
    scale = torch.tensor([[0.25]], dtype=torch.float16)
    shape = torch.tensor([1, 16], dtype=torch.int32)
    source_scale = torch.tensor([[0.5]], dtype=torch.float16)
    calls = []

    def fake_requantize(weight, supplied_scale):
        calls.append((weight.clone(), supplied_scale.clone()))
        return packed, scale, shape

    monkeypatch.setattr(deepseek_v3_int4_bridge, "requantize_int4_with_scales", fake_requantize)
    module = nn.Linear(16, 1, bias=False, dtype=torch.bfloat16)
    module.weight.data.fill_(2)
    bridge = DeepSeekV3INT4Bridge.__new__(DeepSeekV3INT4Bridge)

    saved_bytes = bridge._quantize_one_weight(module, "weight", module.weight.data, 32, source_scale)

    assert len(calls) == 1
    assert torch.equal(calls[0][1], source_scale)
    assert torch.equal(module.weight_packed, packed)
    assert torch.equal(module.weight_scale, scale)
    assert torch.equal(module.weight_shape, shape)
    assert torch.count_nonzero(module.weight) == 0
    assert saved_bytes == 32 - (packed.numel() * packed.element_size() + scale.numel() * scale.element_size())


@pytest.mark.unit
def test_kimi_nvfp4_bridge_emits_quantized_expert_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = (
        torch.tensor([[1]], dtype=torch.uint8),
        torch.tensor([[2]], dtype=torch.uint8),
        torch.tensor([3.0]),
        torch.tensor([1, 2], dtype=torch.int32),
    )
    monkeypatch.setattr(kimi_k25_vl_nvfp4_bridge, "quantize_to_nvfp4", lambda tensor: bundle)
    bridge = KimiK25VLNVFP4Bridge.__new__(KimiK25VLNVFP4Bridge)
    expert_key = "decoder.layers.1.mlp.experts.linear_fc1.weight"
    dense_key = "decoder.layers.1.self_attention.linear_proj.weight"

    result = bridge.maybe_modify_converted_hf_weight(
        None,
        {expert_key: torch.ones(1, 2), dense_key: torch.zeros(1, 2)},
        {},
    )

    expert_base = expert_key.removesuffix(".weight")
    assert result[f"{expert_base}_packed_fp4"] is bundle[0]
    assert result[f"{expert_base}_scale_fp4"] is bundle[1]
    assert result[f"{expert_base}_scale_2_fp4"] is bundle[2]
    assert result[f"{expert_base}_shape_fp4"] is bundle[3]
    assert torch.equal(result[dense_key], torch.zeros(1, 2))
