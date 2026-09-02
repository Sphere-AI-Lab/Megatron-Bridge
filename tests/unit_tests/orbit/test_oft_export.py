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

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from megatron.bridge.orbit.conversion import oft_export


@dataclass
class _OFTConfig:
    block_size: int = 16
    module_dropout: float = 0.1
    target_modules: tuple[str, ...] = ("linear_qkv",)
    layers_to_transform: tuple[int, ...] = (1, 3)


@pytest.mark.unit
@pytest.mark.parametrize(("n_elements", "block_size"), [(1, 2), (6, 4), (120, 16)])
def test_infer_oft_block_size(n_elements: int, block_size: int) -> None:
    assert oft_export._infer_oft_block_size_from_n_elements(n_elements) == block_size


@pytest.mark.unit
def test_infer_oft_block_size_rejects_non_triangular_length() -> None:
    with pytest.raises(ValueError, match="Cannot infer OFT block size"):
        oft_export._infer_oft_block_size_from_n_elements(5)


@pytest.mark.unit
def test_globalize_dsv4_expert_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oft_export.parallel_state, "get_expert_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(oft_export.parallel_state, "get_expert_model_parallel_rank", lambda: 2)

    result = oft_export._globalize_dsv4_native_expert_oft_base_prefix(
        "decoder.layers.3.mlp.experts.1.w2",
        num_moe_experts=16,
    )

    assert result == "decoder.layers.3.mlp.experts.9.w2"


@pytest.mark.unit
def test_build_oft_adapter_config_uses_hf_names_and_json_types() -> None:
    result = oft_export.build_oft_adapter_config_dict(
        _OFTConfig(),
        target_modules=["gate_proj", "q_proj"],
        base_model_name_or_path="radixark/base-model",
    )

    assert result["peft_type"] == "OFT"
    assert result["oft_block_size"] == 16
    assert "block_size" not in result
    assert result["target_modules"] == ["gate_proj", "q_proj"]
    assert result["base_model_name_or_path"] == "radixark/base-model"
    assert result["module_dropout"] == 0.1
    assert result["layers_to_transform"] == [1, 3]


@pytest.mark.unit
def test_infer_oft_target_modules_ignores_non_adapter_weights() -> None:
    names = [
        "model.layers.0.self_attn.q_proj.oft_R.weight",
        "model.layers.0.mlp.gate_proj.oft_R.weight",
        "model.layers.1.self_attn.q_proj.oft_R.weight",
        "model.embed_tokens.weight",
    ]

    assert oft_export.infer_oft_target_modules(names) == ["gate_proj", "q_proj"]


@pytest.mark.unit
def test_save_hf_oft_adapter_writes_loadable_peft_directory(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = [
        ("model.layers.0.self_attn.q_proj.oft_R.weight", torch.arange(6, dtype=torch.float32).reshape(2, 3)),
        ("model.layers.0.mlp.gate_proj.oft_R.weight", torch.ones(2, 3)),
    ]
    monkeypatch.setattr(oft_export, "export_oft_adapter_weights", lambda *args, **kwargs: iter(weights))
    auto_bridge = SimpleNamespace(hf_pretrained=SimpleNamespace(model_name_or_path="radixark/base-model"))

    oft_export.save_hf_oft_adapter(
        auto_bridge,
        object(),
        tmp_path,
        _OFTConfig(),
        show_progress=False,
    )

    config = json.loads((tmp_path / "adapter_config.json").read_text())
    saved = load_file(tmp_path / "adapter_model.safetensors")
    assert config["base_model_name_or_path"] == "radixark/base-model"
    assert config["target_modules"] == ["gate_proj", "q_proj"]
    assert set(saved) == {
        "base_model.model.model.layers.0.mlp.gate_proj.oft_R.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.oft_R.weight",
    }
    assert torch.equal(saved["base_model.model.model.layers.0.self_attn.q_proj.oft_R.weight"], weights[0][1])


@pytest.mark.unit
def test_saved_adapter_roundtrips_through_peft_from_pretrained(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The on-disk key must become ``oft_R.default.weight`` inside PEFT."""
    from peft import PeftModel
    from transformers import LlamaConfig, LlamaForCausalLM

    expected = torch.arange(240, dtype=torch.float32).reshape(2, 120)
    weights = [("model.layers.0.self_attn.q_proj.oft_R.weight", expected)]
    monkeypatch.setattr(oft_export, "export_oft_adapter_weights", lambda *args, **kwargs: iter(weights))

    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=64,
    )
    base_model = LlamaForCausalLM(config)
    auto_bridge = SimpleNamespace(hf_pretrained=SimpleNamespace(model_name_or_path="local/tiny-llama"))
    oft_export.save_hf_oft_adapter(
        auto_bridge,
        object(),
        tmp_path,
        _OFTConfig(target_modules=("linear_qkv",), layers_to_transform=(0,)),
        show_progress=False,
    )

    loaded = PeftModel.from_pretrained(base_model, tmp_path)
    actual = loaded.base_model.model.model.layers[0].self_attn.q_proj.oft_R["default"].weight
    torch.testing.assert_close(actual, expected)


@pytest.mark.unit
def test_save_hf_oft_adapter_rejects_empty_model(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oft_export, "export_oft_adapter_weights", lambda *args, **kwargs: iter(()))

    with pytest.raises(RuntimeError, match="No adapter weights were found"):
        oft_export.save_hf_oft_adapter(SimpleNamespace(), object(), tmp_path, _OFTConfig(), show_progress=False)
