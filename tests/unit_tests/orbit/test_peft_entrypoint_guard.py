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

"""finetune_peft.py OFT-type selection across model and quantization kinds.

Vanilla OFT is the safe default. CanonicalOFT remains an explicit opt-in while
its unsupported fused-QKV layouts fail at transformation time.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


_REPO_ROOT = Path(__file__).parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "orbit" / "finetune_peft.py"


def _load_peft_entrypoint():
    spec = importlib.util.spec_from_file_location("peft_entrypoint_under_test", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    recipes = types.ModuleType("megatron.bridge.recipes")
    recipes.__path__ = []
    common = types.ModuleType("megatron.bridge.recipes.common")
    finetune_module = types.ModuleType("megatron.bridge.training.finetune")
    gpt_step_module = types.ModuleType("megatron.bridge.training.gpt_step")
    mixed_precision_module = types.ModuleType("megatron.bridge.training.mixed_precision")

    def _unused(*_args, **_kwargs):
        return None

    common._peft_common = _unused
    finetune_module.finetune = _unused
    gpt_step_module.forward_step = _unused
    mixed_precision_module.get_mixed_precision_config = _unused
    with patch.dict(
        sys.modules,
        {
            "megatron.bridge.recipes": recipes,
            "megatron.bridge.recipes.common": common,
            "megatron.bridge.training.finetune": finetune_module,
            "megatron.bridge.training.gpt_step": gpt_step_module,
            "megatron.bridge.training.mixed_precision": mixed_precision_module,
        },
    ):
        spec.loader.exec_module(module)
    return module


def _model_dir(tmp_path: Path, config: dict) -> str:
    model_dir = tmp_path / "hf-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(config))
    return str(model_dir)


def _args(entrypoint, model_path: str, *extra: str):
    return entrypoint.parse_args(
        ["--model-path", model_path, "--pretrained-checkpoint", "/ckpt", "--peft", "oft", *extra]
    )


@pytest.mark.unit
@pytest.mark.parametrize("quant", ["fp8", "nvfp4"])
def test_peft_defaults_to_vanilla_oft_on_moe_quantized(tmp_path: Path, quant: str) -> None:
    from megatron.bridge.orbit.oft.oft import OFT

    entrypoint = _load_peft_entrypoint()
    model_path = _model_dir(tmp_path, {"architectures": ["Qwen3MoeForCausalLM"], "num_experts": 128})

    assert isinstance(entrypoint.build_peft(_args(entrypoint, model_path, "--quant", quant)), OFT)


@pytest.mark.unit
def test_peft_defaults_to_vanilla_oft_on_dense_quantized_and_moe_bf16(tmp_path: Path) -> None:
    entrypoint = _load_peft_entrypoint()
    from megatron.bridge.orbit.oft.oft import OFT

    dense = _model_dir(tmp_path, {"architectures": ["Qwen3ForCausalLM"]})
    assert isinstance(entrypoint.build_peft(_args(entrypoint, dense, "--quant", "nvfp4")), OFT)

    moe_dir = tmp_path / "moe"
    moe_dir.mkdir()
    (moe_dir / "config.json").write_text(json.dumps({"num_experts": 128}))
    assert isinstance(entrypoint.build_peft(_args(entrypoint, str(moe_dir), "--quant", "none")), OFT)


@pytest.mark.unit
def test_peft_canonical_oft_remains_an_explicit_opt_in(tmp_path: Path) -> None:
    entrypoint = _load_peft_entrypoint()

    model_path = _model_dir(tmp_path, {"num_experts": 128})
    peft = entrypoint.build_peft(_args(entrypoint, model_path, "--quant", "nvfp4", "--oft-type", "canonical_oft"))
    assert isinstance(peft, entrypoint.CanonicalOFT)


@pytest.mark.unit
@pytest.mark.parametrize("architecture", ["Qwen3MoeForCausalLM", type("Qwen3MoeForCausalLM", (), {})])
def test_peft_build_config_applies_qwen3_moe_provider_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    architecture: object,
) -> None:
    entrypoint = _load_peft_entrypoint()
    hf_config = SimpleNamespace(
        decoder_sparse_step=2,
        mlp_only_layers=[3],
        num_experts=8,
        num_hidden_layers=6,
    )
    provider = SimpleNamespace(expert_tensor_parallel_size=1)
    auto_bridge = SimpleNamespace(
        _causal_lm_architecture=architecture,
        hf_pretrained=SimpleNamespace(config=hf_config),
        to_megatron_provider=lambda **kwargs: provider,
    )
    config = SimpleNamespace(
        model=None,
        tokenizer=SimpleNamespace(),
        checkpoint=SimpleNamespace(),
        dataset=SimpleNamespace(),
        train=SimpleNamespace(),
    )
    monkeypatch.setattr(entrypoint.AutoBridge, "from_hf_pretrained", lambda path: auto_bridge)
    monkeypatch.setattr(entrypoint, "_peft_common", lambda: config)
    args = _args(entrypoint, str(tmp_path / "qwen3-moe"), "--peft", "none")

    result = entrypoint.build_config(args, world_size=1)

    assert result.model is provider
    assert provider.moe_router_dtype == "fp32"
    assert provider.moe_layer_freq == [0, 1, 0, 0, 0, 1]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("world_size", "tp", "ep", "pp", "cp", "etp", "valid", "error_axis"),
    [
        (4, 2, 4, 1, 1, 1, True, None),
        (8, 4, 4, 2, 1, 1, True, None),
        (3, 2, 1, 1, 1, 1, False, r"TP\*PP\*CP"),
        (6, 3, 4, 1, 1, 1, False, r"ETP\*EP\*PP"),
        (8, 2, 2, 2, 2, 1, True, None),
        (4, 2, 3, 1, 1, 2, False, r"ETP\*EP\*PP"),
    ],
)
def test_peft_parallelism_validates_tensor_and_expert_axes_independently(
    world_size: int,
    tp: int,
    ep: int,
    pp: int,
    cp: int,
    etp: int,
    valid: bool,
    error_axis: str | None,
) -> None:
    entrypoint = _load_peft_entrypoint()
    args = SimpleNamespace(quant="none", tp=tp, ep=ep, pp=pp, cp=cp)

    if valid:
        entrypoint.validate_parallelism(args, world_size=world_size, etp=etp)
    else:
        with pytest.raises(SystemExit, match=error_axis):
            entrypoint.validate_parallelism(args, world_size=world_size, etp=etp)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("WORLD_SIZE", 0),
        ("TP", 0),
        ("EP", -1),
        ("PP", 0),
        ("CP", 0),
        ("ETP", 0),
    ],
)
def test_peft_parallelism_rejects_nonpositive_dimensions(name: str, value: int) -> None:
    entrypoint = _load_peft_entrypoint()
    dimensions = {"WORLD_SIZE": 1, "TP": 1, "EP": 1, "PP": 1, "CP": 1, "ETP": 1}
    dimensions[name] = value
    args = SimpleNamespace(
        quant="none",
        tp=dimensions["TP"],
        ep=dimensions["EP"],
        pp=dimensions["PP"],
        cp=dimensions["CP"],
    )

    with pytest.raises(SystemExit, match=rf"{name}.*positive|positive.*{name}"):
        entrypoint.validate_parallelism(
            args,
            world_size=dimensions["WORLD_SIZE"],
            etp=dimensions["ETP"],
        )
