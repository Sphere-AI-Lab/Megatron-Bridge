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
