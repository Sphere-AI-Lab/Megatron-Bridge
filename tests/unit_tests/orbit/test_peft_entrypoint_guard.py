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

"""finetune_peft.py canonical-OFT guard.

finetune_qoft.py rejects canonical OFT on grouped-expert FC1 under FP8/NVFP4 at
launch; finetune_peft.py shipped without the equivalent, so a MoE model under
--quant nvfp4 sailed to the first training step and died in
OFTLinearGroupedSplitFC1UpGate._assert_unquantized. These tests pin the
launch-time guard.
"""

import importlib.util
import json
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "orbit" / "finetune_peft.py"


def _load_peft_entrypoint():
    spec = importlib.util.spec_from_file_location("peft_entrypoint_under_test", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
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
def test_peft_guard_rejects_canonical_on_moe_quantized(tmp_path: Path, quant: str) -> None:
    entrypoint = _load_peft_entrypoint()
    model_path = _model_dir(tmp_path, {"architectures": ["Qwen3MoeForCausalLM"], "num_experts": 128})

    with pytest.raises(SystemExit, match="grouped-expert linear_fc1"):
        entrypoint.build_peft(_args(entrypoint, model_path, "--quant", quant))


@pytest.mark.unit
def test_peft_guard_allows_dense_quantized_and_moe_bf16(tmp_path: Path) -> None:
    entrypoint = _load_peft_entrypoint()
    from megatron.bridge.orbit.oft.canonical_oft import CanonicalOFT

    dense = _model_dir(tmp_path, {"architectures": ["Qwen3ForCausalLM"]})
    assert isinstance(entrypoint.build_peft(_args(entrypoint, dense, "--quant", "nvfp4")), CanonicalOFT)

    moe_dir = tmp_path / "moe"
    moe_dir.mkdir()
    (moe_dir / "config.json").write_text(json.dumps({"num_experts": 128}))
    assert isinstance(entrypoint.build_peft(_args(entrypoint, str(moe_dir), "--quant", "none")), CanonicalOFT)


@pytest.mark.unit
def test_peft_guard_legacy_opt_out_still_works_on_moe_quantized(tmp_path: Path) -> None:
    entrypoint = _load_peft_entrypoint()
    from megatron.bridge.orbit.oft.oft import OFT

    model_path = _model_dir(tmp_path, {"num_experts": 128})
    peft = entrypoint.build_peft(_args(entrypoint, model_path, "--quant", "nvfp4", "--oft-type", "oft"))
    assert isinstance(peft, OFT)
