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
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from megatron.bridge.orbit.low_precision.int4 import quantize_to_int4


_REPO_ROOT = Path(__file__).parents[3]
_CONVERSION_DIR = _REPO_ROOT / "scripts" / "orbit" / "conversion"


def _run_script(script_name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_CONVERSION_DIR / script_name), *args],
        cwd=_REPO_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": f"{_REPO_ROOT / 'src'}:{_REPO_ROOT / '3rdparty' / 'Megatron-LM'}",
        },
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.unit
def test_int4_hf_quantizer_writes_exact_index_and_rejects_stale_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"architectures": ["DeepseekV3ForCausalLM"]}')
    expert_key = "model.layers.0.mlp.experts.0.gate_proj.weight"
    kept_key = "model.layers.0.input_layernorm.weight"
    source_tensors = {
        expert_key: torch.linspace(-1, 1, 64, dtype=torch.bfloat16).reshape(2, 32),
        kept_key: torch.ones(2, dtype=torch.bfloat16),
    }
    save_file(source_tensors, source / "model-00001-of-00001.safetensors")
    output = tmp_path / "output"

    first = _run_script(
        "quantize_to_int4.py",
        "--input",
        str(source),
        "--output",
        str(output),
        "--group-size",
        "32",
    )

    assert first.returncode == 0, first.stderr
    output_tensors = load_file(output / "model-00001-of-00001.safetensors")
    expected_total_size = sum(tensor.numel() * tensor.element_size() for tensor in output_tensors.values())
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == expected_total_size
    assert set(index["weight_map"]) == set(output_tensors)

    (output / "unrelated.txt").write_text("do not delete or mix")
    second = _run_script(
        "quantize_to_int4.py",
        "--input",
        str(source),
        "--output",
        str(output),
    )
    assert second.returncode != 0
    assert "output directory must be empty or absent" in second.stderr
    assert (output / "unrelated.txt").read_text() == "do not delete or mix"


@pytest.mark.unit
def test_int4_quantizer_validates_group_shape_before_reshape() -> None:
    with pytest.raises(ValueError, match="divisible by group_size"):
        quantize_to_int4(torch.randn(2, 64, dtype=torch.bfloat16), group_size=48)


@pytest.mark.unit
@pytest.mark.parametrize(
    "script_name",
    [
        "convert_fp4_checkpoint_direct.py",
        "convert_fp8_checkpoint_direct.py",
        "convert_int4_checkpoint_direct.py",
        "convert_nvfp4_checkpoint_direct.py",
    ],
)
def test_direct_converter_help_imports_cleanly(script_name: str) -> None:
    result = _run_script(script_name, "--help")

    assert result.returncode == 0, result.stderr
    assert "--hf-model-path" in result.stdout
    assert "--megatron-path" in result.stdout


@pytest.mark.unit
@pytest.mark.parametrize(
    "obsolete_name",
    [
        "convert_fp8_checkpoint.py",
        "convert_int4_checkpoint.py",
        "dump_nvfp4_meta_keys.py",
    ],
)
def test_superseded_conversion_entrypoints_are_removed(obsolete_name: str) -> None:
    assert not (_CONVERSION_DIR / obsolete_name).exists()
