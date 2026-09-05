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

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest
import torch
from safetensors.torch import load_file, save_file


_REPO_ROOT = Path(__file__).parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "orbit" / "conversion" / "quantize_to_int4.py"


def _load_quantizer_script():
    spec = importlib.util.spec_from_file_location("quantize_to_int4_under_test", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_quantize_to_int4(
    weight: torch.Tensor,
    *,
    group_size: int,
    scale_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, width = weight.shape
    return (
        torch.zeros((rows, width // 8), dtype=torch.int32),
        torch.ones((rows, width // group_size), dtype=scale_dtype),
        torch.tensor([rows, width], dtype=torch.int32),
    )


@contextmanager
def _stub_int4_quantizer(quantize_fn=_fake_quantize_to_int4):
    module_name = "megatron.bridge.orbit.low_precision.int4"
    int4_module = ModuleType(module_name)
    int4_module.quantize_to_int4 = quantize_fn

    additions: dict[str, ModuleType] = {module_name: int4_module}
    for parent_name in (
        "megatron",
        "megatron.bridge",
        "megatron.bridge.orbit",
        "megatron.bridge.orbit.low_precision",
    ):
        if parent_name not in sys.modules:
            parent = ModuleType(parent_name)
            parent.__path__ = []
            additions[parent_name] = parent

    with pytest.MonkeyPatch.context() as monkeypatch:
        for name, module in additions.items():
            monkeypatch.setitem(sys.modules, name, module)
        yield


def _write_source(
    root: Path,
    *,
    config: dict[str, object],
    shards: list[dict[str, torch.Tensor]],
) -> Path:
    source = root / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(config))
    for index, tensors in enumerate(shards, start=1):
        save_file(tensors, source / f"model-{index:05d}-of-{len(shards):05d}.safetensors")
    return source


def _run_quantizer(source: Path, output: Path, *, group_size: int = 32) -> None:
    quantizer = _load_quantizer_script()
    with _stub_int4_quantizer():
        assert (
            quantizer.main(
                [
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--group-size",
                    str(group_size),
                ]
            )
            == 0
        )


def _expected_quantization_config(group_size: int, targets: list[str]) -> dict[str, object]:
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "targets": targets,
                "weights": {
                    "type": "int",
                    "num_bits": 4,
                    "strategy": "group",
                    "group_size": group_size,
                    "symmetric": True,
                },
                "format": "pack-quantized",
            }
        },
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("model.layers.0.mlp.experts.12.gate_proj.weight", True),
        ("mtp.layers.0.mlp.experts.2.up_proj.weight", True),
        ("model.layers.0.mlp.experts.0.down_proj.weight", True),
        ("model.layers.0.mlp.nonexperts.0.gate_proj.weight", False),
        ("model.layers.0.mlp.experts_extra.0.gate_proj.weight", False),
        ("model.layers.0.mlp.experts.0.not_gate_proj.weight", False),
        ("model.layers.0.mlp.experts.0.gate_proj_extra.weight", False),
        ("model.layers.0.mlp.experts.0.gate_proj.bias", False),
        ("model.layers.0.mlp.experts.zero.gate_proj.weight", False),
    ],
)
def test_should_quantize_requires_exact_routed_expert_weight_path(key: str, expected: bool) -> None:
    quantizer = _load_quantizer_script()

    assert quantizer.should_quantize(key) is expected


@pytest.mark.unit
def test_int4_quantizer_writes_complete_compressed_tensors_config_for_fresh_input(tmp_path: Path) -> None:
    expert_key = "model.layers.0.mlp.experts.0.gate_proj.weight"
    source = _write_source(
        tmp_path,
        config={"model_type": "qwen3_moe"},
        shards=[{expert_key: torch.ones((2, 32), dtype=torch.bfloat16)}],
    )
    output = tmp_path / "output"

    _run_quantizer(source, output)

    config = json.loads((output / "config.json").read_text())
    assert config == {
        "model_type": "qwen3_moe",
        "quantization_config": _expected_quantization_config(32, ["model.layers.0.mlp.experts.0.gate_proj"]),
    }


@pytest.mark.unit
def test_int4_quantizer_targets_only_modules_actually_converted(tmp_path: Path) -> None:
    gate_key = "model.layers.1.mlp.experts.3.gate_proj.weight"
    down_key = "model.layers.0.mlp.experts.1.down_proj.weight"
    false_positive_key = "model.layers.0.mlp.nonexperts.4.up_proj.weight"
    source = _write_source(
        tmp_path,
        config={"model_type": "deepseek_v3"},
        shards=[
            {gate_key: torch.ones((2, 32), dtype=torch.bfloat16)},
            {
                down_key: torch.ones((2, 32), dtype=torch.bfloat16),
                false_positive_key: torch.ones((2, 32), dtype=torch.bfloat16),
            },
        ],
    )
    output = tmp_path / "output"

    _run_quantizer(source, output)

    config = json.loads((output / "config.json").read_text())
    targets = config["quantization_config"]["config_groups"]["group_0"]["targets"]
    assert targets == [
        "model.layers.0.mlp.experts.1.down_proj",
        "model.layers.1.mlp.experts.3.gate_proj",
    ]
    second_shard = load_file(output / "model-00002-of-00002.safetensors")
    assert false_positive_key in second_shard
    assert false_positive_key.removesuffix(".weight") + ".weight_packed" not in second_shard


@pytest.mark.unit
def test_int4_quantizer_replaces_incompatible_inherited_quantization_config(tmp_path: Path) -> None:
    expert_key = "model.layers.0.mlp.experts.0.up_proj.weight"
    source = _write_source(
        tmp_path,
        config={
            "model_type": "qwen3_moe",
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "naive-quantized",
                "quantization_status": "frozen",
                "ignore": ["lm_head"],
                "config_groups": {
                    "old_fp8": {
                        "targets": ["Linear"],
                        "weights": {
                            "type": "float",
                            "num_bits": 8,
                            "strategy": "tensor",
                            "symmetric": False,
                        },
                    }
                },
            },
        },
        shards=[{expert_key: torch.ones((2, 128), dtype=torch.bfloat16)}],
    )
    output = tmp_path / "output"

    _run_quantizer(source, output, group_size=128)

    config = json.loads((output / "config.json").read_text())
    assert config["quantization_config"] == _expected_quantization_config(
        128, ["model.layers.0.mlp.experts.0.up_proj"]
    )


@pytest.mark.unit
def test_int4_quantizer_defers_config_write_until_all_weights_convert(tmp_path: Path) -> None:
    first_key = "model.layers.0.mlp.experts.0.gate_proj.weight"
    second_key = "model.layers.1.mlp.experts.0.gate_proj.weight"
    source = _write_source(
        tmp_path,
        config={"model_type": "qwen3_moe"},
        shards=[
            {first_key: torch.ones((2, 32), dtype=torch.bfloat16)},
            {second_key: torch.ones((2, 32), dtype=torch.bfloat16)},
        ],
    )
    output = tmp_path / "output"
    calls = 0

    def fail_on_second_weight(
        weight: torch.Tensor,
        *,
        group_size: int,
        scale_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal calls
        calls += 1
        staging_paths = list(tmp_path.glob(".output.int4-staging-*"))
        assert len(staging_paths) == 1
        assert not (staging_paths[0] / "config.json").exists()
        if calls == 2:
            raise RuntimeError("injected conversion failure")
        return _fake_quantize_to_int4(weight, group_size=group_size, scale_dtype=scale_dtype)

    quantizer = _load_quantizer_script()
    with _stub_int4_quantizer(fail_on_second_weight):
        with pytest.raises(RuntimeError, match="injected conversion failure"):
            quantizer.main(["--input", str(source), "--output", str(output)])

    assert calls == 2
    assert not output.exists()
    assert list(tmp_path.glob(".output.int4-staging-*")) == []
