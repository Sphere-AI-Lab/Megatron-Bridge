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
import os
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from megatron.bridge.orbit.low_precision import fp8 as fp8_low_precision
from megatron.bridge.orbit.low_precision import int4 as int4_low_precision
from megatron.bridge.orbit.low_precision import nvfp4 as nvfp4_low_precision
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


def _load_conversion_script(script_name: str):
    path = _CONVERSION_DIR / script_name
    spec = importlib.util.spec_from_file_location(f"{path.stem}_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot_files(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def _assert_failed_quantizer_publish_is_clean(output: Path, *, existed: bool) -> None:
    if existed:
        assert output.is_dir()
        assert list(output.iterdir()) == []
    else:
        assert not output.exists()
    assert list(output.parent.glob(f".{output.name}.int4-staging-*")) == []


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
    output.mkdir()

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
    expert_scale_key = expert_key.removesuffix(".weight") + ".weight_scale"
    assert output_tensors[expert_scale_key].dtype == torch.float16
    expected_total_size = sum(tensor.numel() * tensor.element_size() for tensor in output_tensors.values())
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == expected_total_size
    assert set(index["weight_map"]) == set(output_tensors)
    assert list(tmp_path.glob(".output.int4-staging-*")) == []

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
def test_int4_quantizer_persists_nondefault_group_size_for_direct_conversion(tmp_path: Path) -> None:
    """The copied HF config must describe the emitted group-size-128 tensors."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3MoeForCausalLM"],
                "model_type": "qwen3_moe",
                "quantization_config": {
                    "format": "pack-quantized",
                    "config_groups": {"existing": {"weights": {"group_size": 32, "strategy": "group"}}},
                },
            }
        )
    )
    expert_key = "model.layers.0.mlp.experts.0.gate_proj.weight"
    save_file({expert_key: torch.linspace(-1, 1, 256).reshape(2, 128)}, source / "model.safetensors")
    output = tmp_path / "output"

    result = _run_script(
        "quantize_to_int4.py",
        "--input",
        str(source),
        "--output",
        str(output),
        "--group-size",
        "128",
    )

    assert result.returncode == 0, result.stderr
    config = json.loads((output / "config.json").read_text())
    quant_config = config["quantization_config"]
    assert config["model_type"] == "qwen3_moe"
    assert quant_config["config_groups"]["group_0"]["weights"]["group_size"] == 128

    converter = _load_conversion_script("convert_int4_checkpoint_direct.py")
    auto_bridge = SimpleNamespace(hf_pretrained=SimpleNamespace(config=SimpleNamespace(**config)))
    assert converter._infer_group_size(auto_bridge, override=None) == 128


@pytest.mark.unit
def test_int4_converter_import_never_mutates_safe_open_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    import safetensors

    import megatron.bridge.models.hf_pretrained.state as hf_state

    def package_safe_open(*args: object, **kwargs: object) -> object:
        return object()

    def cached_safe_open(*args: object, **kwargs: object) -> object:
        return object()

    monkeypatch.setattr(safetensors, "safe_open", package_safe_open)
    monkeypatch.setattr(hf_state, "safe_open", cached_safe_open, raising=False)

    _load_conversion_script("convert_int4_checkpoint_direct.py")
    assert safetensors.safe_open is package_safe_open
    assert hf_state.safe_open is cached_safe_open

    _load_conversion_script("convert_int4_checkpoint_direct.py")
    assert safetensors.safe_open is package_safe_open
    assert hf_state.safe_open is cached_safe_open


@pytest.mark.unit
def test_int4_converter_reader_scope_restores_exact_aliases_on_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import safetensors

    import megatron.bridge.models.hf_pretrained.state as hf_state

    converter = _load_conversion_script("convert_int4_checkpoint_direct.py")
    calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []
    native_result = object()

    def package_safe_open(path: object, *args: object, **kwargs: object) -> object:
        calls.append((path, args, kwargs))
        return native_result

    def cached_safe_open(*args: object, **kwargs: object) -> object:
        return object()

    monkeypatch.setattr(safetensors, "safe_open", package_safe_open)
    monkeypatch.setattr(hf_state, "safe_open", cached_safe_open, raising=False)
    monkeypatch.setenv("MEGATRON_BRIDGE_PYMMAP_READER", "0")

    with converter._temporary_safetensors_reader(max_attempts=1):
        patched = safetensors.safe_open
        assert patched is hf_state.safe_open
        assert patched is not package_safe_open
        assert patched("shard.safetensors", framework="pt") is native_result
    assert calls == [("shard.safetensors", (), {"framework": "pt"})]
    assert safetensors.safe_open is package_safe_open
    assert hf_state.safe_open is cached_safe_open

    with pytest.raises(RuntimeError, match="injected conversion failure"):
        with converter._temporary_safetensors_reader(max_attempts=1):
            assert safetensors.safe_open is hf_state.safe_open
            raise RuntimeError("injected conversion failure")
    assert safetensors.safe_open is package_safe_open
    assert hf_state.safe_open is cached_safe_open


@pytest.mark.unit
def test_int4_converter_reader_scope_restores_absent_hf_state_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import safetensors

    import megatron.bridge.models.hf_pretrained.state as hf_state

    converter = _load_conversion_script("convert_int4_checkpoint_direct.py")
    original_safe_open = safetensors.safe_open
    monkeypatch.delattr(hf_state, "safe_open", raising=False)

    with converter._temporary_safetensors_reader(max_attempts=1):
        assert hf_state.safe_open is safetensors.safe_open

    assert safetensors.safe_open is original_safe_open
    assert not hasattr(hf_state, "safe_open")


@pytest.mark.unit
def test_int4_converter_scopes_safe_open_patch_to_hf_tensor_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    converter = _load_conversion_script("convert_int4_checkpoint_direct.py")
    reader_active = False

    @contextmanager
    def reader_scope():
        nonlocal reader_active
        assert not reader_active
        reader_active = True
        try:
            yield
        finally:
            reader_active = False

    class MetaModel:
        def sharded_state_dict(self, **kwargs: object) -> dict[str, object]:
            assert not reader_active
            return {}

    class Provider:
        def finalize(self) -> None:
            assert not reader_active

        def provide_distributed_model(self, **kwargs: object) -> list[MetaModel]:
            assert not reader_active
            return [MetaModel()]

    hf_pretrained = SimpleNamespace(
        config=SimpleNamespace(quantization_config={"group_size": 32}),
        trust_remote_code=False,
    )
    auto_bridge = SimpleNamespace(
        hf_pretrained=hf_pretrained,
        _causal_lm_architecture="LlamaForCausalLM",
    )
    monkeypatch.delenv("MEGATRON_BRIDGE_DIRECT_USE_SPILL", raising=False)
    monkeypatch.setattr(
        converter,
        "parse_args",
        lambda: SimpleNamespace(
            hf_model_path="hf",
            megatron_path=str(tmp_path / "out"),
            group_size=None,
        ),
    )
    monkeypatch.setattr(
        converter, "build_single_rank_meta_provider", lambda *args, **kwargs: (auto_bridge, Provider())
    )
    monkeypatch.setattr(converter, "_select_int4_bridge", lambda actual: object())
    monkeypatch.setattr(converter, "patch_meta_init_for_te_modules", lambda: None)
    monkeypatch.setattr(converter, "temporary_distributed_context", lambda **kwargs: nullcontext())
    monkeypatch.setattr(converter, "keep_meta_model_unmaterialized", nullcontext)
    monkeypatch.setattr(converter, "get_pg_collection", lambda model: SimpleNamespace(dp_cp=None))
    monkeypatch.setattr(converter, "_temporary_safetensors_reader", reader_scope)

    def build_state(*args: object, **kwargs: object) -> dict[str, object]:
        assert reader_active
        assert kwargs["scale_dtype"] == torch.bfloat16
        return {}

    def save_state(provider: object, path: str | Path, *args: object, **kwargs: object) -> None:
        assert not reader_active
        artifact = Path(path) / "iter_0000000" / "metadata.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_text("{}")

    monkeypatch.setattr(converter, "build_int4_direct_model_state_dict", build_state)
    monkeypatch.setattr(converter, "_save_direct_checkpoint", save_state)

    assert converter.main() == 0
    assert not reader_active
    assert (tmp_path / "out" / "iter_0000000" / "metadata.json").is_file()


@pytest.mark.unit
@pytest.mark.parametrize(
    "quantization_config",
    [
        {"group_size": 128, "config_groups": {"experts": {"weights": {"group_size": 32}}}},
        {"group_size": "128"},
        {"config_groups": []},
        {"config_groups": {"experts": {"weights": None}}},
    ],
)
def test_int4_direct_group_size_inference_rejects_ambiguous_or_malformed_metadata(
    quantization_config: object,
) -> None:
    converter = _load_conversion_script("convert_int4_checkpoint_direct.py")
    auto_bridge = SimpleNamespace(
        hf_pretrained=SimpleNamespace(config=SimpleNamespace(quantization_config=quantization_config))
    )

    with pytest.raises(ValueError, match="group_size|config_groups"):
        converter._infer_group_size(auto_bridge, override=None)


@pytest.mark.unit
@pytest.mark.parametrize("group_size", ["0", "-32"])
def test_int4_quantizer_rejects_nonpositive_group_size_before_creating_output(tmp_path: Path, group_size: str) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    output = tmp_path / "output"

    result = _run_script(
        "quantize_to_int4.py",
        "--input",
        str(source),
        "--output",
        str(output),
        "--group-size",
        group_size,
    )

    assert result.returncode != 0
    assert "group_size must be positive" in result.stderr
    assert not output.exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config_text", "expected"),
    [
        (None, "config.json is required"),
        ("{", "malformed config.json"),
        ("[]", "config.json must contain a JSON object"),
        ('{"quantization_config": []}', "quantization_config must be a JSON object"),
    ],
)
def test_int4_quantizer_fails_clearly_for_missing_or_malformed_config(
    tmp_path: Path, config_text: str | None, expected: str
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    if config_text is not None:
        (source / "config.json").write_text(config_text)
    save_file({"model.embed_tokens.weight": torch.ones((1, 8))}, source / "model.safetensors")

    result = _run_script(
        "quantize_to_int4.py",
        "--input",
        str(source),
        "--output",
        str(tmp_path / "output"),
    )

    assert result.returncode != 0
    assert expected in result.stderr
    assert not (tmp_path / "output").exists()


@pytest.mark.unit
@pytest.mark.parametrize("precreate_output", [False, True], ids=["absent-output", "empty-output"])
def test_int4_quantizer_rejects_missing_shards_without_publishing(tmp_path: Path, precreate_output: bool) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type": "qwen3_moe"}')
    source_before = _snapshot_files(source)
    output = tmp_path / "output"
    if precreate_output:
        output.mkdir()

    result = _run_script(
        "quantize_to_int4.py",
        "--input",
        str(source),
        "--output",
        str(output),
    )

    assert result.returncode != 0
    assert "no model*.safetensors shards found" in result.stderr
    _assert_failed_quantizer_publish_is_clean(output, existed=precreate_output)
    assert _snapshot_files(source) == source_before


@pytest.mark.unit
def test_int4_quantizer_rejects_output_nested_inside_input(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = source / "output"
    quantizer = _load_conversion_script("quantize_to_int4.py")

    with pytest.raises(SystemExit, match="output directory must not be inside input directory"):
        quantizer._validate_paths(source, output)

    assert not output.exists()
    assert list(source.glob(".output.int4-staging-*")) == []


@pytest.mark.unit
@pytest.mark.parametrize("precreate_output", [False, True], ids=["absent-output", "empty-output"])
def test_int4_quantizer_rejects_checkpoint_without_eligible_expert_weights(
    tmp_path: Path, precreate_output: bool
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type": "qwen3_moe"}')
    save_file({"model.embed_tokens.weight": torch.ones((2, 32))}, source / "model.safetensors")
    source_before = _snapshot_files(source)
    output = tmp_path / "output"
    if precreate_output:
        output.mkdir()

    result = _run_script(
        "quantize_to_int4.py",
        "--input",
        str(source),
        "--output",
        str(output),
    )

    assert result.returncode != 0
    assert "no eligible expert weights" in result.stderr
    _assert_failed_quantizer_publish_is_clean(output, existed=precreate_output)
    assert _snapshot_files(source) == source_before


@pytest.mark.unit
@pytest.mark.parametrize("failure_point", ["open", "save"])
@pytest.mark.parametrize("precreate_output", [False, True], ids=["absent-output", "empty-output"])
def test_int4_quantizer_cleans_staging_after_later_shard_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_point: str,
    precreate_output: bool,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type": "qwen3_moe"}')
    expert_key = "model.layers.0.mlp.experts.0.gate_proj.weight"
    for shard_number in (1, 2):
        save_file(
            {expert_key.replace("experts.0", f"experts.{shard_number}"): torch.ones((2, 32))},
            source / f"model-{shard_number:05d}-of-00002.safetensors",
        )
    source_before = _snapshot_files(source)
    output = tmp_path / "output"
    if precreate_output:
        output.mkdir()

    quantizer = _load_conversion_script("quantize_to_int4.py")
    real_safe_open = quantizer.safe_open
    real_save_file = quantizer.save_file
    save_calls = 0

    def fail_on_second_open(path: str, *args: object, **kwargs: object):
        if Path(path).name == "model-00002-of-00002.safetensors":
            raise RuntimeError("injected later-shard open failure")
        return real_safe_open(path, *args, **kwargs)

    def fail_on_second_save(tensors: object, path: str, *args: object, **kwargs: object):
        nonlocal save_calls
        save_calls += 1
        if save_calls == 2:
            raise RuntimeError("injected later-shard save failure")
        return real_save_file(tensors, path, *args, **kwargs)

    if failure_point == "open":
        monkeypatch.setattr(quantizer, "safe_open", fail_on_second_open)
    else:
        monkeypatch.setattr(quantizer, "save_file", fail_on_second_save)

    with pytest.raises(RuntimeError, match=f"injected later-shard {failure_point} failure"):
        quantizer.main(["--input", str(source), "--output", str(output)])

    _assert_failed_quantizer_publish_is_clean(output, existed=precreate_output)
    assert _snapshot_files(source) == source_before


@pytest.mark.unit
@pytest.mark.parametrize("suffix", ["_packed", "_scale", "_shape"])
def test_hf_weight_has_int4_triplet_detects_weight_companions(suffix: str) -> None:
    state = {f"layer.weight{suffix}": torch.empty(0)}

    assert int4_low_precision.hf_weight_has_int4_triplet("layer.weight", state)


@pytest.mark.unit
def test_int4_direct_builder_rejects_scale_grid_that_disagrees_with_group_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata group size must be checked against every preserved triplet."""
    source_triplet = int4_low_precision._Int4Triplet(
        packed=torch.zeros((2, 16), dtype=torch.int32),
        scale=torch.ones((2, 4), dtype=torch.bfloat16),
        shape=torch.tensor([2, 128], dtype=torch.int32),
    )
    converted_triplet = int4_low_precision._Int4Triplet(
        packed=torch.zeros((2, 16), dtype=torch.int32),
        scale=torch.ones((2, 1), dtype=torch.bfloat16),
        shape=torch.tensor([2, 128], dtype=torch.int32),
    )
    state = {
        "layer.weight_packed": source_triplet.packed,
        "layer.weight_scale": source_triplet.scale,
        "layer.weight_shape": source_triplet.shape,
    }
    mapping = SimpleNamespace(hf_param="layer.weight", tp_size=1)
    task = SimpleNamespace(mapping=mapping, megatron_module=object(), param_name="decoder.layer.weight")
    bridge = SimpleNamespace(
        build_conversion_tasks=lambda hf, model: [task],
        maybe_modify_loaded_hf_weight=lambda *args: pytest.fail("packed source must enter the INT4 validation path"),
    )
    hf_pretrained = SimpleNamespace(state=state)
    monkeypatch.setattr(
        int4_low_precision,
        "_convert_hf_int4_triplet_for_direct_save",
        lambda task, value: converted_triplet,
    )

    with pytest.raises(ValueError, match="group_size=128.*scale|scale.*group_size=128"):
        int4_low_precision.build_int4_direct_model_state_dict(
            bridge,
            hf_pretrained,
            [object()],
            {},
            group_size=128,
            scale_dtype=torch.bfloat16,
        )


@pytest.mark.unit
def test_int4_direct_builder_validates_converted_triplet_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_triplet = int4_low_precision._Int4Triplet(
        packed=torch.zeros((2, 16), dtype=torch.int32),
        scale=torch.ones((2, 1), dtype=torch.bfloat16),
        shape=torch.tensor([2, 128], dtype=torch.int32),
    )
    converted_triplet = int4_low_precision._Int4Triplet(
        packed=torch.zeros((2, 15), dtype=torch.int32),
        scale=torch.ones((2, 1), dtype=torch.bfloat16),
        shape=torch.tensor([2, 128], dtype=torch.int32),
    )
    state = {
        "layer.weight_packed": source_triplet.packed,
        "layer.weight_scale": source_triplet.scale,
        "layer.weight_shape": source_triplet.shape,
    }
    mapping = SimpleNamespace(hf_param="layer.weight", tp_size=1)
    task = SimpleNamespace(mapping=mapping, megatron_module=object(), param_name="decoder.layer.weight")
    bridge = SimpleNamespace(build_conversion_tasks=lambda hf, model: [task])
    monkeypatch.setattr(
        int4_low_precision,
        "_convert_hf_int4_triplet_for_direct_save",
        lambda task, value: converted_triplet,
    )

    with pytest.raises(ValueError, match="decoder.layer.weight.*weight_packed"):
        int4_low_precision.build_int4_direct_model_state_dict(
            bridge,
            SimpleNamespace(state=state),
            [object()],
            {},
            group_size=128,
            scale_dtype=torch.bfloat16,
        )


@pytest.mark.unit
@pytest.mark.parametrize(("mapping_kind", "bad_role"), [("qkv", "k"), ("gated", "up")])
def test_int4_direct_builder_validates_each_composed_source_before_merge(
    monkeypatch: pytest.MonkeyPatch, mapping_kind: str, bad_role: str
) -> None:
    if mapping_kind == "qkv":
        mapping = int4_low_precision.QKVMapping("decoder.qkv.weight", "q.weight", "k.weight", "v.weight")
        roles = {"q": 2, "k": 1, "v": 1}
    else:
        mapping = int4_low_precision.GatedMLPMapping("decoder.fc1.weight", "gate.weight", "up.weight")
        roles = {"gate": 2, "up": 2}

    state = {}
    for role, rows in roles.items():
        packed_width = 15 if role == bad_role else 16
        state[f"{role}.weight_packed"] = torch.zeros((rows, packed_width), dtype=torch.int32)
        state[f"{role}.weight_scale"] = torch.ones((rows, 1), dtype=torch.bfloat16)
        state[f"{role}.weight_shape"] = torch.tensor([rows, 128], dtype=torch.int32)
    task = SimpleNamespace(mapping=mapping, megatron_module=object(), param_name=mapping.megatron_param)
    bridge = SimpleNamespace(build_conversion_tasks=lambda hf, model: [task])
    monkeypatch.setattr(
        int4_low_precision,
        "_convert_hf_int4_triplet_for_direct_save",
        lambda *args: pytest.fail("conversion must not run before all source triplets validate"),
    )

    with pytest.raises(ValueError, match=rf"{bad_role}\.weight.*weight_packed"):
        int4_low_precision.build_int4_direct_model_state_dict(
            bridge,
            SimpleNamespace(state=state),
            [object()],
            {},
            group_size=128,
            scale_dtype=torch.bfloat16,
        )


@pytest.mark.unit
def test_int4_direct_builder_rejects_incomplete_source_triplet() -> None:
    state = {"layer.weight_packed": torch.zeros((2, 16), dtype=torch.int32)}
    mapping = SimpleNamespace(hf_param="layer.weight", tp_size=1)
    task = SimpleNamespace(mapping=mapping, megatron_module=object(), param_name="decoder.layer.weight")
    bridge = SimpleNamespace(build_conversion_tasks=lambda hf, model: [task])

    with pytest.raises(ValueError, match="layer.weight.*weight_scale.*weight_shape"):
        int4_low_precision.build_int4_direct_model_state_dict(
            bridge,
            SimpleNamespace(state=state),
            [object()],
            {},
            group_size=128,
            scale_dtype=torch.bfloat16,
        )


@pytest.mark.unit
def test_int4_direct_builder_requires_positive_group_size() -> None:
    bridge = SimpleNamespace(build_conversion_tasks=lambda hf, model: [])

    with pytest.raises(ValueError, match="group_size must be positive"):
        int4_low_precision.build_int4_direct_model_state_dict(
            bridge,
            SimpleNamespace(state={}),
            [object()],
            {},
            group_size=0,
            scale_dtype=torch.bfloat16,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("builder", "kwargs", "format_name"),
    [
        (
            int4_low_precision.build_int4_direct_model_state_dict,
            {"group_size": 128, "scale_dtype": torch.bfloat16},
            "INT4",
        ),
        (nvfp4_low_precision.build_nvfp4_direct_model_state_dict, {}, "NVFP4"),
    ],
)
@pytest.mark.parametrize(
    ("incomplete_task", "expected_parameter"),
    [
        pytest.param(None, None, id="missing-task"),
        pytest.param(
            SimpleNamespace(
                global_param_name="decoder.layers.0.missing.weight",
                param_name="layers.0.missing.weight",
                megatron_module=None,
            ),
            "decoder.layers.0.missing.weight",
            id="nonlocal-destination",
        ),
    ],
)
def test_single_rank_direct_builder_rejects_incomplete_task_plan_before_source_reads(
    builder,
    kwargs: dict[str, object],
    format_name: str,
    incomplete_task: object,
    expected_parameter: str | None,
) -> None:
    source_key = "model.layers.0.self_attn.o_proj.weight"
    target_key = "decoder.layers.0.self_attention.linear_proj.weight"
    valid_task = SimpleNamespace(
        mapping=int4_low_precision.DirectMapping(target_key, source_key),
        megatron_module=object(),
        param_name=target_key,
    )
    bridge = SimpleNamespace(
        build_conversion_tasks=lambda hf, model: [valid_task, incomplete_task],
        maybe_modify_loaded_hf_weight=lambda *args: pytest.fail(
            "source conversion must not start before task preflight"
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        builder(
            bridge,
            SimpleNamespace(state={source_key: torch.ones((2, 2), dtype=torch.bfloat16)}),
            [object()],
            {},
            **kwargs,
        )

    message = str(exc_info.value)
    assert f"Direct {format_name}" in message
    assert "task index 1" in message
    if expected_parameter is not None:
        assert expected_parameter in message


@pytest.mark.unit
@pytest.mark.parametrize(
    ("builder", "kwargs", "format_name"),
    [
        (
            int4_low_precision.build_int4_direct_model_state_dict,
            {"group_size": 128, "scale_dtype": torch.bfloat16},
            "INT4",
        ),
        (nvfp4_low_precision.build_nvfp4_direct_model_state_dict, {}, "NVFP4"),
        (fp8_low_precision.build_fp8_direct_model_state_dict, {}, "FP8"),
    ],
)
def test_direct_builder_rejects_regular_only_source(
    builder,
    kwargs: dict[str, object],
    format_name: str,
) -> None:
    source_key = "model.layers.0.self_attn.o_proj.weight"
    target_key = "decoder.layers.0.self_attention.linear_proj.weight"
    mapping = int4_low_precision.DirectMapping(target_key, source_key)
    task = SimpleNamespace(mapping=mapping, megatron_module=object(), param_name=target_key)
    hf_pretrained = SimpleNamespace(state={source_key: torch.ones((2, 2), dtype=torch.bfloat16)})
    bridge = SimpleNamespace(
        build_conversion_tasks=lambda hf, model: [task],
        maybe_modify_loaded_hf_weight=lambda hf_param, state: state[hf_param],
    )

    with pytest.raises(RuntimeError, match=rf"no complete {format_name} mappings"):
        builder(bridge, hf_pretrained, [object()], {}, **kwargs)


@pytest.mark.unit
def test_int4_direct_builder_canonicalizes_triplet_to_expert_load_schema() -> None:
    target_key = "decoder.layers.0.mlp.experts.linear_fc2.weight0"
    source_key = "model.layers.0.mlp.experts.0.down_proj.weight"
    source_triplet = int4_low_precision._Int4Triplet(
        packed=torch.full((2, 4), 0x11111111, dtype=torch.int32),
        scale=torch.tensor([[0.5], [0.25]], dtype=torch.float32),
        shape=torch.tensor([2, 32], dtype=torch.int64),
    )
    state = {
        f"{source_key}_packed": source_triplet.packed,
        f"{source_key}_scale": source_triplet.scale,
        f"{source_key}_shape": source_triplet.shape,
    }
    mapping = int4_low_precision.DirectMapping(target_key, source_key)
    task = SimpleNamespace(mapping=mapping, megatron_module=object(), param_name=target_key)
    bridge = SimpleNamespace(build_conversion_tasks=lambda hf, model: [task])

    result = int4_low_precision.build_int4_direct_model_state_dict(
        bridge,
        SimpleNamespace(state=state),
        [object()],
        {},
        group_size=32,
        scale_dtype=torch.float16,
    )

    assert result[f"{target_key}_packed"].dtype == torch.int32
    assert result[f"{target_key}_scale"].dtype == torch.float16
    assert result[f"{target_key}_shape"].dtype == torch.int32
    torch.testing.assert_close(result[f"{target_key}_scale"], source_triplet.scale.to(torch.float16))
    assert result[f"{target_key}_shape"].tolist() == [2, 32]


@pytest.mark.unit
def test_int4_direct_builder_rejects_non_int32_packed_storage() -> None:
    target_key = "decoder.layers.0.mlp.experts.linear_fc2.weight0"
    source_key = "model.layers.0.mlp.experts.0.down_proj.weight"
    state = {
        f"{source_key}_packed": torch.full((2, 4), 0x11, dtype=torch.uint8),
        f"{source_key}_scale": torch.ones((2, 1), dtype=torch.float16),
        f"{source_key}_shape": torch.tensor([2, 32], dtype=torch.int32),
    }
    mapping = int4_low_precision.DirectMapping(target_key, source_key)
    task = SimpleNamespace(mapping=mapping, megatron_module=object(), param_name=target_key)
    bridge = SimpleNamespace(build_conversion_tasks=lambda hf, model: [task])

    with pytest.raises(ValueError, match="weight_packed dtype must be torch.int32"):
        int4_low_precision.build_int4_direct_model_state_dict(
            bridge,
            SimpleNamespace(state=state),
            [object()],
            {},
            group_size=32,
            scale_dtype=torch.float16,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("scale", "shape", "expected"),
    [
        (torch.ones((2, 1), dtype=torch.int32), torch.tensor([2, 32]), "weight_scale must be floating"),
        (torch.tensor([[0.0], [1.0]]), torch.tensor([2, 32]), "weight_scale must be finite and positive"),
        (torch.tensor([[float("nan")], [1.0]]), torch.tensor([2, 32]), "weight_scale must be finite and positive"),
        (torch.ones((2, 1)), torch.tensor([2.0, 32.0]), "weight_shape dtype must be integral"),
    ],
)
def test_int4_triplet_rejects_storage_dtypes_or_invalid_scales(
    scale: torch.Tensor, shape: torch.Tensor, expected: str
) -> None:
    triplet = int4_low_precision._Int4Triplet(
        packed=torch.zeros((2, 4), dtype=torch.int32),
        scale=scale,
        shape=shape,
    )

    with pytest.raises(ValueError, match=expected):
        int4_low_precision._validate_int4_triplet(triplet, group_size=32, key="layer.weight")


@pytest.mark.unit
def test_int4_triplet_rejects_scale_that_underflows_selected_schema_dtype() -> None:
    triplet = int4_low_precision._Int4Triplet(
        packed=torch.zeros((2, 4), dtype=torch.int32),
        scale=torch.full((2, 1), 1e-8, dtype=torch.float32),
        shape=torch.tensor([2, 32], dtype=torch.int32),
    )

    with pytest.raises(ValueError, match="cannot be represented as finite positive torch.float16"):
        int4_low_precision._canonicalize_int4_triplet(
            triplet,
            group_size=32,
            scale_dtype=torch.float16,
            key="layer.weight",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("architecture", "expected"),
    [
        ("KimiK25ForConditionalGeneration", torch.float16),
        ("DeepseekV3ForCausalLM", torch.float16),
        ("DeepSeekV3ForCausalLM", torch.float16),
        ("LlamaForCausalLM", torch.bfloat16),
        ("Qwen3ForCausalLM", torch.bfloat16),
        ("Qwen3MoeForCausalLM", torch.bfloat16),
    ],
)
def test_int4_direct_scale_dtype_matches_runtime_load_scope(architecture: str, expected: torch.dtype) -> None:
    converter = _load_conversion_script("convert_int4_checkpoint_direct.py")
    auto_bridge = SimpleNamespace(_causal_lm_architecture=architecture)

    assert converter._int4_checkpoint_scale_dtype(auto_bridge) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("defect", "expected"),
    [
        ("shape_rank", "weight_shape must be a rank-1 tensor"),
        ("shape_value", "output rows must be a positive integer"),
        ("packed_rows", "weight_packed shape"),
        ("packed_width", "weight_packed shape"),
        ("scale_rows", "weight_scale shape"),
        ("scale_groups", "weight_scale shape"),
    ],
)
def test_int4_triplet_validation_covers_every_logical_grid_dimension(defect: str, expected: str) -> None:
    packed = torch.zeros((2, 16), dtype=torch.int32)
    scale = torch.ones((2, 1), dtype=torch.bfloat16)
    shape = torch.tensor([2, 128], dtype=torch.int32)
    if defect == "shape_rank":
        shape = shape.unsqueeze(0)
    elif defect == "shape_value":
        shape = torch.tensor([0, 128], dtype=torch.int32)
    elif defect == "packed_rows":
        packed = torch.zeros((1, 16), dtype=torch.int32)
    elif defect == "packed_width":
        packed = torch.zeros((2, 15), dtype=torch.int32)
    elif defect == "scale_rows":
        scale = torch.ones((1, 1), dtype=torch.bfloat16)
    elif defect == "scale_groups":
        scale = torch.ones((2, 2), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match=expected):
        int4_low_precision._validate_int4_triplet(
            int4_low_precision._Int4Triplet(packed=packed, scale=scale, shape=shape),
            group_size=128,
            key="model.layers.0.mlp.experts.0.gate_proj.weight",
        )


@pytest.mark.unit
def test_int4_shell_launcher_is_cwd_independent_and_uses_uv(tmp_path: Path) -> None:
    capture = tmp_path / "uv-args.txt"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >"$CAPTURE_PATH"\n')
    uv.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            str(_REPO_ROOT / "scripts" / "orbit" / "quantize_to_int4.sh"),
            "/models/in path",
            "/models/out path",
            "128",
        ],
        cwd=tmp_path,
        env={**os.environ, "PATH": f"{bin_dir}:/usr/bin:/bin", "CAPTURE_PATH": str(capture)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    observed = capture.read_text().splitlines()
    assert observed[:5] == [
        "run",
        "--project",
        str(_REPO_ROOT.resolve()),
        "python",
        str((_CONVERSION_DIR / "quantize_to_int4.py").resolve()),
    ]
    assert observed[-2:] == ["--group-size", "128"]
    follow_up = result.stdout.split("Next: convert to Megatron format:", 1)[1]
    assert f'uv run --project "{_REPO_ROOT.resolve()}" python' in follow_up
    assert f'python "{(_CONVERSION_DIR / "convert_int4_checkpoint_direct.py").resolve()}"' in follow_up
    assert '--hf-model-path "/models/out path"' in follow_up

    without_group_size = subprocess.run(
        ["bash", str(_REPO_ROOT / "scripts" / "orbit" / "quantize_to_int4.sh"), "/models/in", "/models/out"],
        cwd=tmp_path,
        env={**os.environ, "PATH": f"{bin_dir}:/usr/bin:/bin", "CAPTURE_PATH": str(capture)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert without_group_size.returncode == 0, without_group_size.stderr
    assert "--group-size" not in capture.read_text().splitlines()


@pytest.mark.unit
def test_megatron_submodule_uses_impossible_radixark_fork() -> None:
    gitmodules = (_REPO_ROOT / ".gitmodules").read_text()

    assert "url = https://github.com/radixark/Megatron-LM.git" in gitmodules
    assert "url = https://github.com/NVIDIA/Megatron-LM.git" not in gitmodules


@pytest.mark.unit
@pytest.mark.parametrize(
    "script_name",
    [
        "convert_int4_checkpoint_direct.py",
        "convert_fp8_checkpoint_direct.py",
        "convert_nvfp4_checkpoint_direct.py",
    ],
)
@pytest.mark.parametrize("architecture", ["Qwen3MoeForCausalLM", type("Qwen3MoeForCausalLM", (), {})])
def test_qwen3_moe_direct_converter_applies_orbit_provider_settings(
    monkeypatch: pytest.MonkeyPatch, script_name: str, architecture: object
) -> None:
    """Exercise each converter's real main-path provider before model construction."""
    module = _load_conversion_script(script_name)
    hf_config = SimpleNamespace(
        decoder_sparse_step=2,
        mlp_only_layers=[3],
        num_experts=8,
        num_hidden_layers=6,
        quantization_config={"group_size": 128},
    )
    hf_pretrained = SimpleNamespace(config=hf_config, trust_remote_code=False)
    auto_bridge = SimpleNamespace(
        hf_pretrained=hf_pretrained,
        _causal_lm_architecture=architecture,
        _model_bridge=object(),
    )

    class ProviderChecked(RuntimeError):
        pass

    class Provider:
        def finalize(self):
            assert self.moe_router_dtype == "fp32"
            assert self.moe_layer_freq == [0, 1, 0, 0, 0, 1]

        def provide_distributed_model(self, **kwargs):
            assert self.moe_router_dtype == "fp32"
            assert self.moe_layer_freq == [0, 1, 0, 0, 0, 1]
            raise ProviderChecked

    provider = Provider()
    args = SimpleNamespace(hf_model_path="hf", megatron_path="out", group_size=None)
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "build_single_rank_meta_provider", lambda *a, **kw: (auto_bridge, provider))
    monkeypatch.setattr(module, "patch_meta_init_for_te_modules", lambda: None)
    monkeypatch.setattr(module, "temporary_distributed_context", lambda **kwargs: nullcontext())
    monkeypatch.setattr(module, "keep_meta_model_unmaterialized", nullcontext)
    if script_name == "convert_nvfp4_checkpoint_direct.py":
        monkeypatch.setattr(module, "is_nvfp4_source", lambda config: True)

    with pytest.raises(ProviderChecked):
        module.main()


@pytest.mark.unit
def test_fp8_direct_converter_does_not_apply_qwen3_moe_settings_to_other_architectures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_conversion_script("convert_fp8_checkpoint_direct.py")
    auto_bridge = SimpleNamespace(
        hf_pretrained=SimpleNamespace(config=SimpleNamespace()),
        _causal_lm_architecture="Qwen3ForCausalLM",
        _model_bridge=object(),
    )

    class ProviderChecked(RuntimeError):
        pass

    class Provider:
        def provide_distributed_model(self, **kwargs):
            raise ProviderChecked

    monkeypatch.setattr(module, "parse_args", lambda: SimpleNamespace(hf_model_path="hf", megatron_path="out"))
    monkeypatch.setattr(module, "build_single_rank_meta_provider", lambda *args: (auto_bridge, Provider()))
    monkeypatch.setattr(
        module,
        "apply_qwen3_moe_orbit_provider_settings",
        lambda *args: pytest.fail("dense Qwen3 must not receive Qwen3-MoE-only provider settings"),
    )
    monkeypatch.setattr(module, "patch_meta_init_for_te_modules", lambda: None)
    monkeypatch.setattr(module, "temporary_distributed_context", lambda **kwargs: nullcontext())
    monkeypatch.setattr(module, "keep_meta_model_unmaterialized", nullcontext)

    with pytest.raises(ProviderChecked):
        module.main()


@pytest.mark.unit
@pytest.mark.parametrize(
    "script_name",
    [
        "convert_int4_checkpoint_direct.py",
        "convert_nvfp4_checkpoint_direct.py",
        "convert_fp8_checkpoint_direct.py",
    ],
)
def test_direct_converter_does_not_save_regular_only_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    script_name: str,
) -> None:
    module = _load_conversion_script(script_name)
    source_key = "model.layers.0.self_attn.o_proj.weight"
    target_key = "decoder.layers.0.self_attention.linear_proj.weight"
    mapping = int4_low_precision.DirectMapping(target_key, source_key)
    task = SimpleNamespace(mapping=mapping, megatron_module=object(), param_name=target_key)
    hf_pretrained = SimpleNamespace(
        config=SimpleNamespace(quantization_config={"group_size": 128}),
        state={source_key: torch.ones((2, 2), dtype=torch.bfloat16)},
        trust_remote_code=False,
    )

    class Bridge:
        def build_conversion_tasks(self, hf, model):
            return [task]

        def maybe_modify_loaded_hf_weight(self, hf_param, state):
            return state[hf_param]

    bridge = Bridge()
    auto_bridge = SimpleNamespace(
        hf_pretrained=hf_pretrained,
        _causal_lm_architecture="LlamaForCausalLM",
        _model_bridge=bridge,
    )

    class MetaModel:
        def sharded_state_dict(self, **kwargs):
            return {}

    class Provider:
        def provide_distributed_model(self, **kwargs):
            return [MetaModel()]

    args = SimpleNamespace(
        hf_model_path="hf",
        megatron_path=str(tmp_path / "out"),
        group_size=None,
        debug_layer_range=None,
    )
    save_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.delenv("MEGATRON_BRIDGE_DIRECT_USE_SPILL", raising=False)
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "build_single_rank_meta_provider", lambda *a, **kw: (auto_bridge, Provider()))
    monkeypatch.setattr(module, "patch_meta_init_for_te_modules", lambda: None)
    monkeypatch.setattr(module, "temporary_distributed_context", lambda **kwargs: nullcontext())
    monkeypatch.setattr(module, "keep_meta_model_unmaterialized", nullcontext)
    monkeypatch.setattr(module, "get_pg_collection", lambda model: SimpleNamespace(dp_cp=None))
    monkeypatch.setattr(module, "_save_direct_checkpoint", lambda *a, **kw: save_calls.append((a, kw)))

    if script_name == "convert_int4_checkpoint_direct.py":
        monkeypatch.setattr(module, "_select_int4_bridge", lambda actual: bridge)
        monkeypatch.setattr(module, "_temporary_safetensors_reader", nullcontext)
    elif script_name == "convert_nvfp4_checkpoint_direct.py":
        monkeypatch.setattr(module, "_select_nvfp4_bridge", lambda actual: bridge)
        monkeypatch.setattr(module, "is_nvfp4_source", lambda config: True)
        monkeypatch.setattr(module, "apply_modelopt_nvfp4_to_meta_model", lambda *a, **kw: None)
    else:
        monkeypatch.setattr(module, "apply_modelopt_fp8_to_meta_model", lambda *a, **kw: None)
        monkeypatch.setattr(module, "_maybe_create_save_progress_monitor", lambda path: None)

    with pytest.raises(RuntimeError, match=r"no complete (?:INT4|NVFP4|FP8) mappings"):
        module.main()

    assert save_calls == []


@pytest.mark.unit
def test_fp8_direct_converter_targets_only_preflighted_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_conversion_script("convert_fp8_checkpoint_direct.py")
    state = object()
    expected_tasks = [object(), object()]
    target_modules = {"decoder.layers.0.self_attention.linear_proj"}
    expected_plan = SimpleNamespace(module_names=target_modules, fp8_task_ids={id(expected_tasks[0])})
    events = []

    class BuildReached(RuntimeError):
        pass

    class Bridge:
        def build_conversion_tasks(self, hf, model):
            events.append("tasks")
            return expected_tasks

    bridge = Bridge()
    auto_bridge = SimpleNamespace(
        hf_pretrained=SimpleNamespace(config=SimpleNamespace(), state=state),
        _causal_lm_architecture="Qwen3ForCausalLM",
        _model_bridge=bridge,
    )

    class MetaModel:
        def sharded_state_dict(self, **kwargs):
            events.append("template")
            return {}

    class Provider:
        def provide_distributed_model(self, **kwargs):
            return [MetaModel()]

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(hf_model_path="hf", megatron_path="out"),
    )
    monkeypatch.setattr(module, "build_single_rank_meta_provider", lambda *a, **kw: (auto_bridge, Provider()))
    monkeypatch.setattr(module, "patch_meta_init_for_te_modules", lambda: None)
    monkeypatch.setattr(module, "temporary_distributed_context", lambda **kwargs: nullcontext())
    monkeypatch.setattr(module, "keep_meta_model_unmaterialized", nullcontext)
    monkeypatch.setattr(module, "get_pg_collection", lambda model: SimpleNamespace(dp_cp=None))

    def preflight(tasks, actual_state, **kwargs):
        assert tasks is expected_tasks
        assert actual_state is state
        assert kwargs == {"require_complete": True}
        events.append("preflight")
        return expected_plan

    def apply(module_arg, *, module_names):
        assert module_names is target_modules
        events.append("apply")

    def build(bridge_arg, hf_pretrained, meta_model, model_template, *, conversion_tasks, fp8_plan):
        assert bridge_arg is bridge
        assert conversion_tasks is expected_tasks
        assert fp8_plan is expected_plan
        events.append("build")
        raise BuildReached

    monkeypatch.setattr(module, "preflight_fp8_conversion_tasks", preflight)
    monkeypatch.setattr(module, "apply_modelopt_fp8_to_meta_model", apply)
    monkeypatch.setattr(module, "build_fp8_direct_model_state_dict", build)

    with pytest.raises(BuildReached):
        module.main()

    assert events == ["tasks", "preflight", "apply", "template", "build"]


@pytest.mark.unit
def test_nvfp4_direct_converter_keeps_selected_bridge_local(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Selecting Kimi's bridge must not replace AutoBridge's class property."""
    module = _load_conversion_script("convert_nvfp4_checkpoint_direct.py")

    class FakeAutoBridge:
        def __init__(self, bridge):
            self._instance_bridge = bridge
            self._causal_lm_architecture = "KimiK25ForConditionalGeneration"
            self.hf_pretrained = SimpleNamespace(
                config=SimpleNamespace(),
                state={},
                trust_remote_code=False,
            )

        @property
        def _model_bridge(self):
            return self._instance_bridge

    first_bridge = object()
    later_bridge = object()
    auto_bridge = FakeAutoBridge(first_bridge)
    later_auto_bridge = FakeAutoBridge(later_bridge)

    class SelectedBridgeReached(RuntimeError):
        pass

    class SelectedBridge:
        def build_conversion_tasks(self, hf_pretrained, megatron_model):
            return []

    class MetaModel:
        def sharded_state_dict(self, **kwargs):
            return {}

    class Provider:
        def provide_distributed_model(self, **kwargs):
            return [MetaModel()]

    selected_bridge = SelectedBridge()
    monkeypatch.delenv("MEGATRON_BRIDGE_DIRECT_USE_SPILL", raising=False)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            hf_model_path="hf",
            megatron_path=str(tmp_path / "out"),
            debug_layer_range=None,
        ),
    )
    monkeypatch.setattr(module, "build_single_rank_meta_provider", lambda *args, **kwargs: (auto_bridge, Provider()))
    monkeypatch.setattr(module, "_select_nvfp4_bridge", lambda actual: selected_bridge)
    monkeypatch.setattr(module, "is_nvfp4_source", lambda config: True)
    monkeypatch.setattr(module, "patch_meta_init_for_te_modules", lambda: None)
    monkeypatch.setattr(module, "temporary_distributed_context", lambda **kwargs: nullcontext())
    monkeypatch.setattr(module, "keep_meta_model_unmaterialized", nullcontext)
    monkeypatch.setattr(module, "collect_nvfp4_target_module_names", lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "apply_modelopt_nvfp4_to_meta_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "get_pg_collection", lambda model: SimpleNamespace(dp_cp=None))

    def build_state(bridge, *args, **kwargs):
        assert bridge is selected_bridge
        assert kwargs["conversion_tasks"] == []
        raise SelectedBridgeReached

    monkeypatch.setattr(module, "build_nvfp4_direct_model_state_dict", build_state)

    with pytest.raises(SelectedBridgeReached):
        module.main()

    assert auto_bridge._model_bridge is first_bridge
    assert later_auto_bridge._model_bridge is later_bridge


@pytest.mark.unit
@pytest.mark.parametrize(
    "script_name",
    [
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
        "convert_fp4_checkpoint_direct.py",
        "convert_fp8_checkpoint.py",
        "convert_int4_checkpoint.py",
        "dump_nvfp4_meta_keys.py",
    ],
)
def test_superseded_conversion_entrypoints_are_removed(obsolete_name: str) -> None:
    assert not (_CONVERSION_DIR / obsolete_name).exists()
