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
import stat
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from megatron.bridge.orbit.low_precision.common import atomic_direct_checkpoint_directory


_REPO_ROOT = Path(__file__).parents[3]
_CONVERSION_DIR = _REPO_ROOT / "scripts" / "orbit" / "conversion"
_CHECKPOINT_ARTIFACT = Path("iter_0000000") / "metadata.json"


def _load_conversion_script(script_name: str):
    path = _CONVERSION_DIR / script_name
    spec = importlib.util.spec_from_file_location(f"{path.stem}_transaction_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_checkpoint_artifact(root: Path) -> None:
    artifact = root / _CHECKPOINT_ARTIFACT
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}")


def _assert_no_staging_roots(destination: Path) -> None:
    assert list(destination.parent.glob(f".{destination.name}.staging-*")) == []


@pytest.mark.unit
def test_direct_checkpoint_transaction_rejects_existing_destination_without_mutation(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint"
    destination.mkdir()
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("keep me")

    with pytest.raises(FileExistsError, match="already exists"):
        with atomic_direct_checkpoint_directory(destination):
            pytest.fail("transaction must reject the destination before yielding")

    assert sentinel.read_text() == "keep me"
    _assert_no_staging_roots(destination)


@pytest.mark.unit
def test_direct_checkpoint_transaction_rejects_dangling_symlink_destination(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint"
    destination.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    with pytest.raises(FileExistsError, match="symlink"):
        with atomic_direct_checkpoint_directory(destination):
            pytest.fail("transaction must reject a dangling destination symlink")

    assert destination.is_symlink()
    _assert_no_staging_roots(destination)


@pytest.mark.unit
def test_direct_checkpoint_transaction_cleans_partial_dcp_write(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint"

    with pytest.raises(RuntimeError, match="injected DCP failure"):
        with atomic_direct_checkpoint_directory(destination) as staging:
            partial = staging / "iter_0000000" / "__0_0.distcp"
            partial.parent.mkdir(parents=True)
            partial.write_bytes(b"partial checkpoint")
            raise RuntimeError("injected DCP failure")

    assert not destination.exists()
    _assert_no_staging_roots(destination)


@pytest.mark.unit
def test_direct_checkpoint_transaction_cleans_after_tokenizer_failure(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint"

    with pytest.raises(RuntimeError, match="injected tokenizer failure"):
        with atomic_direct_checkpoint_directory(destination) as staging:
            _write_checkpoint_artifact(staging)
            (staging / "iter_0000000" / "tokenizer.json").write_text("partial tokenizer")
            raise RuntimeError("injected tokenizer failure")

    assert not destination.exists()
    _assert_no_staging_roots(destination)


@pytest.mark.unit
def test_direct_checkpoint_transaction_requires_dcp_metadata_before_publish(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint"

    with pytest.raises(RuntimeError, match=r"iter_0000000/metadata\.json"):
        with atomic_direct_checkpoint_directory(destination) as staging:
            (staging / "iter_0000000").mkdir()
            (staging / "iter_0000000" / "tokenizer.json").write_text("complete tokenizer")

    assert not destination.exists()
    _assert_no_staging_roots(destination)


@pytest.mark.unit
def test_direct_checkpoint_transaction_publishes_complete_tree_by_rename(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint"
    observed_staging: Path | None = None
    staging_inode: int | None = None

    with atomic_direct_checkpoint_directory(destination) as staging:
        observed_staging = staging
        staging_inode = staging.stat().st_ino
        assert staging.parent == destination.parent
        assert staging.stat().st_mode & 0o077 == 0
        assert staging != destination
        assert not destination.exists()
        _write_checkpoint_artifact(staging)
        (staging / "iter_0000000" / "tokenizer.json").write_text("complete tokenizer")

    assert observed_staging is not None
    assert staging_inode is not None
    assert not observed_staging.exists()
    assert destination.stat().st_ino == staging_inode
    assert (destination / _CHECKPOINT_ARTIFACT).read_text() == "{}"
    assert (destination / "iter_0000000" / "tokenizer.json").read_text() == "complete tokenizer"
    _assert_no_staging_roots(destination)


@pytest.mark.unit
def test_direct_checkpoint_transaction_publishes_with_normal_directory_mode(tmp_path: Path) -> None:
    expected_directory = tmp_path / "ordinary-directory"
    expected_directory.mkdir()
    expected_mode = stat.S_IMODE(expected_directory.stat().st_mode)
    destination = tmp_path / "checkpoint"

    with atomic_direct_checkpoint_directory(destination) as staging:
        assert stat.S_IMODE(staging.stat().st_mode) == 0o700
        _write_checkpoint_artifact(staging)

    assert stat.S_IMODE(destination.stat().st_mode) == expected_mode


@pytest.mark.unit
def test_direct_checkpoint_transaction_does_not_replace_destination_created_before_publish(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint"

    with pytest.raises(FileExistsError, match="before checkpoint publication"):
        with atomic_direct_checkpoint_directory(destination) as staging:
            _write_checkpoint_artifact(staging)
            destination.mkdir()
            (destination / "sentinel.txt").write_text("keep me")

    assert (destination / "sentinel.txt").read_text() == "keep me"
    _assert_no_staging_roots(destination)


@pytest.mark.unit
@pytest.mark.parametrize(
    "script_name",
    [
        "convert_int4_checkpoint_direct.py",
        "convert_nvfp4_checkpoint_direct.py",
        "convert_fp8_checkpoint_direct.py",
    ],
)
def test_direct_checkpoint_entrypoint_builds_and_saves_inside_shared_transaction(
    script_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_conversion_script(script_name)
    destination = tmp_path / "checkpoint"
    staging = tmp_path / "private-staging"
    events: list[str] = []

    @contextmanager
    def transaction(actual_destination: str | Path):
        assert Path(actual_destination) == destination
        assert not destination.exists()
        staging.mkdir()
        events.append("transaction-enter")
        try:
            yield staging
        finally:
            events.append("transaction-exit")

    class MetaModel:
        def sharded_state_dict(self, **kwargs: object) -> dict[str, object]:
            return {}

    class Provider:
        def finalize(self) -> None:
            pass

        def provide_distributed_model(self, **kwargs: object) -> list[MetaModel]:
            return [MetaModel()]

    hf_pretrained = SimpleNamespace(
        config=SimpleNamespace(quantization_config={"group_size": 32}),
        state={},
        trust_remote_code=False,
    )
    auto_bridge = SimpleNamespace(
        hf_pretrained=hf_pretrained,
        _causal_lm_architecture="LlamaForCausalLM",
        _model_bridge=SimpleNamespace(build_conversion_tasks=lambda hf, model: []),
    )
    args = SimpleNamespace(
        hf_model_path="hf",
        megatron_path=str(destination),
        group_size=None,
        debug_layer_range=None,
    )

    monkeypatch.delenv("MEGATRON_BRIDGE_DIRECT_USE_SPILL", raising=False)
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "build_single_rank_meta_provider", lambda *args, **kwargs: (auto_bridge, Provider()))
    monkeypatch.setattr(module, "patch_meta_init_for_te_modules", lambda: None)
    monkeypatch.setattr(module, "temporary_distributed_context", lambda **kwargs: nullcontext())
    monkeypatch.setattr(module, "keep_meta_model_unmaterialized", nullcontext)
    monkeypatch.setattr(module, "get_pg_collection", lambda model: SimpleNamespace(dp_cp=None))
    monkeypatch.setattr(module, "atomic_direct_checkpoint_directory", transaction)

    def build_state(*args: object, **kwargs: object) -> dict[str, object]:
        assert events == ["transaction-enter"]
        events.append("build")
        return {}

    def save_state(provider: object, path: str | Path, model_state: object, **kwargs: object) -> None:
        assert Path(path) == staging
        assert events == ["transaction-enter", "build"]
        events.append("save")

    if script_name == "convert_int4_checkpoint_direct.py":
        monkeypatch.setattr(module, "_select_int4_bridge", lambda bridge: object())
        monkeypatch.setattr(module, "_temporary_safetensors_reader", nullcontext)
        monkeypatch.setattr(module, "build_int4_direct_model_state_dict", build_state)
    elif script_name == "convert_nvfp4_checkpoint_direct.py":
        selected_bridge = SimpleNamespace(build_conversion_tasks=lambda hf, model: [])
        monkeypatch.setattr(module, "_select_nvfp4_bridge", lambda bridge: selected_bridge)
        monkeypatch.setattr(module, "is_nvfp4_source", lambda config: True)
        monkeypatch.setattr(module, "collect_nvfp4_target_module_names", lambda *args, **kwargs: [])
        monkeypatch.setattr(module, "apply_modelopt_nvfp4_to_meta_model", lambda *args, **kwargs: None)
        monkeypatch.setattr(module, "build_nvfp4_direct_model_state_dict", build_state)
    else:
        monkeypatch.setattr(
            module,
            "preflight_fp8_conversion_tasks",
            lambda *args, **kwargs: SimpleNamespace(fp8_task_ids=frozenset({1}), module_names=frozenset()),
        )
        monkeypatch.setattr(module, "apply_modelopt_fp8_to_meta_model", lambda *args, **kwargs: None)
        monkeypatch.setattr(module, "build_fp8_direct_model_state_dict", build_state)
        monkeypatch.setattr(module, "_maybe_create_save_progress_monitor", lambda path: None)

    monkeypatch.setattr(module, "_save_direct_checkpoint", save_state)

    assert module.main() == 0
    assert events == ["transaction-enter", "build", "save", "transaction-exit"]
