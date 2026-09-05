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
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


_REPO_ROOT = Path(__file__).parents[3]
_COMMON_PATH = _REPO_ROOT / "src" / "megatron" / "bridge" / "orbit" / "low_precision" / "common.py"
_CONVERTER_PATH = _REPO_ROOT / "scripts" / "orbit" / "conversion" / "convert_int4_checkpoint_direct.py"


def _install_module(monkeypatch: pytest.MonkeyPatch, name: str, **attributes: object) -> ModuleType:
    parts = name.split(".")
    for index in range(1, len(parts)):
        package_name = ".".join(parts[:index])
        if package_name not in sys.modules:
            package = ModuleType(package_name)
            package.__path__ = []
            monkeypatch.setitem(sys.modules, package_name, package)

    module = ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_int4_converter(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    placeholder = type("Placeholder", (), {})
    _install_module(
        monkeypatch,
        "megatron.core.dist_checkpointing.mapping",
        ShardedTensor=placeholder,
        ShardedTensorFactory=placeholder,
    )
    common_name = "megatron.bridge.orbit.low_precision.common"
    common = _load_module(common_name, _COMMON_PATH)
    monkeypatch.setitem(sys.modules, common_name, common)

    _install_module(monkeypatch, "megatron.core.optimizer", OptimizerConfig=placeholder)
    _install_module(monkeypatch, "megatron.bridge", AutoBridge=placeholder)
    stubs = {
        "megatron.bridge.models.kimi_vl.kimi_k25_vl_bridge": {"KimiK25VLBridge": placeholder},
        "megatron.bridge.orbit.conversion.compressed_tensors_int4": {"int4_bridge_for": lambda value: value},
        "megatron.bridge.orbit.low_precision.int4": {"build_int4_direct_model_state_dict": lambda *args: {}},
        "megatron.bridge.orbit.model_bridges.deepseek_v3_int4_bridge": {"DeepSeekV3INT4Bridge": placeholder},
        "megatron.bridge.orbit.model_bridges.llama_int4_bridge": {"LlamaINT4Bridge": placeholder},
        "megatron.bridge.orbit.model_bridges.qwen3_int4_bridge": {
            "Qwen3INT4Bridge": placeholder,
            "Qwen3MoEINT4Bridge": placeholder,
        },
        "megatron.bridge.orbit.model_bridges.qwen3_moe_provider_ext": {
            "apply_qwen3_moe_orbit_provider_settings": lambda *args: None
        },
        "megatron.bridge.training.checkpointing": {
            "get_checkpoint_name": lambda *args: "",
            "save_checkpoint": lambda *args, **kwargs: None,
            "save_tokenizer_assets": lambda *args: None,
        },
        "megatron.bridge.training.config": {
            "CheckpointConfig": placeholder,
            "ConfigContainer": placeholder,
            "LoggerConfig": placeholder,
        },
        "megatron.bridge.training.model_load_save": {"temporary_distributed_context": lambda *args: None},
        "megatron.bridge.training.state": {"GlobalState": placeholder},
        "megatron.bridge.training.tokenizers.config": {"TokenizerConfig": placeholder},
        "megatron.bridge.training.tokenizers.tokenizer": {"build_tokenizer": lambda *args: None},
        "megatron.bridge.training.utils.pg_utils": {"get_pg_collection": lambda *args: None},
    }
    for name, attributes in stubs.items():
        _install_module(monkeypatch, name, **attributes)

    return _load_module("convert_int4_checkpoint_direct_portability_test", _CONVERTER_PATH)


@pytest.mark.unit
def test_residency_capped_spill_works_without_posix_fadvise(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    converter = _load_int4_converter(monkeypatch)
    converter._ResidencyCappedSpillManager._libc = SimpleNamespace(madvise=lambda *args: 0)
    converter.os = SimpleNamespace(sysconf=os.sysconf, open=os.open, close=os.close, O_RDONLY=os.O_RDONLY)
    manager = converter._ResidencyCappedSpillManager(tmp_path)

    try:
        spilled = manager.spill_tensor("weight", torch.arange(8, dtype=torch.float32))
        assert torch.equal(spilled, torch.arange(8, dtype=torch.float32))
    finally:
        manager.cleanup()


@pytest.mark.unit
def test_residency_capped_spill_closes_advice_fd_when_posix_fadvise_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    converter = _load_int4_converter(monkeypatch)
    converter._ResidencyCappedSpillManager._libc = SimpleNamespace(madvise=lambda *args: 0)
    opened: list[int] = []
    closed: list[int] = []

    def tracking_open(path: Path, flags: int) -> int:
        fd = os.open(path, flags)
        opened.append(fd)
        return fd

    def tracking_close(fd: int) -> None:
        os.close(fd)
        closed.append(fd)

    def failing_posix_fadvise(*args: object) -> None:
        raise OSError("injected fadvise failure")

    converter.os = SimpleNamespace(
        sysconf=os.sysconf,
        open=tracking_open,
        close=tracking_close,
        posix_fadvise=failing_posix_fadvise,
        POSIX_FADV_DONTNEED=4,
        O_RDONLY=os.O_RDONLY,
    )
    manager = converter._ResidencyCappedSpillManager(tmp_path)

    try:
        with pytest.raises(OSError, match="injected fadvise failure"):
            manager.spill_tensor("weight", torch.arange(8, dtype=torch.float32))
        assert opened == closed
    finally:
        for fd in set(opened).difference(closed):
            os.close(fd)
        manager.cleanup()
