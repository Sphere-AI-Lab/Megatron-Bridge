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

import importlib
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest
import torch


_REPO_ROOT = Path(__file__).parents[3]
_QOFT_COMMON_PATH = _REPO_ROOT / "scripts" / "orbit" / "models" / "_qoft_common.py"
_MISSING = object()


def _load_qoft_common():
    spec = importlib.util.spec_from_file_location("qoft_common_checkpoint_resume_under_test", _QOFT_COMMON_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _restore_attribute(module, name: str, value) -> None:
    if value is _MISSING:
        if hasattr(module, name):
            delattr(module, name)
    else:
        setattr(module, name, value)


@contextmanager
def _installed_quantized_loader(
    monkeypatch: pytest.MonkeyPatch,
    quant: str,
    *,
    stub_nvfp4_dense_registration: bool = True,
) -> Iterator[object]:
    qcommon = _load_qoft_common()
    checkpointing = importlib.import_module("megatron.bridge.training.checkpointing")
    torch_strategy = importlib.import_module("megatron.core.dist_checkpointing.strategies.torch")
    moe_experts = importlib.import_module("megatron.core.transformer.moe.experts")
    modelopt_checkpoint = importlib.import_module("megatron.bridge.orbit.training.modelopt_checkpoint")
    marker = f"_qoft_{quant}_checkpoint_patches_installed"
    snapshots = [
        (checkpointing, "_generate_model_state_dict", checkpointing._generate_model_state_dict),
        (checkpointing, "_load_model_state_dict", checkpointing._load_model_state_dict),
        (checkpointing, marker, getattr(checkpointing, marker, _MISSING)),
        (torch_strategy, "mcore_to_pyt_state_dict", torch_strategy.mcore_to_pyt_state_dict),
        (moe_experts, "apply_swiglu_sharded_factory", moe_experts.apply_swiglu_sharded_factory),
        (
            modelopt_checkpoint,
            "_maybe_restore_modelopt_state_for_sharded_load",
            modelopt_checkpoint._maybe_restore_modelopt_state_for_sharded_load,
        ),
    ]
    if hasattr(checkpointing, marker):
        delattr(checkpointing, marker)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    if quant == "int4":
        int4_utils = importlib.import_module("megatron.bridge.orbit.quant.int4_utils")
        monkeypatch.setattr(int4_utils, "register_int4_buffers_after_load", lambda model, state: None)
        qcommon.install_int4_checkpoint_load_patches(
            scope="experts",
            group_size=32,
            arch_label="test",
        )
    elif quant == "nvfp4":
        nvfp4_dense = importlib.import_module("megatron.bridge.orbit.low_precision.nvfp4")
        nvfp4_expert = importlib.import_module("megatron.bridge.orbit.quant.nvfp4_utils")
        if stub_nvfp4_dense_registration:
            monkeypatch.setattr(nvfp4_dense, "register_nvfp4_buffers_after_load_dense", lambda model, state: None)
        monkeypatch.setattr(nvfp4_expert, "register_nvfp4_buffers_after_load", lambda model, state: None)
        qcommon.install_nvfp4_checkpoint_load_patches(
            pretrained_checkpoint="unused",
            arch_label="test",
        )
    elif quant == "fp8":
        fp8_utils = importlib.import_module("megatron.bridge.orbit.quant.fp8_utils")
        monkeypatch.setattr(fp8_utils, "register_fp8_scale_inv_buffers_after_load", lambda model, state: None)
        qcommon.install_fp8_checkpoint_load_patches(
            pretrained_checkpoint="unused",
            arch_label="test",
        )
    else:  # pragma: no cover - test helper guard
        raise AssertionError(f"unsupported quant mode: {quant}")

    try:
        yield checkpointing._load_model_state_dict
    finally:
        for module, name, value in reversed(snapshots):
            _restore_attribute(module, name, value)


class _TinyQOFTModel(torch.nn.Module):
    def __init__(self, *, meta_base: bool = False) -> None:
        super().__init__()
        device = "meta" if meta_base else "cpu"
        self.base = torch.nn.Parameter(torch.full((2,), -1.0, device=device), requires_grad=False)
        self.oft_r = torch.nn.Parameter(torch.zeros(2))


class _TinyDenseNVFP4Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = torch.nn.Linear(128, 4, bias=False, dtype=torch.bfloat16)
        self.dense.weight.requires_grad_(False)


@pytest.mark.unit
def test_nvfp4_dense_payload_classification_includes_split_weight_family() -> None:
    qcommon = _load_qoft_common()

    assert qcommon._nvfp4_dense_payload_keys(
        [
            "dense.weight_w",
            "dense.weight_v",
            "dense.weight_quantizer._scale",
            "dense.weight_quantizer._double_scale",
            "oft_r",
        ]
    ) == frozenset(
        {
            "dense.weight",
            "dense.weight_w",
            "dense.weight_v",
            "dense.weight_quantizer._scale",
            "dense.weight_quantizer._double_scale",
        }
    )


@pytest.mark.unit
@pytest.mark.parametrize("quant", ["int4", "nvfp4", "fp8"])
def test_adapter_only_resume_allows_frozen_base_omission_and_preserves_optimizer_identity(
    monkeypatch: pytest.MonkeyPatch, quant: str
) -> None:
    model = _TinyQOFTModel()
    original_adapter = model.oft_r
    optimizer = torch.optim.SGD([model.oft_r], lr=0.1)

    with _installed_quantized_loader(monkeypatch, quant) as load_model_state_dict:
        load_model_state_dict(
            model,
            {"oft_r": torch.tensor([3.0, 4.0])},
            strict=False,
            adapter_only=True,
        )

    assert model.oft_r is original_adapter
    assert optimizer.param_groups[0]["params"][0] is original_adapter
    torch.testing.assert_close(model.oft_r, torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(model.base, torch.full((2,), -1.0))


@pytest.mark.unit
@pytest.mark.parametrize("quant", ["int4", "nvfp4", "fp8"])
def test_adapter_only_resume_rejects_missing_trainable_adapter(monkeypatch: pytest.MonkeyPatch, quant: str) -> None:
    model = _TinyQOFTModel()

    with _installed_quantized_loader(monkeypatch, quant) as load_model_state_dict:
        with pytest.raises(RuntimeError, match="missing trainable adapter.*oft_r"):
            load_model_state_dict(model, {}, strict=False, adapter_only=True)


@pytest.mark.unit
@pytest.mark.parametrize("quant", ["int4", "nvfp4", "fp8"])
def test_adapter_only_resume_rejects_unexpected_payload(monkeypatch: pytest.MonkeyPatch, quant: str) -> None:
    model = _TinyQOFTModel()
    state_dict = {
        "oft_r": torch.ones(2),
        "not_a_model_key": torch.ones(1),
    }

    with _installed_quantized_loader(monkeypatch, quant) as load_model_state_dict:
        with pytest.raises(RuntimeError, match="unexpected.*not_a_model_key"):
            load_model_state_dict(model, state_dict, strict=False, adapter_only=True)
    torch.testing.assert_close(model.oft_r, torch.zeros(2))


@pytest.mark.unit
@pytest.mark.parametrize("quant", ["int4", "nvfp4", "fp8"])
def test_adapter_only_resume_rejects_known_frozen_base_payload(monkeypatch: pytest.MonkeyPatch, quant: str) -> None:
    model = _TinyQOFTModel()
    state_dict = {
        "base": torch.full((2,), 9.0),
        "oft_r": torch.ones(2),
    }

    with _installed_quantized_loader(monkeypatch, quant) as load_model_state_dict:
        with pytest.raises(RuntimeError, match="frozen base.*base"):
            load_model_state_dict(model, state_dict, strict=False, adapter_only=True)
    torch.testing.assert_close(model.base, torch.full((2,), -1.0))
    torch.testing.assert_close(model.oft_r, torch.zeros(2))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("quant", "quantized_key"),
    [
        ("int4", "decoder.layers.0.self_attention.linear_qkv.weight_packed"),
        ("nvfp4", "decoder.layers.0.mlp.experts.linear_fc1.weight0_w"),
        ("fp8", "decoder.layers.0.mlp.experts.linear_fc1.weight0_scale_inv"),
    ],
)
def test_adapter_only_resume_rejects_quantized_base_payload(
    monkeypatch: pytest.MonkeyPatch, quant: str, quantized_key: str
) -> None:
    model = _TinyQOFTModel()
    state_dict = {
        "oft_r": torch.ones(2),
        quantized_key: torch.ones(1),
    }

    with _installed_quantized_loader(monkeypatch, quant) as load_model_state_dict:
        with pytest.raises(RuntimeError, match="quantized.*weight"):
            load_model_state_dict(model, state_dict, strict=False, adapter_only=True)
    torch.testing.assert_close(model.oft_r, torch.zeros(2))


@pytest.mark.unit
def test_nvfp4_adapter_only_resume_rejects_dense_split_weight_family(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _TinyQOFTModel()
    state_dict = {
        "oft_r": torch.ones(2),
        "dense.weight_w": torch.ones(2, 64, dtype=torch.uint8),
        "dense.weight_v": torch.ones(2, 64, dtype=torch.uint8),
        "dense.weight_quantizer._scale": torch.ones(4, 4),
        "dense.weight_quantizer._double_scale": torch.ones(()),
    }

    with _installed_quantized_loader(monkeypatch, "nvfp4") as load_model_state_dict:
        with pytest.raises(RuntimeError, match="quantized base payload.*dense.weight_[wv]"):
            load_model_state_dict(model, state_dict, strict=False, adapter_only=True)


@pytest.mark.unit
def test_nvfp4_full_resume_consumes_dense_split_weight_family(monkeypatch: pytest.MonkeyPatch) -> None:
    import megatron.bridge.orbit.low_precision.nvfp4 as nvfp4

    model = _TinyDenseNVFP4Model()
    state_dict = {
        "dense.weight_w": torch.ones(2, 64, dtype=torch.uint8),
        "dense.weight_v": torch.full((2, 64), 2, dtype=torch.uint8),
        "dense.weight_quantizer._scale": torch.ones(4, 4),
        "dense.weight_quantizer._double_scale": torch.ones(()),
    }

    def fake_dequantize(packed, scale, double_scale, shape, *, device, dtype):
        assert torch.equal(packed[:2], state_dict["dense.weight_w"])
        assert torch.equal(packed[2:], state_dict["dense.weight_v"])
        assert shape == (4, 128)
        return torch.full(shape, 3.0, device=device, dtype=dtype)

    monkeypatch.setattr(nvfp4, "dequantize_nvfp4", fake_dequantize)
    with _installed_quantized_loader(
        monkeypatch,
        "nvfp4",
        stub_nvfp4_dense_registration=False,
    ) as load_model_state_dict:
        load_model_state_dict(model, state_dict, strict=False)

    torch.testing.assert_close(model.dense.weight, torch.full_like(model.dense.weight, 3.0))
    assert state_dict == {}


@pytest.mark.unit
@pytest.mark.parametrize("quant", ["int4", "nvfp4"])
def test_non_strict_full_load_still_assigns_meta_base_state(monkeypatch: pytest.MonkeyPatch, quant: str) -> None:
    """Inference strict=False is still a full-base load, not adapter resume."""
    model = _TinyQOFTModel(meta_base=True)
    original_base = model.base
    loaded_base = torch.tensor([5.0, 6.0])

    with _installed_quantized_loader(monkeypatch, quant) as load_model_state_dict:
        load_model_state_dict(
            model,
            {"base": loaded_base, "oft_r": torch.ones(2)},
            strict=False,
        )

    assert model.base is not original_base
    assert model.base.device.type == "cpu"
    torch.testing.assert_close(model.base, loaded_base)


@pytest.mark.unit
def test_fp8_installer_disables_sidecar_restore_and_forwards_full_load(monkeypatch: pytest.MonkeyPatch) -> None:
    qcommon = _load_qoft_common()
    checkpointing = importlib.import_module("megatron.bridge.training.checkpointing")
    modelopt_checkpoint = importlib.import_module("megatron.bridge.orbit.training.modelopt_checkpoint")
    fp8_utils = importlib.import_module("megatron.bridge.orbit.quant.fp8_utils")
    marker = "_qoft_fp8_checkpoint_patches_installed"
    original_generate = checkpointing._generate_model_state_dict
    original_loader = checkpointing._load_model_state_dict
    original_marker = getattr(checkpointing, marker, _MISSING)
    original_modelopt_seam = modelopt_checkpoint._maybe_restore_modelopt_state_for_sharded_load
    calls = []

    def recording_loader(model, state_dict, strict=True, *, adapter_only=False):
        calls.append((model, state_dict, strict, adapter_only))

    if hasattr(checkpointing, marker):
        delattr(checkpointing, marker)
    checkpointing._load_model_state_dict = recording_loader
    monkeypatch.setattr(fp8_utils, "register_fp8_scale_inv_buffers_after_load", lambda model, state: None)

    try:
        qcommon.install_fp8_checkpoint_load_patches(pretrained_checkpoint="unused", arch_label="test")
        installed_loader = checkpointing._load_model_state_dict

        assert modelopt_checkpoint._maybe_restore_modelopt_state_for_sharded_load is not original_modelopt_seam
        assert modelopt_checkpoint._maybe_restore_modelopt_state_for_sharded_load(object(), "unused", None) is False
        installed_loader("model", {"oft_r": "value"}, strict=False, adapter_only=False)
        assert calls == [("model", {"oft_r": "value"}, False, False)]
    finally:
        checkpointing._generate_model_state_dict = original_generate
        checkpointing._load_model_state_dict = original_loader
        _restore_attribute(checkpointing, marker, original_marker)
        modelopt_checkpoint._maybe_restore_modelopt_state_for_sharded_load = original_modelopt_seam

    assert checkpointing._load_model_state_dict is original_loader
    assert modelopt_checkpoint._maybe_restore_modelopt_state_for_sharded_load is original_modelopt_seam
    assert getattr(checkpointing, marker, _MISSING) is original_marker
