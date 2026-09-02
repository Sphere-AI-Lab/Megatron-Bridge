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
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Iterator

import pytest
import torch

from megatron.bridge.orbit.training import modelopt_packed_restore


def _weightless_real_quant_linear():
    from modelopt.torch.quantization.nn.modules.quant_linear import RealQuantLinear

    module = RealQuantLinear.__new__(RealQuantLinear)
    torch.nn.Module.__init__(module)
    module.weight_quantizer = SimpleNamespace(
        is_enabled=True,
        _fake_quant=False,
        get_modelopt_state=lambda: {"format": "NVFP4"},
    )
    return module


@contextmanager
def _isolated_grouped_moe_patch() -> Iterator[dict[str, object]]:
    base_qtensor = importlib.import_module("modelopt.torch.quantization.qtensor.base_qtensor")
    compress_module = importlib.import_module("modelopt.torch.quantization.compress")
    megatron_plugin = importlib.import_module("modelopt.torch.quantization.plugins.megatron")
    mode_module = importlib.import_module("modelopt.torch.quantization.mode")
    qtensor_package = importlib.import_module("modelopt.torch.quantization.qtensor")
    patched_attributes = [
        (base_qtensor, "pack_real_quantize_weight"),
        (qtensor_package, "pack_real_quantize_weight"),
        (compress_module, "pack_real_quantize_weight"),
        (compress_module, "update_compress_metadata"),
        (mode_module, "update_compress_metadata"),
        (megatron_plugin, "real_quant_module_get_extra_state"),
        (megatron_plugin, "real_quant_module_set_extra_state"),
    ]
    originals = {(module, name): getattr(module, name) for module, name in patched_attributes}
    marker = "_megatron_bridge_grouped_moe_patch_applied"
    marker_missing = object()
    original_marker = getattr(base_qtensor, marker, marker_missing)

    if original_marker is not marker_missing:
        delattr(base_qtensor, marker)

    try:
        modelopt_packed_restore._patch_modelopt_pack_for_grouped_moe()
        yield {
            "base_qtensor": base_qtensor,
            "compress": compress_module,
            "megatron_plugin": megatron_plugin,
        }
    finally:
        for (module, name), original in originals.items():
            setattr(module, name, original)
        if original_marker is marker_missing:
            delattr(base_qtensor, marker)
        else:
            setattr(base_qtensor, marker, original_marker)


@pytest.mark.unit
def test_packed_layout_detection_accepts_entry_and_properties_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = {
        "model.layer.weight": SimpleNamespace(dtype=torch.uint8),
        "model.other.weight": SimpleNamespace(properties=SimpleNamespace(dtype=torch.bfloat16)),
    }
    monkeypatch.setattr(modelopt_packed_restore, "_get_modelopt_checkpoint_path", lambda path: f"{path}/iter_7")
    monkeypatch.setattr(modelopt_packed_restore.dist_checkpointing, "load_tensors_metadata", lambda path: metadata)

    assert modelopt_packed_restore._checkpoint_uses_packed_main_weight_layout("checkpoint")


@pytest.mark.unit
def test_packed_layout_detection_ignores_uint8_non_weight_and_load_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        modelopt_packed_restore.dist_checkpointing,
        "load_tensors_metadata",
        lambda path: {"model.layer.weight_scale": SimpleNamespace(dtype=torch.uint8)},
    )
    assert not modelopt_packed_restore._checkpoint_uses_packed_main_weight_layout("checkpoint")

    def fail(_path):
        raise OSError("missing metadata")

    monkeypatch.setattr(modelopt_packed_restore.dist_checkpointing, "load_tensors_metadata", fail)
    assert not modelopt_packed_restore._checkpoint_uses_packed_main_weight_layout("checkpoint")


@pytest.mark.unit
def test_packed_restore_patches_then_compresses_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import modelopt.torch.quantization as mtq

    compress_module = importlib.import_module("modelopt.torch.quantization.compress")

    events = []
    model = object()
    monkeypatch.setattr(modelopt_packed_restore, "_checkpoint_uses_packed_main_weight_layout", lambda path: True)
    monkeypatch.setattr(
        modelopt_packed_restore, "_patch_modelopt_pack_for_grouped_moe", lambda: events.append("patch")
    )
    monkeypatch.setattr(compress_module, "is_real_quantized", lambda candidate: False)
    monkeypatch.setattr(mtq, "compress", lambda candidate: events.append(("compress", candidate)))

    modelopt_packed_restore._maybe_compress_restored_modelopt_model([model], "checkpoint")

    assert events == ["patch", ("compress", model)]


@pytest.mark.unit
def test_packed_restore_is_idempotent_for_already_quantized_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import modelopt.torch.quantization as mtq

    compress_module = importlib.import_module("modelopt.torch.quantization.compress")

    events = []
    monkeypatch.setattr(modelopt_packed_restore, "_checkpoint_uses_packed_main_weight_layout", lambda path: True)
    monkeypatch.setattr(
        modelopt_packed_restore, "_patch_modelopt_pack_for_grouped_moe", lambda: events.append("patch")
    )
    monkeypatch.setattr(compress_module, "is_real_quantized", lambda candidate: True)
    monkeypatch.setattr(mtq, "compress", lambda candidate: events.append("compress"))

    modelopt_packed_restore._maybe_compress_restored_modelopt_model([object()], "checkpoint")

    assert events == ["patch"]


@pytest.mark.unit
def test_packed_restore_rejects_multiple_model_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(modelopt_packed_restore, "_checkpoint_uses_packed_main_weight_layout", lambda path: True)
    monkeypatch.setattr(modelopt_packed_restore, "_patch_modelopt_pack_for_grouped_moe", lambda: None)

    with pytest.raises(ValueError, match="single model chunk"):
        modelopt_packed_restore._maybe_compress_restored_modelopt_model([object(), object()], "checkpoint")


@pytest.mark.unit
def test_grouped_moe_pack_skips_real_quant_linear_without_weight() -> None:
    root = torch.nn.Module()
    grouped = _weightless_real_quant_linear()
    root.add_module("grouped", grouped)

    with _isolated_grouped_moe_patch() as modules:
        modules["base_qtensor"].pack_real_quantize_weight(root)

    assert not hasattr(grouped, "weight")


@pytest.mark.unit
def test_grouped_moe_patch_omits_weightless_qtensor_metadata() -> None:
    root = torch.nn.Module()
    grouped = _weightless_real_quant_linear()
    root.add_module("grouped", grouped)
    metadata = {}

    with _isolated_grouped_moe_patch() as modules:
        modules["compress"].update_compress_metadata(root, {}, metadata)

    assert metadata == {
        "real_quantizer_state": {"grouped": {"format": "NVFP4"}},
        "q_tensor_state": {},
    }


@pytest.mark.unit
def test_grouped_moe_patch_uses_empty_extra_state_for_weightless_linears() -> None:
    grouped = _weightless_real_quant_linear()

    with _isolated_grouped_moe_patch() as modules:
        extra_state = modules["megatron_plugin"].real_quant_module_get_extra_state(grouped)
        modules["megatron_plugin"].real_quant_module_set_extra_state(
            grouped,
            {"modelopt_q_tensor_state": {"metadata": {}, "quantized_data.dtype": torch.uint8}},
        )

    assert extra_state == {
        "modelopt_real_quantizer_state": None,
        "modelopt_q_tensor_state": None,
    }
    assert not hasattr(grouped, "weight")


@pytest.mark.unit
def test_real_unpatched_update_compress_metadata_crashes_on_weightless_grouped() -> None:
    """The concrete reason the patch must precede restore, on real ModelOpt code.

    ``restore_from_modelopt_state`` replaying a ``real_quantize`` mode enters
    ModelOpt's compress-restore, whose ``update_compress_metadata`` reads
    ``module.weight`` on every RealQuantLinear without a hasattr guard. A grouped
    expert linear has ``weight0..weightN`` and no ``.weight``, so the unpatched
    function raises AttributeError -- before, in a fresh process, the patch that
    used to be installed only after restore ever existed. This drives the real
    function (no mocks), then shows the guarded variant handles the same module.
    """
    compress_module = importlib.import_module("modelopt.torch.quantization.compress")
    marker = "_megatron_bridge_grouped_moe_patch_applied"
    base_qtensor = importlib.import_module("modelopt.torch.quantization.qtensor.base_qtensor")
    if getattr(base_qtensor, marker, False):  # pragma: no cover - defensive
        pytest.skip("grouped-MoE patch already installed process-wide; cannot observe the crash")

    root = torch.nn.Module()
    grouped = _weightless_real_quant_linear()
    root.add_module("grouped", grouped)

    # Real, unpatched ModelOpt function: the crash the ordering fix prevents.
    with pytest.raises(AttributeError):
        compress_module.update_compress_metadata(root, {}, {})

    # Same module, same real function once the guards are installed: no crash.
    with _isolated_grouped_moe_patch() as modules:
        metadata: dict = {}
        modules["compress"].update_compress_metadata(root, {}, metadata)
    assert metadata["q_tensor_state"] == {}
    assert not hasattr(grouped, "weight")
