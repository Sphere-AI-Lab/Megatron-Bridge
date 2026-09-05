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
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.orbit.training import modelopt_checkpoint, modelopt_packed_restore
from megatron.bridge.training import checkpointing
from megatron.bridge.training.post_training import checkpointing as post_training_checkpointing


_REPO_ROOT = Path(__file__).parents[3]


@pytest.mark.unit
def test_modelopt_save_uses_default_nvrx_path(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        checkpointing,
        "save_sharded_modelopt_state",
        lambda model, path, checkpoint_format: calls.append((model, path, checkpoint_format)),
    )

    checkpointing._save_sharded_modelopt_state_for_strategy("model", "checkpoint", ("torch_dist", 1), "nvrx")

    assert calls == [("model", "checkpoint", ("torch_dist", 1))]


@pytest.mark.unit
def test_modelopt_save_forwards_explicit_async_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        modelopt_checkpoint,
        "_save_sharded_modelopt_state_with_async_strategy",
        lambda model, path, checkpoint_format, async_strategy: calls.append(
            (model, path, checkpoint_format, async_strategy)
        ),
    )

    checkpointing._save_sharded_modelopt_state_for_strategy(
        "model",
        "checkpoint",
        ("torch_dist", 1),
        "thread",
    )

    assert calls == [("model", "checkpoint", ("torch_dist", 1), "thread")]


@pytest.mark.unit
def test_modelopt_sidecar_save_omits_unsupported_async_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pinned MCore save variants without ``async_strategy`` remain usable."""
    import modelopt.torch.opt as mto
    import modelopt.torch.utils.distributed as modelopt_dist
    from modelopt.torch.opt.plugins import mcore_dist_checkpointing as modelopt_mcore_dcp

    calls = []

    def legacy_save(state, path, strategy):
        calls.append((state, path, strategy))

    monkeypatch.setattr(modelopt_dist, "is_master", lambda: False)
    monkeypatch.setattr(mto.ModeloptStateManager, "is_converted", lambda model: True)
    monkeypatch.setattr(mto, "modelopt_state", lambda model: {"mode": "quantized"})
    monkeypatch.setattr(modelopt_mcore_dcp, "remove_per_module_state", lambda state: None)
    monkeypatch.setattr(modelopt_checkpoint.dist_checkpointing, "save", legacy_save)
    model = [SimpleNamespace(config=SimpleNamespace())]

    modelopt_checkpoint._save_sharded_modelopt_state_with_async_strategy(
        model,
        "checkpoint",
        ("torch_dist", 1),
        async_strategy="mcore",
    )

    assert calls == [({"mode": "quantized"}, "checkpoint/modelopt_state", ("torch_dist", 1))]


@pytest.mark.unit
def test_modelopt_sidecar_save_forwards_supported_async_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    """Modern MCore must receive the configured sidecar async strategy."""
    import modelopt.torch.opt as mto
    import modelopt.torch.utils.distributed as modelopt_dist
    from modelopt.torch.opt.plugins import mcore_dist_checkpointing as modelopt_mcore_dcp

    calls = []

    def modern_save(state, path, strategy, *, async_strategy=None):
        calls.append((state, path, strategy, async_strategy))

    monkeypatch.setattr(modelopt_dist, "is_master", lambda: False)
    monkeypatch.setattr(mto.ModeloptStateManager, "is_converted", lambda model: True)
    monkeypatch.setattr(mto, "modelopt_state", lambda model: {"mode": "quantized"})
    monkeypatch.setattr(modelopt_mcore_dcp, "remove_per_module_state", lambda state: None)
    monkeypatch.setattr(modelopt_checkpoint.dist_checkpointing, "save", modern_save)

    modelopt_checkpoint._save_sharded_modelopt_state_with_async_strategy(
        [SimpleNamespace(config=SimpleNamespace())],
        "checkpoint",
        ("torch_dist", 1),
        async_strategy="mcore",
    )

    assert calls == [({"mode": "quantized"}, "checkpoint/modelopt_state", ("torch_dist", 1), "mcore")]


@pytest.mark.unit
def test_importing_direct_nvfp4_converter_does_not_mutate_global_save(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operational entrypoint imports must not monkeypatch process-global MCore APIs."""
    script_path = _REPO_ROOT / "scripts" / "orbit" / "conversion" / "convert_nvfp4_checkpoint_direct.py"
    spec = importlib.util.spec_from_file_location("nvfp4_direct_save_compat_under_test", script_path)
    assert spec is not None and spec.loader is not None
    original_save = modelopt_checkpoint.dist_checkpointing.save
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        assert modelopt_checkpoint.dist_checkpointing.save is original_save

        requested_strategies = []

        def capture_save(*, state, **kwargs):
            requested_strategies.append(state.cfg.checkpoint.async_strategy)

        monkeypatch.setattr(module, "save_checkpoint", capture_save)
        monkeypatch.setattr(module, "_maybe_create_save_progress_monitor", lambda path: None)
        module._save_direct_checkpoint(
            SimpleNamespace(),
            "checkpoint",
            {},
            model_list=[],
            pg_collection=object(),
            hf_tokenizer_path=None,
            hf_tokenizer_kwargs=None,
        )
    finally:
        modelopt_checkpoint.dist_checkpointing.save = original_save

    assert requested_strategies == ["mcore"]


@pytest.mark.unit
def test_modelopt_restore_runs_before_schema_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    state_dict = {"modelopt_state": object()}
    monkeypatch.setattr(
        modelopt_checkpoint,
        "_maybe_restore_modelopt_state_for_sharded_load",
        lambda model, path, state: calls.append((model, path, state)),
    )

    checkpointing._restore_modelopt_state_before_sharded_schema("model", "checkpoint", state_dict)

    assert calls == [("model", "checkpoint", state_dict)]


@pytest.mark.unit
def test_modelopt_restore_skips_non_path_local_checkpoint_id(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        modelopt_checkpoint,
        "_maybe_restore_modelopt_state_for_sharded_load",
        lambda model, path, state: calls.append((model, path, state)),
    )

    checkpointing._restore_modelopt_state_before_sharded_schema("model", 42, {"iteration": 42})

    assert calls == []


@pytest.mark.unit
def test_post_training_restore_patches_before_restore_then_compresses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Order matters: the grouped-MoE ``.weight`` guards must be installed before
    ModelOpt's restore replays the saved mode list. A run checkpoint written
    after ``mtq.compress`` contains ``real_quantize``; replaying it unpatched in
    a fresh process raised AttributeError on grouped expert linears before the
    patch (which used to be installed after the restore) ever existed."""
    events = []
    model = [object()]
    monkeypatch.setattr(post_training_checkpointing, "_get_modelopt_checkpoint_path", lambda path: f"{path}/iter_9")
    monkeypatch.setattr(post_training_checkpointing, "unwrap_model", lambda chunks: ["unwrapped"])
    monkeypatch.setattr(
        modelopt_packed_restore,
        "_patch_modelopt_pack_for_grouped_moe",
        lambda: events.append(("patch",)),
    )
    monkeypatch.setattr(
        modelopt_checkpoint,
        "restore_sharded_modelopt_state_via_common_reader",
        lambda unwrapped, path: events.append(("restore", unwrapped, path)) or True,
    )
    monkeypatch.setattr(
        modelopt_packed_restore,
        "_maybe_compress_restored_modelopt_model",
        lambda unwrapped, path: events.append(("compress", unwrapped, path)),
    )

    restored = post_training_checkpointing.load_modelopt_state(model, "checkpoint")

    assert restored is True
    assert events == [
        ("patch",),
        ("restore", ["unwrapped"], "checkpoint/iter_9"),
        ("compress", ["unwrapped"], "checkpoint/iter_9"),
    ]


@pytest.mark.unit
def test_post_training_restore_false_skips_packed_compression(monkeypatch: pytest.MonkeyPatch) -> None:
    """An already-converted model reports false and must not be compressed again."""
    monkeypatch.setattr(post_training_checkpointing, "unwrap_model", lambda chunks: chunks)
    monkeypatch.setattr(modelopt_packed_restore, "_patch_modelopt_pack_for_grouped_moe", lambda: None)
    monkeypatch.setattr(
        modelopt_checkpoint,
        "restore_sharded_modelopt_state_via_common_reader",
        lambda model, path: False,
    )
    monkeypatch.setattr(
        modelopt_packed_restore,
        "_maybe_compress_restored_modelopt_model",
        lambda *args: pytest.fail("packed compression must follow a successful sidecar restore only"),
    )

    restored = post_training_checkpointing.load_modelopt_state(["m"], "ckpt")

    assert restored is False


@pytest.mark.unit
def test_sharded_load_restore_installs_grouped_moe_patch_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sidecar replay and packed restoration must finish before schema creation."""
    events = []
    monkeypatch.setattr(post_training_checkpointing, "has_modelopt_state", lambda path: True)
    monkeypatch.setattr(post_training_checkpointing, "unwrap_model", lambda chunks: chunks)
    monkeypatch.setattr(
        modelopt_packed_restore,
        "_patch_modelopt_pack_for_grouped_moe",
        lambda: events.append("patch"),
    )
    monkeypatch.setattr(
        modelopt_checkpoint,
        "restore_sharded_modelopt_state_via_common_reader",
        lambda model, path: events.append(("restore", model, path)) or True,
    )
    monkeypatch.setattr(
        modelopt_checkpoint,
        "restore_modelopt_state",
        lambda *args: events.append(("legacy", *args)),
    )
    monkeypatch.setattr(
        modelopt_packed_restore,
        "_maybe_compress_restored_modelopt_model",
        lambda model, path: events.append(("compress", model, path)),
    )

    restored = modelopt_checkpoint._maybe_restore_modelopt_state_for_sharded_load(["m"], "ckpt", {"k": 1})

    assert restored is True
    assert events == ["patch", ("restore", ["m"], "ckpt"), ("compress", ["m"], "ckpt")]


@pytest.mark.unit
def test_sharded_load_restore_falls_back_to_embedded_modelopt_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Old checkpoints with only embedded ModelOpt state remain loadable."""
    events = []
    state = {"modelopt_state": {"mode": "legacy"}}
    monkeypatch.setattr(post_training_checkpointing, "has_modelopt_state", lambda path: False)
    monkeypatch.setattr(modelopt_checkpoint, "unwrap_model", lambda chunks: ["unwrapped"])
    monkeypatch.setattr(
        modelopt_packed_restore, "_patch_modelopt_pack_for_grouped_moe", lambda: events.append("patch")
    )
    monkeypatch.setattr(
        post_training_checkpointing,
        "load_modelopt_state",
        lambda *args: pytest.fail("sidecar loader must not run without a sidecar"),
    )
    monkeypatch.setattr(
        modelopt_checkpoint,
        "restore_modelopt_state",
        lambda model, common: events.append(("legacy", model, common)),
    )
    monkeypatch.setattr(
        modelopt_packed_restore,
        "_maybe_compress_restored_modelopt_model",
        lambda model, path: events.append(("compress", model, path)),
    )

    restored = modelopt_checkpoint._maybe_restore_modelopt_state_for_sharded_load(["m"], "ckpt", state)

    assert restored is False
    assert events == ["patch", ("legacy", ["m"], state), ("compress", ["unwrapped"], "ckpt")]


@pytest.mark.unit
def test_present_sidecar_does_not_fall_back_when_model_is_already_converted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A truthful false from a present sidecar must not replay embedded modes."""
    state = {"modelopt_state": {"mode": "legacy"}}
    monkeypatch.setattr(post_training_checkpointing, "has_modelopt_state", lambda path: True)
    monkeypatch.setattr(post_training_checkpointing, "load_modelopt_state", lambda model, path: False)
    monkeypatch.setattr(
        modelopt_checkpoint,
        "restore_modelopt_state",
        lambda *args: pytest.fail("embedded fallback must not run when a sidecar is present"),
    )

    restored = modelopt_checkpoint._maybe_restore_modelopt_state_for_sharded_load(["m"], "ckpt", state)

    assert restored is False


@pytest.mark.unit
def test_sharded_sidecar_restore_does_not_require_common_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sidecar is independently loadable when the main common dict is absent."""
    calls = []
    monkeypatch.setattr(post_training_checkpointing, "has_modelopt_state", lambda path: True)
    monkeypatch.setattr(
        post_training_checkpointing,
        "load_modelopt_state",
        lambda model, path: calls.append((model, path)) or True,
    )
    monkeypatch.setattr(
        modelopt_checkpoint,
        "restore_modelopt_state",
        lambda *args: pytest.fail("embedded fallback must not run after sidecar restore"),
    )

    restored = modelopt_checkpoint._maybe_restore_modelopt_state_for_sharded_load(["m"], "ckpt", None)

    assert restored is True
    assert calls == [(["m"], "ckpt")]


@pytest.mark.unit
def test_standard_training_load_completes_modelopt_restore_before_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real training-load branch so moving its seam below schema fails."""
    events = []
    model = [object()]
    checkpoint_cfg = SimpleNamespace(
        ckpt_format="torch_dist",
        finetune=True,
        load_rng=False,
        load_optim=False,
        save_optim=False,
        save_rng=False,
        fully_parallel_save=False,
    )
    cfg = SimpleNamespace(
        checkpoint=checkpoint_cfg,
        model=SimpleNamespace(tensor_model_parallel_size=1, pipeline_model_parallel_size=1),
        optimizer=SimpleNamespace(use_distributed_optimizer=False),
    )
    state = SimpleNamespace(cfg=cfg)
    common_state = {"iteration": 3, "modelopt_state": {"mode": "legacy"}}
    run_config = {
        "model": {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1},
        "checkpoint": {"save_optim": False, "save_rng": False, "fully_parallel_save": False},
    }

    monkeypatch.setattr(checkpointing, "unwrap_model", lambda value: value)
    monkeypatch.setattr(checkpointing, "get_pg_collection", lambda value: SimpleNamespace(dp_cp="dp"))
    monkeypatch.setattr(
        checkpointing,
        "_load_base_checkpoint",
        lambda *args, **kwargs: (common_state, "checkpoint/iter_0000003", False, checkpointing.CheckpointType.GLOBAL),
    )
    monkeypatch.setattr(checkpointing, "get_checkpoint_run_config_filename", lambda path: "run_config.yaml")
    monkeypatch.setattr(checkpointing, "file_exists", lambda path: True)
    monkeypatch.setattr(checkpointing, "read_run_config", lambda path: run_config)
    monkeypatch.setattr(checkpointing.dist_checkpointing, "load_content_metadata", lambda **kwargs: {})
    monkeypatch.setattr(post_training_checkpointing, "has_modelopt_state", lambda path: True)
    monkeypatch.setattr(post_training_checkpointing, "unwrap_model", lambda chunks: chunks)
    monkeypatch.setattr(
        modelopt_packed_restore, "_patch_modelopt_pack_for_grouped_moe", lambda: events.append("patch")
    )
    monkeypatch.setattr(
        modelopt_checkpoint,
        "restore_sharded_modelopt_state_via_common_reader",
        lambda loaded_model, path: events.append("restore") or True,
    )
    monkeypatch.setattr(
        modelopt_packed_restore,
        "_maybe_compress_restored_modelopt_model",
        lambda loaded_model, path: events.append("compress"),
    )
    monkeypatch.setattr(modelopt_checkpoint, "restore_modelopt_state", lambda *args: events.append("legacy"))

    class SchemaGenerated(RuntimeError):
        pass

    def generate_schema(*args, **kwargs):
        events.append("schema")
        raise SchemaGenerated

    monkeypatch.setattr(checkpointing, "generate_state_dict", generate_schema)

    with pytest.raises(SchemaGenerated):
        checkpointing._load_checkpoint_from_path(
            "checkpoint",
            state,
            model,
            optimizer=None,
            opt_param_scheduler=None,
        )

    assert events == ["patch", "restore", "compress", "schema"]


@pytest.mark.unit
def test_inference_weight_loader_completes_modelopt_restore_before_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the real weights-only seam through packed restore and schema."""
    import modelopt.torch.quantization as mtq

    compress_module = importlib.import_module("modelopt.torch.quantization.compress")
    events = []
    state_dict = {"iteration": 12}
    monkeypatch.setattr(checkpointing.dist_checkpointing, "load_common_state_dict", lambda path: state_dict)
    monkeypatch.setattr(checkpointing.dist_checkpointing, "load_content_metadata", lambda **kwargs: {})
    monkeypatch.setattr(post_training_checkpointing, "has_modelopt_state", lambda path: True)
    monkeypatch.setattr(post_training_checkpointing, "_get_modelopt_checkpoint_path", lambda path: path)
    monkeypatch.setattr(
        post_training_checkpointing,
        "unwrap_model",
        lambda model: events.append(("restore_unwrap", model)) or ["restore_model"],
    )
    monkeypatch.setattr(
        modelopt_packed_restore,
        "_patch_modelopt_pack_for_grouped_moe",
        lambda: events.append("patch"),
    )
    monkeypatch.setattr(
        modelopt_checkpoint,
        "restore_sharded_modelopt_state_via_common_reader",
        lambda model, path: events.append(("restore", model, path)) or True,
    )
    monkeypatch.setattr(
        modelopt_packed_restore.dist_checkpointing,
        "load_tensors_metadata",
        lambda path: events.append(("metadata", path))
        or {
            "model.layer.weight_w": SimpleNamespace(dtype=torch.float8_e4m3fn),
            "model.layer.weight_scale_inv": SimpleNamespace(dtype=torch.float32),
        },
    )
    monkeypatch.setattr(
        compress_module,
        "is_real_quantized",
        lambda model: events.append(("is_real_quantized", model)) or False,
    )
    monkeypatch.setattr(mtq, "compress", lambda model: events.append(("compress", model)))
    monkeypatch.setattr(
        modelopt_checkpoint,
        "restore_modelopt_state",
        lambda *args: pytest.fail("embedded fallback must not run for a present sidecar"),
    )
    # Present only on a partial implementation that directly restores the
    # embedded state. ``raising=False`` keeps the sentinel valid after the
    # production import was removed.
    monkeypatch.setattr(
        checkpointing,
        "restore_modelopt_state",
        lambda *args: events.append(("direct_legacy", *args)),
        raising=False,
    )
    monkeypatch.setattr(
        checkpointing,
        "unwrap_model",
        lambda model: events.append(("schema_unwrap", model)) or ["schema_model"],
    )
    monkeypatch.setattr(checkpointing, "get_pg_collection", lambda model: SimpleNamespace(dp_cp="dp"))

    def generate_schema(model, kwargs, pg_collection):
        events.append(("schema", model))
        raise RuntimeError("schema generated")

    monkeypatch.setattr(checkpointing, "_generate_model_state_dict", generate_schema)

    with pytest.raises(RuntimeError, match="schema generated"):
        checkpointing._load_model_weights_from_checkpoint("checkpoint", ["model"])

    assert events == [
        "patch",
        ("restore_unwrap", ["model"]),
        ("restore", ["restore_model"], "checkpoint"),
        ("metadata", "checkpoint"),
        "patch",
        ("is_real_quantized", "restore_model"),
        ("compress", "restore_model"),
        ("schema_unwrap", ["model"]),
        ("schema", ["schema_model"]),
    ]


@pytest.mark.unit
def test_setup_registers_modelopt_before_peft_and_selects_load_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    setup_module = importlib.import_module("megatron.bridge.training.setup")
    hooks = []
    loads = []

    def peft_hook(model):
        return model

    monkeypatch.setattr(setup_module, "_register_pre_wrap_hook", lambda model_cfg, hook: hooks.append(hook))
    monkeypatch.setattr(setup_module, "_create_peft_pre_wrap_hook", lambda cfg, state: peft_hook)
    monkeypatch.setattr(setup_module, "print_rank_0", lambda message: None)
    monkeypatch.setattr(post_training_checkpointing, "has_modelopt_state", lambda path: path == "load-checkpoint")
    monkeypatch.setattr(
        post_training_checkpointing,
        "load_modelopt_state",
        lambda model, path: loads.append((model, path)),
    )
    cfg = SimpleNamespace(
        model=SimpleNamespace(restore_modelopt_state=True),
        checkpoint=SimpleNamespace(pretrained_checkpoint="pretrained", load="load-checkpoint"),
        peft=object(),
    )

    setup_module._register_modelopt_and_peft_pre_wrap_hooks(cfg, SimpleNamespace())

    assert [hook.__name__ for hook in hooks] == ["modelopt_pre_wrap_hook", "peft_hook"]
    model = [object()]
    assert hooks[0](model) is model
    assert loads == [(model, "load-checkpoint")]
