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
from types import SimpleNamespace

import pytest

from megatron.bridge.orbit.training import modelopt_checkpoint, modelopt_packed_restore
from megatron.bridge.training import checkpointing
from megatron.bridge.training.post_training import checkpointing as post_training_checkpointing


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
def test_post_training_restore_compresses_after_modelopt_state(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []
    model = [object()]
    monkeypatch.setattr(post_training_checkpointing, "_get_modelopt_checkpoint_path", lambda path: f"{path}/iter_9")
    monkeypatch.setattr(post_training_checkpointing, "unwrap_model", lambda chunks: ["unwrapped"])
    monkeypatch.setattr(
        post_training_checkpointing,
        "restore_sharded_modelopt_state",
        lambda unwrapped, path: events.append(("restore", unwrapped, path)),
    )
    monkeypatch.setattr(
        modelopt_packed_restore,
        "_maybe_compress_restored_modelopt_model",
        lambda unwrapped, path: events.append(("compress", unwrapped, path)),
    )

    post_training_checkpointing.load_modelopt_state(model, "checkpoint")

    assert events == [
        ("restore", ["unwrapped"], "checkpoint/iter_9"),
        ("compress", ["unwrapped"], "checkpoint/iter_9"),
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
