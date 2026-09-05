# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import datetime
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union
from unittest.mock import Mock, patch

import megatron.core.parallel_state as parallel_state
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.core import dist_checkpointing as mcore_dist_checkpointing
from megatron.core.dist_checkpointing import serialization as mcore_dist_checkpointing_serialization
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor, ShardedTensorFactory
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.transformer.module import MegatronModule

import megatron.bridge.training.checkpointing as checkpointing
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.training.checkpointing import (
    CheckpointType,
    _is_model_section,
    _validate_peft_run_resume_tensor_schema,
    apply_peft_adapter_filter_to_state_dict,
    load_checkpoint,
)
from megatron.bridge.training.config import CheckpointConfig, ConfigContainer
from megatron.bridge.training.state import GlobalState


@dataclass
class MockPEFT(PEFT):
    """Mock PEFT implementation for testing."""

    def __post_init__(self) -> None:
        """Set up mock parameters after dataclass initialization."""
        self.params_to_save = {
            "layer1.adapter.weight",
            "layer2.adapter.bias",
            "layer3.adapters.lora_A",
            "layer3.adapters.lora_B",
        }

    def transform(self, module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None) -> nn.Module:
        """Transform method that returns the module unchanged for testing."""
        return module

    def adapter_key_filter(self, key: Union[str, tuple]) -> bool:
        """Filter function that only allows adapter parameters."""
        if isinstance(key, tuple):
            return key[1].requires_grad
        return key in self.params_to_save or ".adapter." in key or key.endswith(".adapters") or "lora_" in key


class TestIsModelSection:
    """Test suite for _is_model_section helper function."""

    def test_is_model_section_single_model(self):
        """Test detection of single model section."""
        assert _is_model_section("model") == True

    def test_is_model_section_pipeline_models(self):
        """Test detection of pipeline model sections."""
        assert _is_model_section("model0") == True
        assert _is_model_section("model1") == True
        assert _is_model_section("model42") == True
        assert _is_model_section("model999") == True

    def test_is_model_section_non_model_sections(self):
        """Test that non-model sections are correctly identified."""
        assert _is_model_section("optimizer") == False
        assert _is_model_section("iteration") == False
        assert _is_model_section("checkpoint_version") == False
        assert _is_model_section("rng_state") == False
        assert _is_model_section("opt_param_scheduler") == False

    def test_is_model_section_invalid_model_keys(self):
        """Test that invalid model-like keys are correctly rejected."""
        assert _is_model_section("model_not_digit") == False
        assert _is_model_section("modelabc") == False
        assert _is_model_section("model_") == False
        assert _is_model_section("model10_extra") == False
        assert _is_model_section("my_model") == False
        assert _is_model_section("models") == False

    def test_is_model_section_edge_cases(self):
        """Test edge cases for model section detection."""
        assert _is_model_section("") == False
        assert _is_model_section("model00") == True  # Leading zeros are valid digits
        assert _is_model_section("model01") == True


def _make_sharded_tensor(
    key: str,
    shape: tuple[int, ...] = (2, 3),
    dtype: torch.dtype = torch.float32,
) -> ShardedTensor:
    return ShardedTensor.from_rank_offsets(key, torch.empty(shape, dtype=dtype))


def _make_sharded_object(key: str, offset: int = 0, fragments: int = 1) -> ShardedObject:
    return ShardedObject(key, {"offset": offset}, (fragments,), (offset,))


@pytest.fixture
def single_rank_dcp_process_group(tmp_path: Path):
    """Provide the real one-rank Gloo context required by MCore DCP."""
    created = False
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{tmp_path}/peft_schema_pg",
            rank=0,
            world_size=1,
        )
        created = True
    elif dist.get_world_size() != 1:
        pytest.skip("one-rank DCP regression requires a one-rank process group")
    yield
    if created:
        dist.destroy_process_group()


class TestPEFTRunResumeTensorSchemaValidation:
    """Validate adapter-only payloads before distributed checkpoint loading."""

    def test_object_metadata_uses_serialization_api_when_package_root_does_not_export_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pinned MCore exposes object metadata only from serialization.py."""
        optimizer_object = _make_sharded_object("optimizer.distributed.param_state")
        metadata = {optimizer_object.unique_key: optimizer_object.without_data()}
        calls: list[str] = []

        monkeypatch.delattr(checkpointing.dist_checkpointing, "load_sharded_metadata", raising=False)
        monkeypatch.setattr(
            checkpointing.dist_checkpointing_serialization,
            "load_sharded_metadata",
            lambda checkpoint_name: calls.append(checkpoint_name) or metadata,
        )

        loaded = checkpointing._load_global_sharded_metadata("/checkpoint/iter_0000007", include_objects=True)

        assert loaded["tensors"] == {}
        assert loaded["objects"] == {optimizer_object.unique_key}
        assert calls == ["/checkpoint/iter_0000007"]

    @staticmethod
    def _schemas() -> tuple[dict, dict]:
        adapter = _make_sharded_tensor("decoder.layers.0.adapter.weight")
        base = _make_sharded_tensor("decoder.layers.0.linear.weight", (4, 3))
        optimizer = _make_sharded_tensor("optimizer.state.exp_avg.decoder.layers.0.adapter.weight")
        optimizer_section = {"state": optimizer, "param_groups": [{"lr": 0.001, "step": 7}]}
        full = {"model": {"adapter": adapter, "base": base}, "optimizer": optimizer_section}
        filtered = {"model": {"adapter": adapter}, "optimizer": optimizer_section}
        return full, filtered

    @pytest.mark.parametrize(
        "unexpected_key",
        [
            "decoder.layers.0.linear.weight",
            "different_architecture.layers.0.weight",
            "decoder.layers.0.linear.weight_scale",
            "decoder.layers.0.linear.weight_scale_2",
        ],
    )
    def test_rejects_checkpoint_only_model_or_quantized_tensor_before_load(self, unexpected_key):
        full, filtered = self._schemas()
        metadata = {
            "decoder.layers.0.adapter.weight": _make_sharded_tensor("decoder.layers.0.adapter.weight"),
            unexpected_key: _make_sharded_tensor(unexpected_key),
        }

        with (
            patch(
                "megatron.bridge.training.checkpointing.dist_checkpointing_serialization.load_sharded_metadata",
                return_value=metadata,
            ),
            pytest.raises(RuntimeError, match=unexpected_key),
        ):
            _validate_peft_run_resume_tensor_schema(
                full,
                filtered,
                "/checkpoint/iter_0000007",
                CheckpointType.GLOBAL,
                "torch_dist",
            )

    @pytest.mark.parametrize(
        ("metadata", "message"),
        [
            ({}, "missing adapter tensor"),
            (
                {"decoder.layers.0.adapter.weight": _make_sharded_tensor("decoder.layers.0.adapter.weight", (3, 3))},
                "shape mismatch",
            ),
            (
                {
                    "decoder.layers.0.adapter.weight": _make_sharded_tensor(
                        "decoder.layers.0.adapter.weight", dtype=torch.float16
                    )
                },
                "dtype mismatch",
            ),
        ],
    )
    def test_rejects_missing_or_incompatible_adapter(self, metadata, message):
        full, filtered = self._schemas()

        with (
            patch(
                "megatron.bridge.training.checkpointing.dist_checkpointing_serialization.load_sharded_metadata",
                return_value=metadata,
            ),
            pytest.raises(RuntimeError, match=message),
        ):
            _validate_peft_run_resume_tensor_schema(
                full,
                filtered,
                "/checkpoint/iter_0000007",
                CheckpointType.GLOBAL,
                "torch_dist",
            )

    def test_expands_factory_children_without_mutating_live_schema(self):
        def build_factory(key, data, replica_id, flattened_range):
            del data, replica_id, flattened_range
            return {
                "left": _make_sharded_tensor(f"{key}.left", (2, 2)),
                "right": _make_sharded_tensor(f"{key}.right", (2, 2)),
            }

        factory = ShardedTensorFactory(
            "decoder.layers.0.adapter.weight",
            torch.empty(2, 4),
            build_factory,
            lambda state: state,
        )
        full = {"model": {"adapter": factory}}
        filtered = {"model": {"adapter": factory}}
        metadata = {
            "decoder.layers.0.adapter.weight.left": _make_sharded_tensor(
                "decoder.layers.0.adapter.weight.left", (2, 2)
            ),
            "decoder.layers.0.adapter.weight.right": _make_sharded_tensor(
                "decoder.layers.0.adapter.weight.right", (2, 2)
            ),
        }

        with patch(
            "megatron.bridge.training.checkpointing.dist_checkpointing.load_tensors_metadata",
            return_value=metadata,
        ):
            _validate_peft_run_resume_tensor_schema(
                full,
                filtered,
                "/checkpoint/iter_0000007",
                CheckpointType.GLOBAL,
                "torch_dist",
            )

        assert full["model"]["adapter"] is factory
        assert filtered["model"]["adapter"] is factory

        with (
            patch(
                "megatron.bridge.training.checkpointing.dist_checkpointing.load_tensors_metadata",
                return_value={
                    "decoder.layers.0.adapter.weight.left": metadata["decoder.layers.0.adapter.weight.left"]
                },
            ),
            pytest.raises(RuntimeError, match="adapter.weight.right"),
        ):
            _validate_peft_run_resume_tensor_schema(
                full,
                filtered,
                "/checkpoint/iter_0000007",
                CheckpointType.GLOBAL,
                "torch_dist",
            )

    def test_requires_adapters_from_every_model_chunk(self):
        adapter0 = _make_sharded_tensor("decoder.layers.0.adapter.weight")
        adapter1 = _make_sharded_tensor("decoder.layers.1.adapter.weight")
        full = {"model0": {"adapter": adapter0}, "model1": {"adapter": adapter1}}
        filtered = {"model0": {"adapter": adapter0}, "model1": {"adapter": adapter1}}

        with (
            patch(
                "megatron.bridge.training.checkpointing.dist_checkpointing.load_tensors_metadata",
                return_value={"decoder.layers.0.adapter.weight": adapter0},
            ),
            pytest.raises(RuntimeError, match="decoder.layers.1.adapter.weight"),
        ):
            _validate_peft_run_resume_tensor_schema(
                full,
                filtered,
                "/checkpoint/iter_0000007",
                CheckpointType.GLOBAL,
                "torch_dist",
            )

    @pytest.mark.parametrize(
        "optimizer_key",
        [
            "optimizer.state.exp_avg.decoder.layers.0.adapter.weight",
            "chained_2.optimizer.state.exp_avg.decoder.layers.0.adapter.weight",
            "mimo.language.optimizer.state.exp_avg.decoder.layers.0.adapter.weight",
            "mimo.language.chained_2.optimizer.state.exp_avg.decoder.layers.0.adapter.weight",
        ],
    )
    def test_accepts_anchored_saved_optimizer_namespaces_when_optimizer_not_loaded(self, optimizer_key):
        adapter = _make_sharded_tensor("decoder.layers.0.adapter.weight")
        full = {"model": {"adapter": adapter}}
        filtered = {"model": {"adapter": adapter}}
        metadata = {
            "decoder.layers.0.adapter.weight": _make_sharded_tensor("decoder.layers.0.adapter.weight"),
            optimizer_key: _make_sharded_tensor(optimizer_key),
        }

        with patch(
            "megatron.bridge.training.checkpointing.dist_checkpointing.load_tensors_metadata",
            return_value=metadata,
        ):
            _validate_peft_run_resume_tensor_schema(
                full,
                filtered,
                "/checkpoint/iter_0000007",
                CheckpointType.GLOBAL,
                "torch_dist",
            )

    def test_accepts_exact_requested_optimizer_descriptor(self):
        full, filtered = self._schemas()
        optimizer_key = "optimizer.state.exp_avg.decoder.layers.0.adapter.weight"
        metadata = {
            "decoder.layers.0.adapter.weight": _make_sharded_tensor("decoder.layers.0.adapter.weight"),
            optimizer_key: _make_sharded_tensor(optimizer_key),
        }

        with patch(
            "megatron.bridge.training.checkpointing.dist_checkpointing_serialization.load_sharded_metadata",
            return_value=metadata,
        ):
            _validate_peft_run_resume_tensor_schema(
                full,
                filtered,
                "/checkpoint/iter_0000007",
                CheckpointType.GLOBAL,
                "torch_dist",
            )

    def test_rejects_unrequested_optimizer_tensor_when_optimizer_is_loaded(self):
        full, filtered = self._schemas()
        unexpected_key = "optimizer.state.exp_avg.decoder.layers.9.adapter.weight"
        metadata = {
            "decoder.layers.0.adapter.weight": _make_sharded_tensor("decoder.layers.0.adapter.weight"),
            unexpected_key: _make_sharded_tensor(unexpected_key),
        }

        with (
            patch(
                "megatron.bridge.training.checkpointing.dist_checkpointing_serialization.load_sharded_metadata",
                return_value=metadata,
            ),
            pytest.raises(RuntimeError, match="unrequested optimizer tensor"),
        ):
            _validate_peft_run_resume_tensor_schema(
                full,
                filtered,
                "/checkpoint/iter_0000007",
                CheckpointType.GLOBAL,
                "torch_dist",
            )

    @pytest.mark.parametrize("unexpected_key", ["model.optimizer_like.weight", "myoptimizer.state", "unknown.tensor"])
    def test_rejects_unanchored_non_adapter_namespaces(self, unexpected_key):
        full, filtered = self._schemas()
        metadata = {
            "decoder.layers.0.adapter.weight": _make_sharded_tensor("decoder.layers.0.adapter.weight"),
            unexpected_key: _make_sharded_tensor(unexpected_key),
        }

        with (
            patch(
                "megatron.bridge.training.checkpointing.dist_checkpointing_serialization.load_sharded_metadata",
                return_value=metadata,
            ),
            pytest.raises(RuntimeError, match=unexpected_key),
        ):
            _validate_peft_run_resume_tensor_schema(
                full,
                filtered,
                "/checkpoint/iter_0000007",
                CheckpointType.GLOBAL,
                "torch_dist",
            )

    def test_uses_resolved_iteration_path_for_metadata(self):
        full, filtered = self._schemas()
        checkpoint_path = "/checkpoint/iter_0000007"

        with patch(
            "megatron.bridge.training.checkpointing.dist_checkpointing_serialization.load_sharded_metadata",
            return_value={
                "decoder.layers.0.adapter.weight": _make_sharded_tensor("decoder.layers.0.adapter.weight"),
                "optimizer.state.exp_avg.decoder.layers.0.adapter.weight": _make_sharded_tensor(
                    "optimizer.state.exp_avg.decoder.layers.0.adapter.weight"
                ),
            },
        ) as load_metadata:
            _validate_peft_run_resume_tensor_schema(
                full,
                filtered,
                checkpoint_path,
                CheckpointType.GLOBAL,
                "torch_dist",
            )

        load_metadata.assert_called_once_with(checkpoint_path)

    def test_rejects_requested_optimizer_descriptor_mismatch(self):
        full, filtered = self._schemas()
        metadata = {
            "decoder.layers.0.adapter.weight": _make_sharded_tensor("decoder.layers.0.adapter.weight"),
            "optimizer.state.exp_avg.decoder.layers.0.adapter.weight": _make_sharded_tensor(
                "optimizer.state.exp_avg.decoder.layers.0.adapter.weight", dtype=torch.float16
            ),
        }

        with (
            patch(
                "megatron.bridge.training.checkpointing.dist_checkpointing_serialization.load_sharded_metadata",
                return_value=metadata,
            ),
            pytest.raises(RuntimeError, match="optimizer.state.exp_avg.*mismatch"),
        ):
            _validate_peft_run_resume_tensor_schema(
                full,
                filtered,
                "/checkpoint/iter_0000007",
                CheckpointType.GLOBAL,
                "torch_dist",
            )

    def test_rejects_missing_requested_optimizer_tensor(self):
        full, filtered = self._schemas()

        with (
            patch(
                "megatron.bridge.training.checkpointing.dist_checkpointing_serialization.load_sharded_metadata",
                return_value={
                    "decoder.layers.0.adapter.weight": _make_sharded_tensor("decoder.layers.0.adapter.weight")
                },
            ),
            pytest.raises(RuntimeError, match="missing optimizer tensor"),
        ):
            _validate_peft_run_resume_tensor_schema(
                full,
                filtered,
                "/checkpoint/iter_0000007",
                CheckpointType.GLOBAL,
                "torch_dist",
            )

    def test_rejects_empty_adapter_schema(self):
        full, _ = self._schemas()

        with (
            patch("megatron.bridge.training.checkpointing.dist_checkpointing.load_tensors_metadata") as load_metadata,
            pytest.raises(RuntimeError, match="no adapter tensor descriptors"),
        ):
            _validate_peft_run_resume_tensor_schema(
                full,
                {"model": {}},
                "/checkpoint/iter_0000007",
                CheckpointType.GLOBAL,
                "torch_dist",
            )

        load_metadata.assert_not_called()

    @pytest.mark.parametrize("model_section", ["model", "model1"])
    def test_rejects_common_model_payload_before_metadata_read(self, model_section):
        """A plain model tensor in DCP common state must not bypass tensor metadata checks."""
        full, filtered = self._schemas()
        common_state_dict = {
            model_section: {"decoder.layers.0.linear.weight": torch.ones(4, 3)},
            "optimizer": {"grad_scaler": {"scale": torch.tensor([65536.0])}},
        }

        with (
            patch("megatron.bridge.training.checkpointing.dist_checkpointing.load_tensors_metadata") as load_metadata,
            pytest.raises(RuntimeError, match="common.*model"),
        ):
            _validate_peft_run_resume_tensor_schema(
                full,
                filtered,
                "/checkpoint/iter_0000007",
                CheckpointType.GLOBAL,
                "torch_dist",
                common_state_dict=common_state_dict,
            )

        load_metadata.assert_not_called()

    @pytest.mark.unit
    def test_rejects_model_tensor_exposed_by_real_dcp_common_reader(
        self,
        single_rank_dcp_process_group,
        tmp_path: Path,
    ):
        """A real DCP common tensor is rejected before it can mutate the live adapter."""
        del single_rank_dcp_process_group
        checkpoint_dir = tmp_path / "common_model_checkpoint"
        checkpoint_dir.mkdir()
        saved_adapter = _make_sharded_tensor("decoder.layers.0.adapter.weight")
        common_base = torch.full((4, 3), 17.0)
        mcore_dist_checkpointing.save(
            {
                "model": {
                    "adapter": saved_adapter,
                    "decoder.layers.0.linear.weight": common_base,
                }
            },
            str(checkpoint_dir),
        )
        common_state_dict = mcore_dist_checkpointing.load_common_state_dict(str(checkpoint_dir))
        torch.testing.assert_close(
            common_state_dict["model"]["decoder.layers.0.linear.weight"],
            common_base,
        )

        live_adapter_data = torch.full((2, 3), -9.0)
        live_adapter = ShardedTensor.from_rank_offsets(
            "decoder.layers.0.adapter.weight",
            live_adapter_data,
        )
        live_base = _make_sharded_tensor("decoder.layers.0.linear.weight", (4, 3))
        full = {"model": {"adapter": live_adapter, "base": live_base}}
        filtered = {"model": {"adapter": live_adapter}}

        with pytest.raises(RuntimeError, match="common.*model"):
            _validate_peft_run_resume_tensor_schema(
                full,
                filtered,
                checkpoint_dir,
                CheckpointType.GLOBAL,
                "torch_dist",
                common_state_dict=common_state_dict,
            )

        torch.testing.assert_close(live_adapter_data, torch.full((2, 3), -9.0))

    def test_allows_legitimate_optimizer_common_state(self):
        """Optimizer parameter groups, common step, and grad-scaler tensors are valid common state."""
        full, filtered = self._schemas()
        optimizer_key = "optimizer.state.exp_avg.decoder.layers.0.adapter.weight"
        common_state_dict = {
            "optimizer": {
                "optimizer": {
                    "state": {"common_step": torch.tensor(7)},
                    "param_groups": [{"lr": 0.001}],
                },
                "grad_scaler": {"scale": torch.tensor([65536.0])},
            }
        }
        metadata = {
            "decoder.layers.0.adapter.weight": _make_sharded_tensor("decoder.layers.0.adapter.weight"),
            optimizer_key: _make_sharded_tensor(optimizer_key),
        }

        with patch(
            "megatron.bridge.training.checkpointing.dist_checkpointing_serialization.load_sharded_metadata",
            return_value=metadata,
        ):
            _validate_peft_run_resume_tensor_schema(
                full,
                filtered,
                "/checkpoint/iter_0000007",
                CheckpointType.GLOBAL,
                "torch_dist",
                common_state_dict=common_state_dict,
            )

    def test_allows_empty_common_model_containers(self):
        """DCP may retain empty model containers after extracting every sharded leaf."""
        full, filtered = self._schemas()
        optimizer_key = "optimizer.state.exp_avg.decoder.layers.0.adapter.weight"
        metadata = {
            "decoder.layers.0.adapter.weight": _make_sharded_tensor("decoder.layers.0.adapter.weight"),
            optimizer_key: _make_sharded_tensor(optimizer_key),
        }

        with patch(
            "megatron.bridge.training.checkpointing.dist_checkpointing_serialization.load_sharded_metadata",
            return_value=metadata,
        ):
            _validate_peft_run_resume_tensor_schema(
                full,
                filtered,
                "/checkpoint/iter_0000007",
                CheckpointType.GLOBAL,
                "torch_dist",
                common_state_dict={"model": {}, "model1": {"nested": [[], {}]}},
            )

    def test_accepts_exact_object_only_optimizer_schema(self):
        adapter = _make_sharded_tensor("decoder.layers.0.adapter.weight")
        optimizer_object = _make_sharded_object("optimizer.distributed.param_state")
        full = {"model": {"adapter": adapter}, "optimizer": {"param_state": optimizer_object}}
        filtered = {"model": {"adapter": adapter}, "optimizer": {"param_state": optimizer_object}}
        tensor_metadata = {adapter.key: adapter.without_data()}
        sharded_metadata = {**tensor_metadata, optimizer_object.unique_key: optimizer_object.without_data()}

        with (
            patch(
                "megatron.bridge.training.checkpointing.dist_checkpointing.load_tensors_metadata",
                return_value=tensor_metadata,
            ) as load_tensor_metadata,
            patch(
                "megatron.bridge.training.checkpointing.dist_checkpointing_serialization.load_sharded_metadata",
                return_value=sharded_metadata,
            ) as load_sharded_metadata,
        ):
            _validate_peft_run_resume_tensor_schema(
                full,
                filtered,
                "/checkpoint/iter_0000007",
                CheckpointType.GLOBAL,
                "torch_dist",
                common_state_dict={},
            )

        load_tensor_metadata.assert_not_called()
        load_sharded_metadata.assert_called_once_with("/checkpoint/iter_0000007")

    def test_accepts_mixed_tensor_and_object_optimizer_schema(self):
        adapter = _make_sharded_tensor("decoder.layers.0.adapter.weight")
        optimizer_tensor = _make_sharded_tensor("optimizer.state.exp_avg.decoder.layers.0.adapter.weight")
        optimizer_object = _make_sharded_object("optimizer.distributed.param_state")
        optimizer_state = {"exp_avg": optimizer_tensor, "param_state": optimizer_object}
        full = {"model": {"adapter": adapter}, "optimizer": optimizer_state}
        filtered = {"model": {"adapter": adapter}, "optimizer": optimizer_state}
        sharded_metadata = {
            adapter.key: adapter.without_data(),
            optimizer_tensor.key: optimizer_tensor.without_data(),
            optimizer_object.unique_key: optimizer_object.without_data(),
        }

        with patch(
            "megatron.bridge.training.checkpointing.dist_checkpointing_serialization.load_sharded_metadata",
            return_value=sharded_metadata,
        ):
            _validate_peft_run_resume_tensor_schema(
                full,
                filtered,
                "/checkpoint/iter_0000007",
                CheckpointType.GLOBAL,
                "torch_dist",
                common_state_dict={},
            )

    @pytest.mark.parametrize(
        "actual_object",
        [
            _make_sharded_object("optimizer.distributed.param_state", offset=1, fragments=2),
            _make_sharded_object("optimizer.distributed.param_state", offset=0, fragments=2),
        ],
    )
    def test_rejects_optimizer_object_global_metadata_mismatch(self, actual_object):
        adapter = _make_sharded_tensor("decoder.layers.0.adapter.weight")
        expected_object = _make_sharded_object("optimizer.distributed.param_state")
        full = {"model": {"adapter": adapter}, "optimizer": {"param_state": expected_object}}
        filtered = {"model": {"adapter": adapter}, "optimizer": {"param_state": expected_object}}
        sharded_metadata = {
            adapter.key: adapter.without_data(),
            actual_object.unique_key: actual_object.without_data(),
        }

        with (
            patch(
                "megatron.bridge.training.checkpointing.dist_checkpointing_serialization.load_sharded_metadata",
                return_value=sharded_metadata,
            ),
            pytest.raises(RuntimeError, match="missing optimizer object"),
        ):
            _validate_peft_run_resume_tensor_schema(
                full,
                filtered,
                "/checkpoint/iter_0000007",
                CheckpointType.GLOBAL,
                "torch_dist",
                common_state_dict={},
            )

    def test_rejects_missing_object_only_optimizer_state(self):
        adapter = _make_sharded_tensor("decoder.layers.0.adapter.weight")
        optimizer_object = _make_sharded_object("optimizer.distributed.param_state")
        full = {"model": {"adapter": adapter}, "optimizer": {"param_state": optimizer_object}}
        filtered = {"model": {"adapter": adapter}, "optimizer": {"param_state": optimizer_object}}
        tensor_metadata = {adapter.key: adapter.without_data()}

        with (
            patch(
                "megatron.bridge.training.checkpointing.dist_checkpointing.load_tensors_metadata",
                return_value=tensor_metadata,
            ),
            patch(
                "megatron.bridge.training.checkpointing.dist_checkpointing_serialization.load_sharded_metadata",
                return_value=tensor_metadata,
            ),
            pytest.raises(RuntimeError, match="missing optimizer object"),
        ):
            _validate_peft_run_resume_tensor_schema(
                full,
                filtered,
                "/checkpoint/iter_0000007",
                CheckpointType.GLOBAL,
                "torch_dist",
                common_state_dict={},
            )

    def test_object_only_optimizer_rejects_extra_optimizer_tensor(self):
        adapter = _make_sharded_tensor("decoder.layers.0.adapter.weight")
        optimizer_object = _make_sharded_object("optimizer.distributed.param_state")
        full = {"model": {"adapter": adapter}, "optimizer": {"param_state": optimizer_object}}
        filtered = {"model": {"adapter": adapter}, "optimizer": {"param_state": optimizer_object}}
        unexpected_key = "optimizer.state.exp_avg.decoder.layers.9.adapter.weight"
        tensor_metadata = {
            adapter.key: adapter.without_data(),
            unexpected_key: _make_sharded_tensor(unexpected_key).without_data(),
        }
        sharded_metadata = {**tensor_metadata, optimizer_object.unique_key: optimizer_object.without_data()}

        with (
            patch(
                "megatron.bridge.training.checkpointing.dist_checkpointing.load_tensors_metadata",
                return_value=tensor_metadata,
            ),
            patch(
                "megatron.bridge.training.checkpointing.dist_checkpointing_serialization.load_sharded_metadata",
                return_value=sharded_metadata,
            ),
            pytest.raises(RuntimeError, match="unrequested optimizer tensor"),
        ):
            _validate_peft_run_resume_tensor_schema(
                full,
                filtered,
                "/checkpoint/iter_0000007",
                CheckpointType.GLOBAL,
                "torch_dist",
                common_state_dict={},
            )

    @pytest.mark.unit
    def test_real_dcp_tensor_only_optimizer_rejects_unrequested_optimizer_object(
        self,
        single_rank_dcp_process_group,
        tmp_path: Path,
    ):
        """Object metadata must be checked even when the live optimizer has only tensors."""
        del single_rank_dcp_process_group
        checkpoint_dir = tmp_path / "tensor_optimizer_with_extra_object"
        checkpoint_dir.mkdir()

        adapter_key = "decoder.layers.0.adapter.weight"
        optimizer_key = "optimizer.state.exp_avg.decoder.layers.0.adapter.weight"
        unexpected_object = _make_sharded_object("optimizer.distributed.stale_param_state")
        mcore_dist_checkpointing.save(
            {
                "model": {"adapter": _make_sharded_tensor(adapter_key)},
                "optimizer": {
                    "exp_avg": _make_sharded_tensor(optimizer_key),
                    "stale_param_state": unexpected_object,
                },
            },
            str(checkpoint_dir),
        )

        tensor_metadata = mcore_dist_checkpointing.load_tensors_metadata(str(checkpoint_dir))
        sharded_metadata = mcore_dist_checkpointing_serialization.load_sharded_metadata(str(checkpoint_dir))
        assert unexpected_object.unique_key not in tensor_metadata
        assert unexpected_object.unique_key in sharded_metadata

        live_adapter = _make_sharded_tensor(adapter_key)
        live_optimizer = _make_sharded_tensor(optimizer_key)
        schema = {
            "model": {"adapter": live_adapter},
            "optimizer": {"exp_avg": live_optimizer},
        }
        with pytest.raises(
            RuntimeError,
            match=r"unrequested optimizer object.*optimizer\.distributed\.stale_param_state",
        ):
            _validate_peft_run_resume_tensor_schema(
                schema,
                schema,
                checkpoint_dir,
                CheckpointType.GLOBAL,
                "torch_dist",
                common_state_dict=mcore_dist_checkpointing.load_common_state_dict(str(checkpoint_dir)),
            )

    @pytest.mark.parametrize(
        ("checkpoint_type", "checkpoint_format"),
        [(CheckpointType.LOCAL, "torch_dist"), (CheckpointType.FSDP_DTENSOR, "fsdp_dtensor")],
    )
    def test_fails_closed_for_unsupported_checkpoint_formats(self, checkpoint_type, checkpoint_format):
        full, filtered = self._schemas()

        with (
            patch("megatron.bridge.training.checkpointing.dist_checkpointing.load_tensors_metadata") as load_metadata,
            pytest.raises(NotImplementedError, match="PEFT run-resume tensor validation"),
        ):
            _validate_peft_run_resume_tensor_schema(
                full,
                filtered,
                "/checkpoint/iter_0000007",
                checkpoint_type,
                checkpoint_format,
            )

        load_metadata.assert_not_called()

    def test_coordinates_rank_local_descriptor_errors_before_metadata_read(self):
        full, filtered = self._schemas()

        def gather_with_remote_error(output, local_payload):
            output[:] = [local_payload, {"error": "rank 1: factory expansion failed"}]

        with (
            patch("megatron.bridge.training.checkpointing.torch.distributed.is_initialized", return_value=True),
            patch("megatron.bridge.training.checkpointing.torch.distributed.get_rank", return_value=0),
            patch("megatron.bridge.training.checkpointing.torch.distributed.get_world_size", return_value=2),
            patch(
                "megatron.bridge.training.checkpointing.torch.distributed.all_gather_object",
                side_effect=gather_with_remote_error,
            ),
            patch("megatron.bridge.training.checkpointing.dist_checkpointing.load_tensors_metadata") as load_metadata,
            pytest.raises(RuntimeError, match="rank 1: factory expansion failed"),
        ):
            _validate_peft_run_resume_tensor_schema(
                full,
                filtered,
                "/checkpoint/iter_0000007",
                CheckpointType.GLOBAL,
                "torch_dist",
            )

        load_metadata.assert_not_called()

    def test_gathers_local_factory_failure_as_data(self):
        def fail_build(key, data, replica_id, flattened_range):
            del key, data, replica_id, flattened_range
            raise ValueError("invalid local factory")

        factory = ShardedTensorFactory("decoder.adapter", torch.empty(2, 3), fail_build, lambda state: state)
        schema = {"model": {"adapter": factory}}

        def gather_local_error(output, payload):
            output[0] = payload

        with (
            patch("megatron.bridge.training.checkpointing.torch.distributed.is_initialized", return_value=True),
            patch("megatron.bridge.training.checkpointing.torch.distributed.get_rank", return_value=0),
            patch("megatron.bridge.training.checkpointing.torch.distributed.get_world_size", return_value=1),
            patch(
                "megatron.bridge.training.checkpointing.torch.distributed.all_gather_object",
                side_effect=gather_local_error,
            ),
            patch("megatron.bridge.training.checkpointing.dist_checkpointing.load_tensors_metadata") as load_metadata,
            pytest.raises(RuntimeError, match="rank 0:.*invalid local factory"),
        ):
            _validate_peft_run_resume_tensor_schema(
                schema,
                schema,
                "/checkpoint/iter_0000007",
                CheckpointType.GLOBAL,
                "torch_dist",
            )

        load_metadata.assert_not_called()

    def test_broadcasts_rank_zero_metadata_failure(self):
        full, filtered = self._schemas()

        with (
            patch("megatron.bridge.training.checkpointing.torch.distributed.is_initialized", return_value=True),
            patch("megatron.bridge.training.checkpointing.torch.distributed.get_rank", return_value=0),
            patch("megatron.bridge.training.checkpointing.torch.distributed.get_world_size", return_value=1),
            patch(
                "megatron.bridge.training.checkpointing.torch.distributed.all_gather_object",
                side_effect=lambda output, payload: output.__setitem__(0, payload),
            ),
            patch("megatron.bridge.training.checkpointing.torch.distributed.broadcast_object_list") as broadcast,
            patch(
                "megatron.bridge.training.checkpointing.dist_checkpointing_serialization.load_sharded_metadata",
                side_effect=OSError("metadata unavailable"),
            ),
            pytest.raises(RuntimeError, match="rank 0 metadata read failed: metadata unavailable"),
        ):
            _validate_peft_run_resume_tensor_schema(
                full,
                filtered,
                "/checkpoint/iter_0000007",
                CheckpointType.GLOBAL,
                "torch_dist",
            )

        broadcast.assert_called_once()

    def test_unions_rank_local_pipeline_descriptors_before_validation(self):
        adapter0 = _make_sharded_tensor("decoder.layers.0.adapter.weight")
        full = {"model0": {"adapter": adapter0}}
        filtered = {"model0": {"adapter": adapter0}}
        remote_descriptor = ((2, 3), "torch.float32")

        def gather_with_remote_adapter(output, local_payload):
            output[:] = [
                local_payload,
                {
                    "error": None,
                    "model": {"decoder.layers.1.adapter.weight": remote_descriptor},
                    "adapter": {"decoder.layers.1.adapter.weight": remote_descriptor},
                    "optimizer": {},
                    "optimizer_objects": set(),
                    "optimizer_requested": False,
                },
            ]

        metadata = {
            "decoder.layers.0.adapter.weight": adapter0,
            "decoder.layers.1.adapter.weight": _make_sharded_tensor("decoder.layers.1.adapter.weight"),
        }
        with (
            patch("megatron.bridge.training.checkpointing.torch.distributed.is_initialized", return_value=True),
            patch("megatron.bridge.training.checkpointing.torch.distributed.get_rank", return_value=0),
            patch("megatron.bridge.training.checkpointing.torch.distributed.get_world_size", return_value=2),
            patch(
                "megatron.bridge.training.checkpointing.torch.distributed.all_gather_object",
                side_effect=gather_with_remote_adapter,
            ),
            patch("megatron.bridge.training.checkpointing.torch.distributed.broadcast_object_list"),
            patch(
                "megatron.bridge.training.checkpointing.dist_checkpointing.load_tensors_metadata",
                return_value=metadata,
            ),
        ):
            _validate_peft_run_resume_tensor_schema(
                full,
                filtered,
                "/checkpoint/iter_0000007",
                CheckpointType.GLOBAL,
                "torch_dist",
            )


class TestApplyPeftAdapterFilterToStateDict:
    """Test suite for apply_peft_adapter_filter_to_state_dict functionality.

    Tests the PEFT adapter filtering that processes complete checkpoint state dictionaries
    to retain only adapter parameters in model sections while preserving metadata.
    """

    @pytest.fixture
    def mock_peft_config(self):
        """Create a mock PEFT configuration."""
        return MockPEFT()

    @pytest.fixture
    def sample_complete_state_dict(self):
        """Create a sample complete state dict with all checkpoint components."""
        return {
            # Metadata
            "checkpoint_version": 3.0,
            "iteration": 1000,
            # Single model state
            "model": {
                # Base model parameters
                "embedding.weight": torch.randn(1000, 512),
                "layer1.linear.weight": torch.randn(512, 512),
                "layer2.attention.weight": torch.randn(512, 512),
                # Adapter parameters
                "layer1.adapter.weight": torch.randn(8, 512),
                "layer2.adapter.bias": torch.randn(8),
                "layer3.adapters.lora_A": torch.randn(8, 512),
                "layer3.adapters.lora_B": torch.randn(512, 8),
                # Base model output
                "output.weight": torch.randn(512, 1000),
            },
            # Optimizer state
            "optimizer": {
                "state": {},
                "param_groups": [],
            },
            # Scheduler state
            "opt_param_scheduler": {
                "lr": 0.001,
            },
            # RNG state
            "rng_state": [{"random_rng_state": "mock_state"}],
        }

    @pytest.fixture
    def sample_multi_model_state_dict(self):
        """Create a sample state dict with multiple model chunks (pipeline parallelism)."""
        return {
            # Metadata
            "checkpoint_version": 3.0,
            "iteration": 1000,
            # Multiple model states (pipeline parallelism)
            "model0": {
                "layer1.linear.weight": torch.randn(512, 512),
                "layer1.adapter.weight": torch.randn(8, 512),
            },
            "model1": {
                "layer2.attention.weight": torch.randn(512, 512),
                "layer2.adapter.bias": torch.randn(8),
            },
            "model2": {
                "layer3.output.weight": torch.randn(512, 1000),
                "layer3.adapters.lora_A": torch.randn(8, 512),
                "layer3.adapters.lora_B": torch.randn(512, 8),
            },
            # Optimizer and other states
            "optimizer": {"state": {}, "param_groups": []},
            "opt_param_scheduler": {"lr": 0.001},
            "rng_state": [{"random_rng_state": "mock_state"}],
        }

    def test_apply_peft_adapter_filter_single_model(self, mock_peft_config, sample_complete_state_dict):
        """Test filtering a complete state dict with a single model."""
        filtered_dict = apply_peft_adapter_filter_to_state_dict(sample_complete_state_dict, mock_peft_config)

        # Verify metadata is preserved
        assert filtered_dict["checkpoint_version"] == 3.0
        assert filtered_dict["iteration"] == 1000
        assert "optimizer" in filtered_dict
        assert "opt_param_scheduler" in filtered_dict
        assert "rng_state" in filtered_dict

        # Verify model state is filtered
        expected_adapter_keys = {
            "layer1.adapter.weight",
            "layer2.adapter.bias",
            "layer3.adapters.lora_A",
            "layer3.adapters.lora_B",
        }
        assert set(filtered_dict["model"].keys()) == expected_adapter_keys
        assert len(filtered_dict["model"]) == 4

        # Verify values are preserved correctly
        for key in expected_adapter_keys:
            assert torch.equal(filtered_dict["model"][key], sample_complete_state_dict["model"][key])

    def test_apply_peft_adapter_filter_drops_adapter_extra_state(self, mock_peft_config):
        """Adapter-only checkpoint filtering should not keep Transformer Engine extra state."""
        state_dict = {
            "checkpoint_version": 3.0,
            "model": {
                "layer1.adapter.weight": torch.randn(8, 512),
                "layer1.adapter._extra_state": "te-metadata",
                "layer2.adapters.lora_A": torch.randn(8, 512),
                "layer2.adapters.lora_A._extra_state": "te-metadata",
            },
        }

        filtered_dict = apply_peft_adapter_filter_to_state_dict(state_dict, mock_peft_config)

        assert set(filtered_dict["model"].keys()) == {
            "layer1.adapter.weight",
            "layer2.adapters.lora_A",
        }

    def test_apply_peft_adapter_filter_multi_model(self, mock_peft_config, sample_multi_model_state_dict):
        """Test filtering a complete state dict with multiple model chunks."""
        filtered_dict = apply_peft_adapter_filter_to_state_dict(sample_multi_model_state_dict, mock_peft_config)

        # Verify metadata is preserved
        assert filtered_dict["checkpoint_version"] == 3.0
        assert filtered_dict["iteration"] == 1000
        assert "optimizer" in filtered_dict
        assert "opt_param_scheduler" in filtered_dict
        assert "rng_state" in filtered_dict

        # Verify each model chunk is filtered correctly
        assert set(filtered_dict["model0"].keys()) == {"layer1.adapter.weight"}
        assert set(filtered_dict["model1"].keys()) == {"layer2.adapter.bias"}
        assert set(filtered_dict["model2"].keys()) == {"layer3.adapters.lora_A", "layer3.adapters.lora_B"}

        # Verify values are preserved correctly
        for model_key in ["model0", "model1", "model2"]:
            for param_key in filtered_dict[model_key].keys():
                assert torch.equal(
                    filtered_dict[model_key][param_key], sample_multi_model_state_dict[model_key][param_key]
                )

    def test_apply_peft_adapter_filter_no_adapters_found(self, mock_peft_config):
        """Test filtering when no adapter parameters match the filter."""
        state_dict_no_adapters = {
            "checkpoint_version": 3.0,
            "iteration": 1000,
            "model": {
                # Only base model parameters (no adapters)
                "embedding.weight": torch.randn(1000, 512),
                "layer1.linear.weight": torch.randn(512, 512),
                "output.weight": torch.randn(512, 1000),
            },
            "optimizer": {"state": {}, "param_groups": []},
        }

        filtered_dict = apply_peft_adapter_filter_to_state_dict(state_dict_no_adapters, mock_peft_config)

        # Model section should be empty since no adapters match
        assert filtered_dict["model"] == {}

        # Non-model sections should be preserved
        assert filtered_dict["checkpoint_version"] == 3.0
        assert filtered_dict["iteration"] == 1000
        assert "optimizer" in filtered_dict

    def test_apply_peft_adapter_filter_no_model_states(self, mock_peft_config):
        """Test filtering when no model states are present."""
        state_dict = {
            "checkpoint_version": 3.0,
            "iteration": 1000,
            "optimizer": {"state": {}, "param_groups": []},
            "rng_state": [{"random_rng_state": "mock_state"}],
        }

        filtered_dict = apply_peft_adapter_filter_to_state_dict(state_dict, mock_peft_config)

        # Should preserve all non-model keys
        assert filtered_dict == state_dict

    def test_apply_peft_adapter_filter_empty_model_states(self, mock_peft_config):
        """Test filtering when model states are empty."""
        state_dict = {
            "checkpoint_version": 3.0,
            "iteration": 1000,
            "model": {},
            "model0": {},
            "model1": {},
            "optimizer": {"state": {}, "param_groups": []},
        }

        filtered_dict = apply_peft_adapter_filter_to_state_dict(state_dict, mock_peft_config)

        # Model states should remain empty but present
        assert filtered_dict["model"] == {}
        assert filtered_dict["model0"] == {}
        assert filtered_dict["model1"] == {}
        assert filtered_dict["checkpoint_version"] == 3.0
        assert filtered_dict["iteration"] == 1000

    def test_apply_peft_adapter_filter_mixed_model_keys(self, mock_peft_config):
        """Test filtering with mixed model keys (some numerical, some not)."""
        state_dict = {
            "checkpoint_version": 3.0,
            "model": {"layer1.adapter.weight": torch.randn(8, 512)},
            "model0": {"layer2.adapter.bias": torch.randn(8)},
            "model5": {"layer3.adapters.lora_A": torch.randn(8, 512)},
            "model_not_digit": {"should.not.be.filtered": torch.randn(512)},  # Should not be filtered
            "modelabc": {"also.not.filtered": torch.randn(256)},  # Should not be filtered
            "optimizer": {"state": {}},
        }

        filtered_dict = apply_peft_adapter_filter_to_state_dict(state_dict, mock_peft_config)

        # Verify correct models are filtered
        assert set(filtered_dict["model"].keys()) == {"layer1.adapter.weight"}
        assert set(filtered_dict["model0"].keys()) == {"layer2.adapter.bias"}
        assert set(filtered_dict["model5"].keys()) == {"layer3.adapters.lora_A"}

        # Verify non-model keys are preserved unchanged
        assert filtered_dict["model_not_digit"] == state_dict["model_not_digit"]
        assert filtered_dict["modelabc"] == state_dict["modelabc"]
        assert filtered_dict["optimizer"] == state_dict["optimizer"]

    def test_apply_peft_adapter_filter_uses_correct_filter_function(self, sample_complete_state_dict):
        """Test that apply_peft_adapter_filter_to_state_dict uses the PEFT config's adapter_key_filter method."""

        # Create a custom PEFT config with specific filtering logic
        class CustomPEFT(PEFT):
            def __init__(self):
                self.allowed_keys = {"layer1.adapter.weight", "layer3.adapters.lora_A"}

            def transform(self, module, name=None, prefix=None):
                return module

            def adapter_key_filter(self, key):
                # Custom logic: only allow specific keys
                return key in self.allowed_keys

        custom_peft = CustomPEFT()

        filtered_dict = apply_peft_adapter_filter_to_state_dict(sample_complete_state_dict, custom_peft)

        # Should only contain the keys that the custom filter allows
        expected_keys = {"layer1.adapter.weight", "layer3.adapters.lora_A"}
        assert set(filtered_dict["model"].keys()) == expected_keys
        assert len(filtered_dict["model"]) == 2

        # Verify metadata is still preserved
        assert filtered_dict["checkpoint_version"] == 3.0
        assert filtered_dict["iteration"] == 1000


class TestPEFTCheckpointLoading:
    """Test suite for PEFT checkpoint loading functionality."""

    @pytest.fixture
    def mock_peft_config(self):
        """Create a mock PEFT configuration."""
        return MockPEFT()

    @pytest.fixture
    def sample_state_dict(self):
        """Create a sample state dict with mixed parameters."""
        return {
            # Base model parameters
            "embedding.weight": torch.randn(1000, 512),
            "layer1.linear.weight": torch.randn(512, 512),
            "layer2.attention.weight": torch.randn(512, 512),
            # Adapter parameters
            "layer1.adapter.weight": torch.randn(8, 512),
            "layer2.adapter.bias": torch.randn(8),
            "layer3.adapters.lora_A": torch.randn(8, 512),
            "layer3.adapters.lora_B": torch.randn(512, 8),
            # Base model output
            "output.weight": torch.randn(512, 1000),
        }

    def test_apply_peft_adapter_filter_uses_adapter_key_filter(self):
        """Test that apply_peft_adapter_filter_to_state_dict correctly uses PEFT's adapter_key_filter method."""
        # Create sample complete state dict with mixed parameters
        sample_complete_state_dict = {
            "checkpoint_version": 3.0,
            "iteration": 1000,
            "model": {
                # Base model parameters
                "embedding.weight": torch.randn(1000, 512),
                "layer1.linear.weight": torch.randn(512, 512),
                "layer2.attention.weight": torch.randn(512, 512),
                # Adapter parameters
                "layer1.adapter.weight": torch.randn(8, 512),
                "layer2.adapter.bias": torch.randn(8),
                "layer3.adapters.lora_A": torch.randn(8, 512),
                "layer3.adapters.lora_B": torch.randn(512, 8),
                # Base model output
                "output.weight": torch.randn(512, 1000),
            },
            "optimizer": {"state": {}, "param_groups": []},
        }

        # Create a custom PEFT config with specific filtering logic
        class CustomPEFT(PEFT):
            def __init__(self):
                self.custom_adapter_keys = {"layer1.adapter.weight", "layer3.adapters.lora_A"}

            def transform(self, module, name=None, prefix=None):
                return module

            def adapter_key_filter(self, key):
                # Custom logic: only allow specific keys
                return key in self.custom_adapter_keys

        custom_peft = CustomPEFT()

        filtered_dict = apply_peft_adapter_filter_to_state_dict(sample_complete_state_dict, custom_peft)

        # Should only contain the keys that the custom filter allows in the model section
        expected_model_keys = {"layer1.adapter.weight", "layer3.adapters.lora_A"}
        assert set(filtered_dict["model"].keys()) == expected_model_keys
        assert len(filtered_dict["model"]) == 2

        # Verify values are preserved correctly
        for key in expected_model_keys:
            assert torch.equal(filtered_dict["model"][key], sample_complete_state_dict["model"][key])

        # Verify non-model sections are preserved
        assert filtered_dict["checkpoint_version"] == 3.0
        assert filtered_dict["iteration"] == 1000
        assert "optimizer" in filtered_dict

    @patch("megatron.bridge.training.checkpointing._load_base_checkpoint")
    @patch("megatron.bridge.training.checkpointing.checkpoint_exists")
    @patch("megatron.bridge.training.checkpointing.apply_peft_adapter_filter_to_state_dict")
    @patch("megatron.bridge.training.checkpointing.generate_state_dict")
    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.checkpointing._validate_peft_run_resume_tensor_schema")
    def test_load_checkpoint_peft_resume_detection(
        self,
        mock_validate_schema,
        mock_dist_ckpt,
        mock_generate_state_dict,
        mock_filter,
        mock_checkpoint_exists,
        mock_load_base,
    ):
        """Test that PEFT resume is properly detected and triggers filtering."""
        # Setup mocks
        mock_checkpoint_exists.return_value = True

        # Create deterministic tensors to avoid random comparison issues
        torch.manual_seed(42)
        base_weight = torch.randn(512, 512)
        adapter_weight = torch.randn(8, 512)

        # This is what generate_state_dict would return (full state dict)
        full_generated_state_dict = {
            "model": {
                "layer1.linear.weight": base_weight,
                "layer1.adapter.weight": adapter_weight,
            },
            "checkpoint_version": 3.0,
        }

        # This is what apply_peft_adapter_filter_to_state_dict should return (filtered)
        filtered_sharded_state_dict = {
            "model": {"layer1.adapter.weight": adapter_weight},
            "checkpoint_version": 3.0,
        }
        mock_filter.return_value = filtered_sharded_state_dict

        common_state_dict = {"checkpoint_version": 3.0}

        # The first pass returns common state only; the second pass returns the
        # tensors requested by the filtered sharded schema.
        def load_base_by_pass(*args, rank0=False, **kwargs):
            del args, kwargs
            loaded = common_state_dict if rank0 else filtered_sharded_state_dict
            return loaded, "/path/to/checkpoint", False, CheckpointType.GLOBAL

        mock_load_base.side_effect = load_base_by_pass

        def validate_before_second_load(*args, **kwargs):
            del args, kwargs
            assert mock_load_base.call_count == 1

        mock_validate_schema.side_effect = validate_before_second_load

        # Create mock global state for PEFT resume scenario
        mock_state = Mock(spec=GlobalState)
        mock_cfg = Mock(spec=ConfigContainer)
        mock_cfg.peft = MockPEFT()
        mock_cfg.checkpoint = Mock(spec=CheckpointConfig)
        mock_cfg.checkpoint.pretrained_checkpoint = "/path/to/pretrained"
        mock_cfg.checkpoint.load = "/path/to/checkpoint"
        mock_cfg.checkpoint.finetune = False
        mock_cfg.checkpoint.load_rng = False  # Disable RNG loading for focused testing
        mock_cfg.checkpoint.load_optim = False  # Disable optimizer loading for focused testing

        # Add necessary model config attributes
        mock_cfg.model = Mock()
        mock_cfg.model.tensor_model_parallel_size = 1
        mock_cfg.model.pipeline_model_parallel_size = 1
        mock_cfg.checkpoint.auto_detect_ckpt_format = False
        mock_cfg.checkpoint.ckpt_format = "torch_dist"
        mock_cfg.checkpoint.non_persistent_save_interval = None
        mock_cfg.dist = Mock()
        mock_cfg.dist.use_decentralized_pg = False
        mock_state.cfg = mock_cfg
        mock_state.train_state = Mock()
        mock_state.train_state.consumed_train_samples = 0
        mock_state.train_state.skipped_train_samples = 0
        mock_state.train_state.consumed_valid_samples = 0
        mock_state.train_state.step = 1000  # Set to integer for comparisons
        mock_state.train_state.floating_point_operations_so_far = 50000
        mock_cfg.ddp = Mock()
        mock_cfg.ddp.use_megatron_fsdp = False

        # Mock dist_checkpointing
        mock_dist_ckpt.load_content_metadata.return_value = {}
        mock_dist_ckpt.load.return_value = {}

        # Create mock model
        mock_model = [Mock()]
        mock_model[0].load_state_dict = Mock()

        # Call load_checkpoint
        with (
            patch("megatron.bridge.training.checkpointing.read_train_state") as mock_read_train_state,
            patch("megatron.bridge.training.checkpointing.get_checkpoint_train_state_filename"),
            patch("megatron.bridge.training.checkpointing.update_num_microbatches"),
            patch("megatron.bridge.training.checkpointing.get_checkpoint_version") as mock_get_version,
            patch("megatron.bridge.training.checkpointing.set_checkpoint_version"),
            patch("torch.distributed.barrier"),
            patch("megatron.bridge.training.checkpointing.print_rank_0"),
            patch("megatron.bridge.training.checkpointing.read_run_config") as mock_read_run_config,
            patch("megatron.bridge.training.checkpointing.unwrap_model") as mock_unwrap_model,
            patch("megatron.bridge.training.checkpointing.get_pg_collection") as mock_get_pg_collection,
            patch("os.path.exists") as mock_exists,
        ):
            mock_read_train_state.return_value = mock_state.train_state
            mock_get_version.return_value = 3.0
            mock_unwrap_model.return_value = mock_model

            # Create mock pg_collection
            mock_pg_collection = Mock()
            mock_pg_collection.tp.rank.return_value = 0
            mock_pg_collection.tp.size.return_value = 1
            mock_pg_collection.pp.rank.return_value = 0
            mock_pg_collection.pp.size.return_value = 1
            mock_pg_collection.dp_cp.rank.return_value = 0
            mock_get_pg_collection.return_value = mock_pg_collection

            # Mock file existence - run_config.yaml exists, train_state.pt doesn't (to use read_train_state mock)
            def mock_exists_side_effect(path):
                if "run_config.yaml" in path:
                    return True  # run_config.yaml exists
                elif "train_state.pt" in path:
                    return False  # train_state.pt doesn't exist, use mock
                return False

            mock_exists.side_effect = mock_exists_side_effect

            # Mock generate_state_dict to return the full state dict (before filtering)
            mock_generate_state_dict.return_value = full_generated_state_dict

            # Mock run config for non-PEFT scenario
            mock_run_config = {
                "model": {
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                },
                "checkpoint": {"save_rng": True, "save_optim": True, "fully_parallel_save": True},
            }
            mock_read_run_config.return_value = mock_run_config

            with patch("megatron.bridge.training.checkpointing._load_model_state_dict") as mock_load_model_state_dict:
                _ = load_checkpoint(
                    mock_state,
                    mock_model,
                    None,  # No optimizer
                    None,  # No scheduler
                    strict=True,
                    checkpointing_context={},
                    skip_load_to_model_and_opt=False,
                )

            # Verify PEFT filtering was called on the generated sharded state dict
            mock_filter.assert_called_once_with(full_generated_state_dict, mock_cfg.peft)
            mock_validate_schema.assert_called_once_with(
                full_generated_state_dict,
                filtered_sharded_state_dict,
                "/path/to/checkpoint",
                CheckpointType.GLOBAL,
                "torch_dist",
                common_state_dict=common_state_dict,
            )

            # PEFT run-checkpoint resume is explicit. Quantized loaders must not
            # guess adapter-only semantics from strict=False, because inference
            # also uses non-strict full-base loads.
            mock_load_model_state_dict.assert_called_once_with(
                mock_model[0],
                filtered_sharded_state_dict["model"],
                False,
                adapter_only=True,
            )

            mock_load_base.reset_mock()
            mock_validate_schema.reset_mock()
            mock_load_model_state_dict.reset_mock()
            mock_validate_schema.side_effect = RuntimeError("missing adapter tensor")

            with pytest.raises(RuntimeError, match="missing adapter tensor"):
                load_checkpoint(
                    mock_state,
                    mock_model,
                    None,
                    None,
                    strict=True,
                    checkpointing_context={},
                    skip_load_to_model_and_opt=False,
                )

            assert mock_load_base.call_count == 1
            mock_load_model_state_dict.assert_not_called()

    @pytest.mark.parametrize(
        ("checkpoint_format", "checkpoint_type", "checkpoint_name"),
        [
            ("torch_dist", CheckpointType.LOCAL, (7, 0)),
            ("fsdp_dtensor", CheckpointType.FSDP_DTENSOR, "/path/to/checkpoint/iter_0000007"),
        ],
    )
    def test_peft_resume_preserves_non_global_checkpoint_load_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
        checkpoint_format: str,
        checkpoint_type: CheckpointType,
        checkpoint_name: object,
    ) -> None:
        """Local and FSDP PEFT resumes retain their adapter-filtered load path."""
        base_weight = torch.full((2, 2), 1.0)
        adapter_weight = torch.full((2, 2), 2.0)
        full_sharded_state_dict = {
            "checkpoint_version": 3.0,
            "model": {
                "layer1.linear.weight": base_weight,
                "layer1.adapter.weight": adapter_weight,
            },
        }
        requested_model_keys: list[set[str]] = []

        def load_base_checkpoint(*args, rank0=False, sharded_state_dict=None, **kwargs):
            del args, kwargs
            if rank0:
                assert checkpoint_type == CheckpointType.LOCAL
                return {"checkpoint_version": 3.0}, checkpoint_name, False, checkpoint_type

            requested_model_keys.append(set(sharded_state_dict["model"]))
            return (
                {
                    "checkpoint_version": 3.0,
                    "iteration": 0,
                },
                checkpoint_name,
                False,
                checkpoint_type,
            )

        checkpoint = CheckpointConfig(
            ckpt_format=checkpoint_format,
            load="/path/to/checkpoint/iter_0000007",
            load_optim=False,
            load_rng=False,
            save_optim=False,
            save_rng=False,
        )
        cfg = SimpleNamespace(
            checkpoint=checkpoint,
            ddp=SimpleNamespace(use_megatron_fsdp=checkpoint_format == "fsdp_dtensor"),
            model=SimpleNamespace(
                bf16=False,
                fp16=False,
                pipeline_model_parallel_size=1,
                tensor_model_parallel_size=1,
            ),
            optimizer=SimpleNamespace(use_distributed_optimizer=False),
            peft=MockPEFT(),
            rng=SimpleNamespace(data_parallel_random_init=False),
        )
        state = SimpleNamespace(
            cfg=cfg,
            comet_logger=None,
            mlflow_logger=None,
            train_state=None,
            wandb_logger=None,
        )
        rank_group = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
        pg_collection = SimpleNamespace(
            dp=rank_group,
            dp_cp=object(),
            pp=rank_group,
            tp=rank_group,
        )
        reader = SimpleNamespace(read_metadata=lambda: SimpleNamespace(state_dict_metadata={}))

        monkeypatch.setattr(checkpointing, "_get_filesystem_reader", lambda path: reader)
        monkeypatch.setattr(checkpointing, "_load_base_checkpoint", load_base_checkpoint)
        monkeypatch.setattr(checkpointing, "_restore_modelopt_state_before_sharded_schema", lambda *args: None)
        monkeypatch.setattr(checkpointing, "file_exists", lambda path: False)
        monkeypatch.setattr(checkpointing, "generate_state_dict", lambda *args, **kwargs: full_sharded_state_dict)
        monkeypatch.setattr(checkpointing, "get_checkpoint_version", lambda: 3.0)
        monkeypatch.setattr(checkpointing, "is_checkpoint_iteration_directory", lambda path: True)
        monkeypatch.setattr(checkpointing, "print_rank_0", lambda *args, **kwargs: None)
        monkeypatch.setattr(checkpointing, "set_checkpoint_version", lambda version: None)
        monkeypatch.setattr(checkpointing, "update_num_microbatches", lambda **kwargs: None)

        result = load_checkpoint(
            state,
            [nn.Module()],
            optimizer=None,
            opt_param_scheduler=None,
            checkpointing_context={},
            skip_load_to_model_and_opt=True,
            pg_collection=pg_collection,
        )

        assert result == (0, 0)
        assert requested_model_keys == [{"layer1.adapter.weight"}]

    @patch("megatron.bridge.training.checkpointing._load_base_checkpoint")
    @patch("megatron.bridge.training.checkpointing.checkpoint_exists")
    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    def test_load_checkpoint_non_peft_regular_loading(self, mock_dist_ckpt, mock_checkpoint_exists, mock_load_base):
        """Test that non-PEFT scenarios use regular loading without filtering."""
        # Setup mocks
        mock_checkpoint_exists.return_value = True

        torch.manual_seed(44)
        linear_weight_1 = torch.randn(512, 512)
        linear_weight_2 = torch.randn(512, 512)

        mock_state_dict = {
            "model": {
                "layer1.linear.weight": linear_weight_1,
                "layer2.linear.weight": linear_weight_2,
            },
            "checkpoint_version": 3.0,
        }
        mock_load_base.return_value = (mock_state_dict, "/path/to/checkpoint", False, None)

        # Create mock global state for non-PEFT scenario
        mock_state = Mock(spec=GlobalState)
        mock_cfg = Mock(spec=ConfigContainer)
        mock_cfg.peft = None  # No PEFT
        mock_cfg.checkpoint = Mock(spec=CheckpointConfig)
        mock_cfg.checkpoint.pretrained_checkpoint = None
        mock_cfg.checkpoint.load = "/path/to/checkpoint"
        mock_cfg.checkpoint.finetune = False
        mock_cfg.checkpoint.load_rng = False  # Disable RNG loading for focused testing
        mock_cfg.checkpoint.load_optim = False  # Disable optimizer loading for focused testing

        # Add necessary model config attributes
        mock_cfg.model = Mock()
        mock_cfg.model.tensor_model_parallel_size = 1
        mock_cfg.model.pipeline_model_parallel_size = 1
        mock_cfg.checkpoint.auto_detect_ckpt_format = False
        mock_cfg.checkpoint.ckpt_format = "torch_dist"
        mock_cfg.checkpoint.non_persistent_save_interval = None
        mock_cfg.dist = Mock()
        mock_cfg.dist.use_decentralized_pg = False
        mock_state.cfg = mock_cfg
        mock_state.train_state = Mock()
        mock_state.train_state.consumed_train_samples = 0
        mock_state.train_state.skipped_train_samples = 0
        mock_state.train_state.consumed_valid_samples = 0
        mock_state.train_state.step = 1000  # Set to integer for comparisons
        mock_state.train_state.floating_point_operations_so_far = 50000
        mock_cfg.ddp = Mock()
        mock_cfg.ddp.use_megatron_fsdp = False

        # Mock dist_checkpointing
        mock_dist_ckpt.load_content_metadata.return_value = {}
        mock_dist_ckpt.load.return_value = {}

        # Create mock model
        mock_model = [Mock()]
        mock_model[0].load_state_dict = Mock()

        # Call load_checkpoint
        with (
            patch("megatron.bridge.training.checkpointing.read_train_state") as mock_read_train_state,
            patch("megatron.bridge.training.checkpointing.get_checkpoint_train_state_filename"),
            patch("megatron.bridge.training.checkpointing.update_num_microbatches"),
            patch("megatron.bridge.training.checkpointing.get_checkpoint_version") as mock_get_version,
            patch("megatron.bridge.training.checkpointing.set_checkpoint_version"),
            patch("torch.distributed.barrier"),
            patch("megatron.bridge.training.checkpointing.print_rank_0"),
            patch("megatron.bridge.training.checkpointing.read_run_config") as mock_read_run_config,
            patch("megatron.bridge.training.checkpointing.unwrap_model") as mock_unwrap_model,
            patch("megatron.bridge.training.checkpointing.get_pg_collection") as mock_get_pg_collection,
            patch("os.path.exists") as mock_exists,
        ):
            mock_read_train_state.return_value = mock_state.train_state
            mock_get_version.return_value = 3.0
            mock_unwrap_model.return_value = mock_model

            # Create mock pg_collection
            mock_pg_collection = Mock()
            mock_pg_collection.tp.rank.return_value = 0
            mock_pg_collection.tp.size.return_value = 1
            mock_pg_collection.pp.rank.return_value = 0
            mock_pg_collection.pp.size.return_value = 1
            mock_pg_collection.dp_cp.rank.return_value = 0
            mock_get_pg_collection.return_value = mock_pg_collection

            # Mock file existence - run_config.yaml exists, train_state.pt doesn't (to use read_train_state mock)
            def mock_exists_side_effect(path):
                if "run_config.yaml" in path:
                    return True  # run_config.yaml exists
                elif "train_state.pt" in path:
                    return False  # train_state.pt doesn't exist, use mock
                return False

            mock_exists.side_effect = mock_exists_side_effect

            # Mock run config for non-PEFT scenario
            mock_run_config = {
                "model": {
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                },
                "checkpoint": {"save_rng": True, "save_optim": True, "fully_parallel_save": True},
            }
            mock_read_run_config.return_value = mock_run_config

            _ = load_checkpoint(
                mock_state,
                mock_model,
                None,  # No optimizer
                None,  # No scheduler
                strict=True,
                checkpointing_context={},
                skip_load_to_model_and_opt=False,
            )

            # Verify model.load_state_dict was called with full dict and original strict value
            mock_model[0].load_state_dict.assert_called_once_with(mock_state_dict["model"], strict=True)

    @patch("megatron.bridge.training.checkpointing._load_base_checkpoint")
    @patch("megatron.bridge.training.checkpointing.checkpoint_exists")
    @patch("megatron.bridge.training.checkpointing.apply_peft_adapter_filter_to_state_dict")
    @patch("megatron.bridge.training.checkpointing.generate_state_dict")
    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    @patch("megatron.bridge.training.checkpointing._validate_peft_run_resume_tensor_schema")
    def test_load_checkpoint_peft_resume_multi_model(
        self,
        mock_validate_schema,
        mock_dist_ckpt,
        mock_generate_state_dict,
        mock_filter,
        mock_checkpoint_exists,
        mock_load_base,
    ):
        """Test PEFT resume with multiple model chunks (pipeline parallelism)."""
        # Setup mocks
        mock_checkpoint_exists.return_value = True

        torch.manual_seed(43)
        base_weight_0 = torch.randn(512, 512)
        adapter_weight_0 = torch.randn(8, 512)
        base_weight_1 = torch.randn(512, 512)
        adapter_weight_1 = torch.randn(8, 512)

        # This is what generate_state_dict would return (full state dict)
        full_generated_state_dict = {
            "model0": {
                "layer1.linear.weight": base_weight_0,
                "layer1.adapter.weight": adapter_weight_0,
            },
            "model1": {
                "layer2.linear.weight": base_weight_1,
                "layer2.adapter.weight": adapter_weight_1,
            },
            "checkpoint_version": 3.0,
        }

        # This is what apply_peft_adapter_filter_to_state_dict should return (filtered)
        filtered_sharded_state_dict = {
            "model0": {"layer1.adapter.weight": adapter_weight_0},
            "model1": {"layer2.adapter.weight": adapter_weight_1},
            "checkpoint_version": 3.0,
        }
        mock_filter.return_value = filtered_sharded_state_dict

        common_state_dict = {"checkpoint_version": 3.0}

        def load_base_by_pass(*args, rank0=False, **kwargs):
            del args, kwargs
            loaded = common_state_dict if rank0 else filtered_sharded_state_dict
            return loaded, "/path/to/checkpoint", False, CheckpointType.GLOBAL

        mock_load_base.side_effect = load_base_by_pass

        def validate_before_second_load(*args, **kwargs):
            del args, kwargs
            assert mock_load_base.call_count == 1

        mock_validate_schema.side_effect = validate_before_second_load

        # Create mock global state for PEFT resume scenario
        mock_state = Mock(spec=GlobalState)
        mock_cfg = Mock(spec=ConfigContainer)
        mock_cfg.peft = MockPEFT()
        mock_cfg.checkpoint = Mock(spec=CheckpointConfig)
        mock_cfg.checkpoint.pretrained_checkpoint = "/path/to/pretrained"
        mock_cfg.checkpoint.load = "/path/to/checkpoint"
        mock_cfg.checkpoint.finetune = False
        mock_cfg.checkpoint.load_rng = False  # Disable RNG loading for focused testing
        mock_cfg.checkpoint.load_optim = False  # Disable optimizer loading for focused testing

        # Add necessary model config attributes
        mock_cfg.model = Mock()
        mock_cfg.model.tensor_model_parallel_size = 1
        mock_cfg.model.pipeline_model_parallel_size = 1
        mock_cfg.checkpoint.auto_detect_ckpt_format = False
        mock_cfg.checkpoint.ckpt_format = "torch_dist"
        mock_cfg.checkpoint.non_persistent_save_interval = None
        mock_cfg.dist = Mock()
        mock_cfg.dist.use_decentralized_pg = False
        mock_state.cfg = mock_cfg
        mock_state.train_state = Mock()
        mock_state.train_state.consumed_train_samples = 0
        mock_state.train_state.skipped_train_samples = 0
        mock_state.train_state.consumed_valid_samples = 0
        mock_state.train_state.step = 1000  # Set to integer for comparisons
        mock_state.train_state.floating_point_operations_so_far = 50000
        mock_cfg.ddp = Mock()
        mock_cfg.ddp.use_megatron_fsdp = False

        # Mock dist_checkpointing
        mock_dist_ckpt.load_content_metadata.return_value = {}
        mock_dist_ckpt.load.return_value = {}

        # Create mock models (2 chunks for pipeline parallelism)
        mock_model = [Mock(), Mock()]
        mock_model[0].load_state_dict = Mock()
        mock_model[1].load_state_dict = Mock()

        # Call load_checkpoint
        with (
            patch("megatron.bridge.training.checkpointing.read_train_state") as mock_read_train_state,
            patch("megatron.bridge.training.checkpointing.get_checkpoint_train_state_filename"),
            patch("megatron.bridge.training.checkpointing.update_num_microbatches"),
            patch("megatron.bridge.training.checkpointing.get_checkpoint_version") as mock_get_version,
            patch("megatron.bridge.training.checkpointing.set_checkpoint_version"),
            patch("megatron.core.mpu.set_virtual_pipeline_model_parallel_rank"),
            patch("torch.distributed.barrier"),
            patch("megatron.bridge.training.checkpointing.print_rank_0"),
            patch("megatron.bridge.training.checkpointing.read_run_config") as mock_read_run_config,
            patch("megatron.bridge.training.checkpointing.unwrap_model") as mock_unwrap_model,
            patch("megatron.bridge.training.checkpointing.get_pg_collection") as mock_get_pg_collection,
            patch("megatron.bridge.training.checkpointing._load_model_state_dict") as mock_load_model_state_dict,
            patch("os.path.exists") as mock_exists,
        ):
            mock_read_train_state.return_value = mock_state.train_state
            mock_get_version.return_value = 3.0
            mock_unwrap_model.return_value = mock_model

            # Create mock pg_collection
            mock_pg_collection = Mock()
            mock_pg_collection.tp.rank.return_value = 0
            mock_pg_collection.tp.size.return_value = 1
            mock_pg_collection.pp.rank.return_value = 0
            mock_pg_collection.pp.size.return_value = 1
            mock_pg_collection.dp_cp.rank.return_value = 0
            mock_get_pg_collection.return_value = mock_pg_collection

            # Mock file existence - run_config.yaml exists, train_state.pt doesn't (to use read_train_state mock)
            def mock_exists_side_effect(path):
                if "run_config.yaml" in path:
                    return True  # run_config.yaml exists
                elif "train_state.pt" in path:
                    return False  # train_state.pt doesn't exist, use mock
                return False

            mock_exists.side_effect = mock_exists_side_effect

            # Mock generate_state_dict to return the full state dict (before filtering)
            mock_generate_state_dict.return_value = full_generated_state_dict

            # Mock run config for multi-model PEFT scenario
            mock_run_config = {
                "model": {
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                },
                "checkpoint": {"save_rng": True, "save_optim": True, "fully_parallel_save": True},
            }
            mock_read_run_config.return_value = mock_run_config

            _ = load_checkpoint(
                mock_state,
                mock_model,
                None,  # No optimizer
                None,  # No scheduler
                strict=True,
                checkpointing_context={},
                skip_load_to_model_and_opt=False,
            )

            # Verify filtering was called once with the complete state dict
            mock_filter.assert_called_once_with(full_generated_state_dict, mock_cfg.peft)
            mock_validate_schema.assert_called_once_with(
                full_generated_state_dict,
                filtered_sharded_state_dict,
                "/path/to/checkpoint",
                CheckpointType.GLOBAL,
                "torch_dist",
                common_state_dict=common_state_dict,
            )

            # Every pipeline chunk receives the same explicit adapter-only signal.
            assert mock_load_model_state_dict.call_args_list == [
                ((mock_model[0], filtered_sharded_state_dict["model0"], False), {"adapter_only": True}),
                ((mock_model[1], filtered_sharded_state_dict["model1"], False), {"adapter_only": True}),
            ]


class TestPEFTCheckpointingIntegration:
    """Integration tests using real GPT models and LoRA PEFT configurations."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown_parallel_state(self):
        """Setup and teardown parallel state for Megatron tests."""

        if not dist.is_initialized():
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29500"
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"

            device_count = torch.cuda.device_count()
            if device_count > 0:
                torch.cuda.set_device(0)

            init_process_group_kwargs = {
                "backend": "nccl" if device_count > 0 else "gloo",
                "world_size": 1,
                "rank": 0,
                "timeout": datetime.timedelta(minutes=30),
            }

            dist.init_process_group(**init_process_group_kwargs)

        assert dist.is_initialized(), "Distributed backend not initialized"

        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                virtual_pipeline_model_parallel_size=None,
                context_parallel_size=1,
            )

        assert parallel_state.model_parallel_is_initialized(), "Model parallel not initialized"

        from megatron.core.process_groups_config import ProcessGroupCollection

        from megatron.bridge.training.initialize import _set_random_seed

        # Create pg_collection from initialized mpu
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        _set_random_seed(
            seed_=1234,
            data_parallel_random_init=False,
            te_rng_tracker=True,
            inference_rng_tracker=False,
            pg_collection=pg_collection,
        )

        yield

        try:
            if parallel_state.model_parallel_is_initialized():
                parallel_state.destroy_model_parallel()
            if dist.is_initialized():
                dist.destroy_process_group()
                # Clean up environment variables
                for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE"):
                    os.environ.pop(key, None)
        except (NameError, AttributeError, RuntimeError):
            pass

    @pytest.fixture
    def gpt_model_and_config(self):
        """Create a minimal GPT model with Megatron modules for integration testing."""

        # Create minimal GPT model provider for testing
        model_provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            seq_length=64,
            vocab_size=256,
            ffn_hidden_size=256,
        )

        from megatron.core.process_groups_config import ProcessGroupCollection

        model_provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        # Create LoRA PEFT config
        lora_config = LoRA(
            target_modules=["linear_qkv", "linear_proj"],
            dim=8,
            alpha=16,
            dropout=0.1,
        )

        return model_provider, lora_config

    def _create_lora_pre_wrap_hook(self, lora_config: LoRA):
        """Create a pre-wrap hook that applies LoRA to the model.

        Args:
            lora_config: LoRA configuration instance

        Returns:
            A callable hook that can be registered with the model provider
        """

        def lora_pre_wrap_hook(model: list[MegatronModule]) -> list[MegatronModule]:
            """Pre-wrap hook that applies LoRA transformation.

            Args:
                model: List of base model modules before distributed wrapping

            Returns:
                List of LoRA-transformed model modules
            """
            return lora_config(model, training=True)

        return lora_pre_wrap_hook

    def test_apply_peft_adapter_filter_integration_with_peft(self, gpt_model_and_config):
        """Test apply_peft_adapter_filter_to_state_dict with real GPT model and LoRA PEFT."""
        model_provider, lora_config = gpt_model_and_config

        # Register LoRA pre-wrap hook and get model with PEFT applied
        lora_hook = self._create_lora_pre_wrap_hook(lora_config)
        model_provider.register_pre_wrap_hook(lora_hook)
        model_provider.finalize()
        peft_model = model_provider.provide_distributed_model(ddp_config=None, wrap_with_ddp=False)

        # Verify we got Megatron modules
        assert isinstance(peft_model, list)
        assert len(peft_model) > 0
        assert all(isinstance(chunk, MegatronModule) for chunk in peft_model)

        # Move to CUDA
        peft_model = [chunk.cuda() for chunk in peft_model]

        # Set up params_to_save for the PEFT config
        lora_config.set_params_to_save(peft_model)

        # Create a complete state dict as it would appear in checkpointing
        complete_state_dict = {
            "checkpoint_version": 3.0,
            "iteration": 1000,
            "model": peft_model[0].sharded_state_dict(),
            "optimizer": {"state": {}, "param_groups": []},
            "rng_state": [{"random_rng_state": "mock_state"}],
        }

        # Filter for adapter parameters only using the main function
        filtered_state_dict = apply_peft_adapter_filter_to_state_dict(complete_state_dict, lora_config)

        # Verify filtering worked on the model section
        assert len(filtered_state_dict["model"]) < len(complete_state_dict["model"]), (
            f"Filtered model state dict ({len(filtered_state_dict['model'])}) should be smaller than "
            f"full model state dict ({len(complete_state_dict['model'])})"
        )

        # Verify only adapter parameters are in filtered model state dict
        for param_name in filtered_state_dict["model"].keys():
            assert lora_config.adapter_key_filter(param_name), (
                f"Parameter '{param_name}' should not be in filtered state dict"
            )

        # Verify some adapter parameters were found
        assert len(filtered_state_dict["model"]) > 0, "No adapter parameters found in filtered state dict"

        # Check that adapter parameters have expected naming patterns
        adapter_param_names = list(filtered_state_dict["model"].keys())
        has_lora_params = any("lora" in name.lower() or "adapter" in name.lower() for name in adapter_param_names)

        assert has_lora_params, f"Expected LoRA or adapter parameters in {adapter_param_names}"

        # Verify non-model sections are preserved
        assert filtered_state_dict["checkpoint_version"] == 3.0
        assert filtered_state_dict["iteration"] == 1000
        assert "optimizer" in filtered_state_dict
        assert "rng_state" in filtered_state_dict

    def test_apply_peft_adapter_filter_integration(self, gpt_model_and_config):
        """Test apply_peft_adapter_filter_to_state_dict with real model state dict."""
        model_provider, lora_config = gpt_model_and_config

        # Register LoRA pre-wrap hook and get model with PEFT applied
        lora_hook = self._create_lora_pre_wrap_hook(lora_config)
        model_provider.register_pre_wrap_hook(lora_hook)
        model_provider.finalize()
        peft_model = model_provider.provide_distributed_model(ddp_config=None, wrap_with_ddp=False)
        peft_model = [chunk.cuda() for chunk in peft_model]
        lora_config.set_params_to_save(peft_model)

        # Create a realistic complete state dict
        complete_state_dict = {
            "checkpoint_version": 3.0,
            "iteration": 1000,
            "model": peft_model[0].sharded_state_dict(),
            "optimizer": {"state": {}, "param_groups": []},
            "rng_state": [{"random_rng_state": "mock_state"}],
        }

        # Apply PEFT filtering
        filtered_dict = apply_peft_adapter_filter_to_state_dict(complete_state_dict, lora_config)

        # Verify metadata is preserved
        assert filtered_dict["checkpoint_version"] == 3.0
        assert filtered_dict["iteration"] == 1000
        assert "optimizer" in filtered_dict
        assert "rng_state" in filtered_dict

        # Verify model state is filtered
        original_model_param_count = len(complete_state_dict["model"])
        filtered_model_param_count = len(filtered_dict["model"])

        assert filtered_model_param_count < original_model_param_count, (
            f"Expected filtering to reduce parameters from {original_model_param_count} "
            f"to fewer, but got {filtered_model_param_count}"
        )

        # Verify only adapter parameters remain in model
        for param_name in filtered_dict["model"].keys():
            assert lora_config.adapter_key_filter(param_name), (
                f"Parameter '{param_name}' should not be in filtered model state dict"
            )

    def test_adapter_filtering_with_distributed_model(self, gpt_model_and_config):
        """Test that adapter filtering works with distributed models (DDP/FSDP wrapped)."""
        model_provider, lora_config = gpt_model_and_config

        # Register LoRA pre-wrap hook
        lora_hook = self._create_lora_pre_wrap_hook(lora_config)
        model_provider.register_pre_wrap_hook(lora_hook)
        model_provider.finalize()

        # Create DDP config
        ddp_config = DistributedDataParallelConfig()

        # Get the model with distributed wrappers (DDP) and PEFT applied via hook
        distributed_model = model_provider.provide_distributed_model(
            ddp_config=ddp_config,
            overlap_param_gather_with_optimizer_step=False,
            use_torch_fsdp2=False,
            wrap_with_ddp=True,
            data_parallel_random_init=False,
        )
        distributed_model = [chunk.cuda() for chunk in distributed_model]
        lora_config.set_params_to_save(distributed_model)

        # Verify the model is wrapped with DDP
        assert len(distributed_model) > 0

        # Create a complete state dict from the distributed model
        complete_distributed_state_dict = {
            "checkpoint_version": 3.0,
            "iteration": 1000,
            "model": distributed_model[0].sharded_state_dict(),
            "optimizer": {"state": {}, "param_groups": []},
            "rng_state": [{"random_rng_state": "mock_state"}],
        }

        # Test apply_peft_adapter_filter_to_state_dict with distributed model
        filtered_state_dict = apply_peft_adapter_filter_to_state_dict(complete_distributed_state_dict, lora_config)

        # Verify filtering worked
        assert len(filtered_state_dict["model"]) > 0, "Should find adapter parameters in distributed model"

        # Verify only adapter parameters are in filtered state dict
        for param_name in filtered_state_dict["model"].keys():
            assert lora_config.adapter_key_filter(param_name), (
                f"Parameter '{param_name}' should be an adapter parameter"
            )

        # Check that adapter parameters have expected naming patterns
        adapter_param_names = list(filtered_state_dict["model"].keys())
        has_lora_params = any("lora" in name.lower() or "adapter" in name.lower() for name in adapter_param_names)
        assert has_lora_params, f"Expected LoRA or adapter parameters in {adapter_param_names}"

        # Verify metadata is preserved in the filtered result
        assert filtered_state_dict["checkpoint_version"] == 3.0
        assert filtered_state_dict["iteration"] == 1000
        assert "optimizer" in filtered_state_dict
        assert "rng_state" in filtered_state_dict

        # Verify model state is filtered correctly
        original_model_param_count = len(complete_distributed_state_dict["model"])
        filtered_model_param_count = len(filtered_state_dict["model"])

        assert filtered_model_param_count < original_model_param_count, (
            f"Expected filtering to reduce parameters from {original_model_param_count} "
            f"to fewer, but got {filtered_model_param_count}"
        )

        # Verify only adapter parameters remain in model state
        for param_name in filtered_state_dict["model"].keys():
            assert lora_config.adapter_key_filter(param_name), (
                f"Parameter '{param_name}' should be an adapter parameter in filtered distributed model state dict"
            )


class TestPEFTCheckpointingValidation:
    """Simple validation tests to ensure test infrastructure works correctly."""

    def test_mock_peft_basic_functionality(self):
        """Test that MockPEFT behaves as expected."""
        mock_peft = MockPEFT()

        # Test adapter key filtering
        assert mock_peft.adapter_key_filter("layer1.adapter.weight") == True
        assert mock_peft.adapter_key_filter("layer1.linear.weight") == False
        assert mock_peft.adapter_key_filter("layer3.adapters.lora_A") == True

        # Test params_to_save is set
        assert hasattr(mock_peft, "params_to_save")
        assert len(mock_peft.params_to_save) > 0
