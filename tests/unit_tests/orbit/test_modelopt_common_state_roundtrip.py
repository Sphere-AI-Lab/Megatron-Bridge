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

"""Real filesystem roundtrip for the modelopt common-state layout.

The pinned MCore's ``dist_checkpointing.save()`` embeds all common (non-sharded)
data as a ``ShardedObject("common_state")`` inside the torch-dist files and
writes no ``common.pt``. Detection (``has_modelopt_state``) and restore used to
open ``common.pt`` by literal filename, so every checkpoint saved by this repo
raised FileNotFoundError on the next resume. These tests save through the real
``dist_checkpointing.save`` -- exactly what both branches of
``_save_sharded_modelopt_state_with_async_strategy`` call -- and read back
through the fixed code paths. No mocks on the save/detect path.
"""

from pathlib import Path

import pytest
import torch
import torch.distributed
from megatron.core import dist_checkpointing

from megatron.bridge.orbit.training.modelopt_checkpoint import (
    restore_sharded_modelopt_state_via_common_reader,
)
from megatron.bridge.training.post_training.checkpointing import has_modelopt_state


@pytest.fixture
def single_rank_process_group(tmp_path: Path):
    """dist_checkpointing.save needs an initialized process group; one gloo rank."""
    created = False
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=f"file://{tmp_path}/pg_store",
            rank=0,
            world_size=1,
        )
        created = True
    yield
    if created:
        torch.distributed.destroy_process_group()


def _save_modelopt_state(tmp_path: Path, modes: list) -> Path:
    """Save a modelopt-state dict the way production does: plain (non-sharded)
    data straight through dist_checkpointing.save into <ckpt>/modelopt_state."""
    checkpoint_dir = tmp_path / "ckpt"
    state_dir = checkpoint_dir / "modelopt_state"
    state_dir.mkdir(parents=True)
    state = {"modelopt_state_dict": modes, "modelopt_version": "pinned-test"}
    dist_checkpointing.save(state, str(state_dir))
    return checkpoint_dir


@pytest.mark.unit
def test_pinned_mcore_save_writes_no_common_pt(single_rank_process_group, tmp_path: Path) -> None:
    """The premise of the bug: the production save call produces no common.pt."""
    checkpoint_dir = _save_modelopt_state(tmp_path, [("quantize", {"quant_cfg": "dummy"})])

    assert not (checkpoint_dir / "modelopt_state" / "common.pt").exists()


@pytest.mark.unit
def test_detection_reads_torch_dist_embedded_common_state(single_rank_process_group, tmp_path: Path) -> None:
    """save -> detect roundtrip on the real filesystem layout.

    Under the old torch.load(common.pt) detection this raised FileNotFoundError;
    with MCore's layout-aware reader it must simply answer True."""
    checkpoint_dir = _save_modelopt_state(tmp_path, [("quantize", {"quant_cfg": "dummy"})])

    assert has_modelopt_state(str(checkpoint_dir)) is True


@pytest.mark.unit
def test_detection_still_ignores_kd_only_state(single_rank_process_group, tmp_path: Path) -> None:
    """The mode filtering semantics must survive the reader swap."""
    checkpoint_dir = _save_modelopt_state(tmp_path, [("kd_loss", {})])

    assert has_modelopt_state(str(checkpoint_dir)) is False


@pytest.mark.unit
def test_detection_returns_false_without_modelopt_state_dir(tmp_path: Path) -> None:
    assert has_modelopt_state(str(tmp_path)) is False


@pytest.mark.unit
def test_common_reader_restore_returns_quietly_when_dir_missing(tmp_path: Path) -> None:
    """Parity with ModelOpt's own guard: no modelopt_state dir -> no-op."""
    restore_sharded_modelopt_state_via_common_reader([torch.nn.Linear(2, 2)], str(tmp_path))


@pytest.mark.unit
def test_common_reader_restore_rejects_virtual_pipeline_chunks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="virtual pipeline"):
        restore_sharded_modelopt_state_via_common_reader([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)], str(tmp_path))
