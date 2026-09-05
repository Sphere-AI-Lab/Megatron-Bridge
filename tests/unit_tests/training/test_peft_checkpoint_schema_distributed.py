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

"""Real two-rank collective coverage for PEFT resume schema failures."""

import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed
import torch.multiprocessing as mp
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor


_WORLD_SIZE = 2


def _tensor(key: str) -> ShardedTensor:
    return ShardedTensor.from_rank_offsets(key, torch.empty(2, 3))


def _optimizer_object(rank: int) -> ShardedObject:
    return ShardedObject(
        "optimizer.distributed.param_state",
        {"rank": rank},
        (_WORLD_SIZE,),
        (rank,),
    )


def _worker(rank: int, tmpdir: str, result_dir: str) -> None:
    error_path = Path(result_dir, f"rank{rank}.err")
    try:
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=f"file://{tmpdir}/peft_schema_store",
            rank=rank,
            world_size=_WORLD_SIZE,
            timeout=datetime.timedelta(seconds=30),
        )
        from megatron.bridge.training.checkpointing import (
            CheckpointType,
            _validate_peft_run_resume_tensor_schema,
        )

        local_adapter = _tensor(f"decoder.layers.{rank}.adapter.weight")
        local_optimizer_object = _optimizer_object(rank)
        schema = {
            "model": {"adapter": local_adapter},
            "optimizer": {"param_state": local_optimizer_object},
        }

        tensor_metadata = {
            f"decoder.layers.{adapter_rank}.adapter.weight": _tensor(
                f"decoder.layers.{adapter_rank}.adapter.weight"
            ).without_data()
            for adapter_rank in range(_WORLD_SIZE)
        }
        present_optimizer_object = _optimizer_object(0)
        sharded_metadata = {
            **tensor_metadata,
            present_optimizer_object.unique_key: present_optimizer_object.without_data(),
        }

        with (
            patch(
                "megatron.bridge.training.checkpointing.dist_checkpointing.load_tensors_metadata",
                return_value=tensor_metadata,
            ),
            patch(
                "megatron.bridge.training.checkpointing.dist_checkpointing_serialization.load_sharded_metadata",
                return_value=sharded_metadata,
            ),
        ):
            try:
                _validate_peft_run_resume_tensor_schema(
                    schema,
                    schema,
                    "/checkpoint/iter_0000007",
                    CheckpointType.GLOBAL,
                    "torch_dist",
                    common_state_dict={},
                )
            except RuntimeError as error:
                outcome = str(error)
            else:
                outcome = "validator accepted missing optimizer object"

        outcomes = [None] * _WORLD_SIZE
        torch.distributed.all_gather_object(outcomes, outcome)
        missing_identifier = _optimizer_object(1).unique_key
        assert all("missing optimizer object" in item for item in outcomes), outcomes
        assert all(missing_identifier in item for item in outcomes), outcomes
        assert len(set(outcomes)) == 1, outcomes
        if rank == 0:
            Path(result_dir, "coordinated_failure_ok").touch()
    except BaseException:
        import traceback

        error_path.write_text(traceback.format_exc())
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@pytest.mark.unit
def test_two_rank_missing_optimizer_object_fails_identically_before_dcp(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    try:
        mp.spawn(
            _worker,
            args=(str(tmp_path), str(result_dir)),
            nprocs=_WORLD_SIZE,
            join=True,
        )
    except Exception as spawn_error:
        details = "\n".join(path.read_text() for path in sorted(result_dir.glob("rank*.err")))
        raise AssertionError(f"two-rank PEFT schema validation failed\n{details}") from spawn_error

    details = "\n".join(path.read_text() for path in sorted(result_dir.glob("rank*.err")))
    assert (result_dir / "coordinated_failure_ok").exists(), details
