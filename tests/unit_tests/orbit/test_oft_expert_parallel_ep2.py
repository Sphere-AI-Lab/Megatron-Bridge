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

"""EP=2 end-to-end sentinel tests for expert OFT sharding and export.

Two real processes (gloo, CPU), real ``parallel_state`` with
``expert_model_parallel_size=2``, real ``dist_checkpointing`` save/load on a
real filesystem. Rank 0 fills its rotations with 111, rank 1 with 222; the
checkpoint (and the export gather) must contain BOTH. Against the pre-fix
code these fail by construction: nonzero-EP-rank rotations were marked
non-main replicas and dropped at save, and the native-expert export path
never gathered peer tensors to rank 0.
"""

import os
from pathlib import Path

import pytest
import torch
import torch.distributed
import torch.multiprocessing as mp


_WORLD = 2
_SENTINELS = {0: 111.0, 1: 222.0}


def _distributed_worker(rank: int, tmpdir: str, scenario: str, result_dir: str) -> None:
    try:
        _distributed_worker_body(rank, tmpdir, scenario, result_dir)
    except BaseException:
        import traceback

        Path(result_dir, f"rank{rank}.err").write_text(traceback.format_exc())
        raise


def _distributed_worker_body(rank: int, tmpdir: str, scenario: str, result_dir: str) -> None:
    from megatron.core import dist_checkpointing, parallel_state

    torch.distributed.init_process_group(
        backend="gloo", init_method=f"file://{tmpdir}/pg_store", rank=rank, world_size=_WORLD
    )
    parallel_state.initialize_model_parallel(expert_model_parallel_size=_WORLD)
    try:
        ckpt_dir = os.path.join(tmpdir, "ckpt")
        # dist_checkpointing.save requires an existing directory.
        os.makedirs(ckpt_dir, exist_ok=True)
        metadata = {"dp_cp_group": parallel_state.get_data_parallel_group(with_context_parallel=True)}

        if scenario == "grouped_3d":
            from megatron.bridge.orbit.oft.canonical_oft import GroupedOFTRotation

            def build():
                return GroupedOFTRotation(num_local_experts=2, in_features=8, block_size=4, is_expert=True)

        elif scenario == "rank_local_shared":
            from megatron.bridge.orbit.oft.oft_layers import OFTRotationModule

            def build():
                rotation = OFTRotationModule(in_features=8, block_size=4, is_expert=True)
                rotation.ep_axis_sharded = True
                return rotation

        elif scenario == "native_export_gather":
            from megatron.bridge.orbit.conversion.oft_export import _gather_dsv4_native_expert_variants

            local = torch.full((4, 6), _SENTINELS[rank])
            # 8 global experts over EP=2 -> 4 per rank; this task is local expert 1.
            variants = _gather_dsv4_native_expert_variants(
                "decoder.layers.0.mlp.experts.1.w1", local, num_moe_experts=8
            )
            if rank == 0:
                names = [name for name, _ in variants]
                values = {name: t[0, 0].item() for name, t in variants}
                assert names == [
                    "decoder.layers.0.mlp.experts.1.w1",  # peer 0: global expert 0*4+1
                    "decoder.layers.0.mlp.experts.5.w1",  # peer 1: global expert 1*4+1
                ], names
                assert values["decoder.layers.0.mlp.experts.1.w1"] == 111.0
                assert values["decoder.layers.0.mlp.experts.5.w1"] == 222.0
                Path(result_dir, "export_ok").touch()
            return

        else:  # pragma: no cover
            raise AssertionError(scenario)

        rotation = build()
        with torch.no_grad():
            rotation.oft_r.fill_(_SENTINELS[rank])
        dist_checkpointing.save(rotation.sharded_state_dict(prefix="adapter.", metadata=metadata), ckpt_dir)
        torch.distributed.barrier()

        # Each rank must get ITS OWN values back, not rank 0's broadcast.
        reloaded = build()
        state = dist_checkpointing.load(reloaded.sharded_state_dict(prefix="adapter.", metadata=metadata), ckpt_dir)
        loaded = state["adapter.oft_r"]
        expected = _SENTINELS[rank]
        assert torch.all(loaded == expected), f"rank {rank}: expected {expected}, got {loaded.unique()}"

        # And the checkpoint must contain BOTH sentinels: read the full global
        # tensor (both ranks request a full replicated copy, which is legal).
        from megatron.core.dist_checkpointing.mapping import ShardedTensor

        local_shape = tuple(rotation.oft_r.shape)
        if scenario == "grouped_3d":
            global_shape = (local_shape[0] * _WORLD, *local_shape[1:])
        else:
            global_shape = (_WORLD, *local_shape)
        full = ShardedTensor.from_rank_offsets("adapter.oft_r", torch.zeros(global_shape), replica_id=rank)
        full_state = dist_checkpointing.load({"adapter.oft_r": full}, ckpt_dir)
        full_tensor = full_state["adapter.oft_r"]
        halves = torch.chunk(full_tensor, _WORLD, dim=0)
        assert torch.all(halves[0] == 111.0), "rank 0's rotations missing from the checkpoint"
        assert torch.all(halves[1] == 222.0), "rank 1's rotations missing from the checkpoint"
        if rank == 0:
            Path(result_dir, f"{scenario}_ok").touch()
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def _spawn(tmp_path: Path, scenario: str) -> None:
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    try:
        mp.spawn(
            _distributed_worker,
            args=(str(tmp_path), scenario, str(result_dir)),
            nprocs=_WORLD,
            join=True,
        )
    except Exception as spawn_error:
        errors = sorted(result_dir.glob("rank*.err"))
        details = "\n".join(f"--- {e.name} ---\n{e.read_text()}" for e in errors)
        raise AssertionError(f"spawn failed for {scenario}\n{details}") from spawn_error
    marker = "export_ok" if scenario == "native_export_gather" else f"{scenario}_ok"
    errors = sorted(result_dir.glob("rank*.err"))
    details = "\n".join(f"--- {e.name} ---\n{e.read_text()}" for e in errors)
    assert (result_dir / marker).exists(), f"worker assertions for {scenario} did not complete\n{details}"


@pytest.mark.unit
def test_grouped_3d_oft_r_survives_ep2_save_load(tmp_path: Path) -> None:
    _spawn(tmp_path, "grouped_3d")


@pytest.mark.unit
def test_rank_local_shared_oft_r_survives_ep2_save_load(tmp_path: Path) -> None:
    _spawn(tmp_path, "rank_local_shared")


@pytest.mark.unit
def test_native_expert_export_gathers_all_ep_ranks(tmp_path: Path) -> None:
    _spawn(tmp_path, "native_export_gather")
