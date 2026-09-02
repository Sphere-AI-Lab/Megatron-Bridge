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

"""Sharding metadata for expert OFT rotations.

Distributed checkpointing separates *identity* (key + global offsets: what data
this is) from *replication* (``replica_id``: who else holds an identical copy;
only the all-zero holder writes). Expert ``oft_r`` is rank-local learned state,
so its EP rank belongs in the offsets. The removed code encoded it in
``replica_id`` instead -- marking each rank's unique rotations as skippable
duplicates, which made ``keep_only_main_replica`` saves silently drop every
nonzero-EP-rank rotation. These tests pin the corrected metadata; the
2-process end-to-end sentinel roundtrip lives in
``test_oft_expert_parallel_ep2.py``.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed
from megatron.core import parallel_state

from megatron.bridge.orbit.oft.canonical_oft import GroupedOFTRotation
from megatron.bridge.orbit.oft.oft_layers import (
    OFTLinear,
    OFTRotationModule,
    _make_expert_ep_sharded_tensor,
)


@pytest.fixture
def expert_parallel_ranks(monkeypatch: pytest.MonkeyPatch):
    """Pretend to be EP rank 1 of 2, ETP rank 0 of 1, expert-DP rank 1."""
    monkeypatch.setattr(parallel_state, "get_expert_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_expert_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(parallel_state, "get_expert_tensor_parallel_rank", lambda: 0)
    monkeypatch.setattr(parallel_state, "get_expert_tensor_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_expert_data_parallel_rank", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_expert_tensor_parallel_group", lambda: object())
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_group", lambda: object())


@pytest.mark.unit
def test_helper_prepends_ep_axis_for_rank_local_shared_rotation(expert_parallel_ranks) -> None:
    """One shared rotation per EP rank: EP becomes a new leading global axis."""
    sh_ten = _make_expert_ep_sharded_tensor(
        torch.zeros(4, 6),
        "adapter.oft_r",
        ep_new_axis=True,
        blocks_local_axis=0,
        blocks_tp_sharded=False,
        sharded_offsets=(),
    )

    assert sh_ten.global_shape == (2, 4, 6)  # (ep_size, num_blocks, n_elements)
    assert sh_ten.global_offset == (1, 0, 0)  # this rank owns slot ep_rank=1
    assert sh_ten.replica_id == (0, 0, 1)  # replicas = expert-DP group only
    assert sh_ten.prepend_axis_num == 1


@pytest.mark.unit
def test_helper_shards_existing_expert_axis_for_grouped_3d(expert_parallel_ranks, monkeypatch) -> None:
    """GroupedOFTRotation's 3D layout: axis 0 already is the local-expert axis."""
    monkeypatch.setattr(parallel_state, "get_expert_tensor_parallel_rank", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_expert_tensor_parallel_world_size", lambda: 2)

    sh_ten = _make_expert_ep_sharded_tensor(
        torch.zeros(2, 4, 6),
        "adapter.oft_r",
        ep_new_axis=False,
        blocks_local_axis=1,
        blocks_tp_sharded=False,  # column-parallel fc1: blocks ETP-replicated
        sharded_offsets=(),
    )

    assert sh_ten.global_shape == (4, 4, 6)  # 2 local experts x ep_size=2
    assert sh_ten.global_offset == (2, 0, 0)  # ep_rank=1 -> experts 2..3
    # Blocks replicated across ETP -> the ETP rank is a true replica coordinate.
    assert sh_ten.replica_id == (0, 1, 1)
    assert sh_ten.prepend_axis_num == 0


@pytest.mark.unit
def test_helper_offsets_blocks_axis_when_etp_sharded(expert_parallel_ranks, monkeypatch) -> None:
    """RowParallel base: the blocks axis identity moves into the offsets and
    leaves the replica ETP slot at zero."""
    monkeypatch.setattr(parallel_state, "get_expert_tensor_parallel_rank", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_expert_tensor_parallel_world_size", lambda: 2)

    sh_ten = _make_expert_ep_sharded_tensor(
        torch.zeros(4, 6),
        "adapter.oft_r",
        ep_new_axis=True,
        blocks_local_axis=0,
        blocks_tp_sharded=True,
        sharded_offsets=(),
    )

    assert sh_ten.global_shape == (2, 8, 6)  # blocks doubled across ETP
    assert sh_ten.global_offset == (1, 4, 0)  # (ep_rank, etp_rank * local_blocks)
    assert sh_ten.replica_id == (0, 0, 1)


@pytest.fixture
def single_rank_process_group(tmp_path):
    created = False
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo", init_method=f"file://{tmp_path}/pg", rank=0, world_size=1)
        created = True
    yield
    if created:
        torch.distributed.destroy_process_group()


@pytest.mark.unit
def test_sequential_expert_adapter_keeps_offsets_and_gets_edp_replica(
    single_rank_process_group, expert_parallel_ranks, monkeypatch, tmp_path
) -> None:
    """Regression for the removed replica rewrite.

    The old code computed ``(ep+1)*(edp+1)-1 if dp==1 else ep`` and wrote it
    into the TP slot -- under the mocked ranks that value is nonzero either
    way, so this asserting a zero TP slot fails against the old code. The
    parent-supplied expert offset must survive untouched, and the DP slot must
    hold the expert-DP rank.
    """
    world = torch.distributed.group.WORLD
    monkeypatch.setattr(parallel_state, "get_expert_tensor_parallel_group", lambda: world)

    adapter = OFTRotationModule(in_features=8, block_size=4, input_is_parallel=True, is_expert=True)
    sd = adapter.sharded_state_dict(
        prefix="adapter.",
        sharded_offsets=((0, 3, 16),),  # what SequentialMLP passes: global expert 3 of 16
        metadata={"dp_cp_group": world},
    )

    sh_ten = sd["adapter.oft_r"]
    assert sh_ten.global_offset[0] == 3  # the parent's expert identity survives
    assert sh_ten.replica_id[1] == 0  # no EP value smuggled into the TP slot
    assert sh_ten.replica_id[2] == 1  # expert-DP rank, not dense dp_cp rank


@pytest.mark.unit
def test_oftlinear_stamps_adapters_on_grouped_expert_bases(expert_parallel_ranks) -> None:
    """The wrapper is the only place that can see the container kind."""

    class _FakeGroupedBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_gemms = 2
            self.weight0 = torch.nn.Parameter(torch.zeros(4, 8))
            self.weight1 = torch.nn.Parameter(torch.zeros(4, 8))
            self.config = SimpleNamespace(sequence_parallel=False)

    class _FakeDenseBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(4, 8))
            self.config = SimpleNamespace(sequence_parallel=False)

    grouped_adapter = OFTRotationModule(in_features=8, block_size=4, is_expert=True)
    OFTLinear(_FakeGroupedBase(), grouped_adapter)
    assert grouped_adapter.ep_axis_sharded is True

    dense_adapter = OFTRotationModule(in_features=8, block_size=4, is_expert=True)
    OFTLinear(_FakeDenseBase(), dense_adapter)
    assert dense_adapter.ep_axis_sharded is False


@pytest.mark.unit
def test_grouped_oft_rotation_uses_expert_dp_replica(expert_parallel_ranks) -> None:
    """The 3D grouped path had correct EP offsets but a dense dp_cp replica tag."""
    rotation = GroupedOFTRotation(num_local_experts=2, in_features=8, block_size=4, is_expert=True)
    sd = rotation.sharded_state_dict(prefix="adapter_gate.", metadata={"dp_cp_group": object()})

    sh_ten = sd["adapter_gate.oft_r"]
    assert sh_ten.global_shape[0] == 4  # 2 local x ep_size=2
    assert sh_ten.global_offset[0] == 2  # ep_rank=1
    assert sh_ten.replica_id == (0, 0, 1)  # (PP, ETP-replica, expert-DP rank)
