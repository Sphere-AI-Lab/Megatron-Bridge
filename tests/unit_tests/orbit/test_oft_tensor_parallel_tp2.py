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

"""Two-rank CPU regressions for OFT tensor-parallel gradient semantics."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed
import torch.multiprocessing as mp


_WORLD = 2


def _base(local_weight: torch.Tensor) -> torch.nn.Module:
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(local_weight, requires_grad=False)
    module.bias = None
    module.config = SimpleNamespace(sequence_parallel=False)
    module.allreduce_dgrad = True
    module.sequence_parallel = False
    module.explicit_expert_comm = False
    module.disable_grad_reduce = False
    return module


def _worker(rank: int, tmpdir: str, scenario: str, result_dir: str) -> None:
    try:
        torch.distributed.init_process_group(
            backend="gloo", init_method=f"file://{tmpdir}/pg_store", rank=rank, world_size=_WORLD
        )
        from megatron.core import parallel_state

        parallel_state.initialize_model_parallel(tensor_model_parallel_size=_WORLD)
        try:
            if scenario in {"split_qkv_dgrad", "split_fc1_dgrad"}:
                from megatron.bridge.orbit.oft.canonical_oft import (
                    OFTLinearSplitFC1UpGate,
                    OFTLinearSplitQKV,
                )

                torch.manual_seed(100 + rank)
                local_weight = torch.randn(8, 8)
                base = _base(local_weight)
                if scenario == "split_qkv_dgrad":
                    # TE column-parallel modules expose tp_size/_tp_group but
                    # not MCore's allreduce_dgrad flag.
                    del base.allreduce_dgrad
                    base.tp_size = _WORLD
                    base._tp_group = parallel_state.get_tensor_model_parallel_group()
                    wrapper = OFTLinearSplitQKV(
                        base,
                        in_features=8,
                        provider=SimpleNamespace(
                            num_attention_heads=4,
                            num_query_groups=2,
                            kv_channels=2,
                            attention_output_gate=False,
                        ),
                        block_size=4,
                        input_is_parallel=False,
                    )
                else:
                    wrapper = OFTLinearSplitFC1UpGate(
                        base,
                        in_features=8,
                        block_size=4,
                        input_is_parallel=False,
                    )

                # Both wrapper classes have the same column-parallel dgrad contract.
                torch.manual_seed(9)
                x = torch.randn(3, 8, requires_grad=True)
                out, _ = wrapper(x)
                out.sum().backward()

                gathered = [torch.empty_like(local_weight) for _ in range(_WORLD)]
                torch.distributed.all_gather(gathered, local_weight)
                expected = torch.cat(gathered, dim=0).sum(dim=0).expand_as(x)
                torch.testing.assert_close(x.grad, expected)
            elif scenario == "block_share":
                from megatron.bridge.orbit.oft.oft_layers import OFTRotationModule

                rotation = OFTRotationModule(
                    in_features=8,
                    block_size=4,
                    block_share=True,
                    input_is_parallel=True,
                    dtype=torch.float64,
                )
                with torch.no_grad():
                    rotation.oft_r.fill_(0.05)
                torch.manual_seed(20 + rank)
                rotation(torch.randn(4, 8, dtype=torch.float64)).sum().backward()

                peer_grad = (
                    rotation.oft_r.grad.detach().clone() if rank == 0 else torch.empty_like(rotation.oft_r.grad)
                )
                torch.distributed.broadcast(peer_grad, src=0)
                torch.testing.assert_close(rotation.oft_r.grad, peer_grad)
            else:  # pragma: no cover
                raise AssertionError(scenario)

            if rank == 0:
                Path(result_dir, f"{scenario}_ok").touch()
        finally:
            parallel_state.destroy_model_parallel()
            torch.distributed.destroy_process_group()
    except BaseException:
        import traceback

        Path(result_dir, f"rank{rank}.err").write_text(traceback.format_exc())
        raise


def _spawn(tmp_path: Path, scenario: str) -> None:
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    try:
        mp.spawn(_worker, args=(str(tmp_path), scenario, str(result_dir)), nprocs=_WORLD, join=True)
    except Exception as spawn_error:
        details = "\n".join(path.read_text() for path in sorted(result_dir.glob("rank*.err")))
        raise AssertionError(f"spawn failed for {scenario}\n{details}") from spawn_error
    assert (result_dir / f"{scenario}_ok").exists()


@pytest.mark.unit
@pytest.mark.parametrize("scenario", ["split_qkv_dgrad", "split_fc1_dgrad"])
def test_split_column_parallel_wrappers_reduce_input_gradients(tmp_path: Path, scenario: str) -> None:
    _spawn(tmp_path, scenario)


@pytest.mark.unit
def test_row_parallel_block_share_reduces_replicated_rotation_gradients(tmp_path: Path) -> None:
    _spawn(tmp_path, "block_share")
