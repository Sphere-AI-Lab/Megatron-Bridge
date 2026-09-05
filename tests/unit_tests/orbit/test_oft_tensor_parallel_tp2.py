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

"""Two-rank CPU and CUDA regressions for OFT tensor-parallel gradient semantics."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed
import torch.multiprocessing as mp


_WORLD = 2
_OUTER_SP_SCENARIOS = [
    "outer_qkv_sp",
    "outer_fc1_sp",
    "outer_qkv_fp8_disabled_sp",
    "outer_fc1_fp8_disabled_sp",
]


class _LinearShell(torch.nn.Module):
    """Avoid allocating a TE linear when an outer-wrapper test only needs its norm."""

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()


def _base(local_weight: torch.Tensor, *, sequence_parallel: bool = False) -> torch.nn.Module:
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(local_weight, requires_grad=False)
    module.bias = None
    module.config = SimpleNamespace(sequence_parallel=sequence_parallel)
    module.allreduce_dgrad = not sequence_parallel
    module.sequence_parallel = sequence_parallel
    module.explicit_expert_comm = False
    module.disable_grad_reduce = False
    return module


def _add_layer_norm_fixture(module: torch.nn.Module, device: torch.device) -> None:
    module.normalization = "LayerNorm"
    module.eps = 1e-5
    module.zero_centered_gamma = False
    module.layer_norm_weight = torch.nn.Parameter(torch.linspace(0.75, 1.25, 8, device=device))
    module.layer_norm_bias = torch.nn.Parameter(torch.linspace(-0.2, 0.2, 8, device=device))


def _outer_wrapper(base: torch.nn.Module, split_wrapper: torch.nn.Module, *, qkv: bool) -> torch.nn.Module:
    import megatron.core.extensions.transformer_engine as te_extensions

    from megatron.bridge.orbit.oft.canonical_oft import (
        _SplitLNCanonicalOFTFC1,
        _SplitLNCanonicalOFTQKV,
    )

    original_te_linear = getattr(te_extensions, "TEColumnParallelLinear", None)
    te_extensions.TEColumnParallelLinear = _LinearShell
    try:
        outer_type = _SplitLNCanonicalOFTQKV if qkv else _SplitLNCanonicalOFTFC1
        return outer_type(base, split_wrapper)
    finally:
        if original_te_linear is None:
            del te_extensions.TEColumnParallelLinear
        else:
            te_extensions.TEColumnParallelLinear = original_te_linear


def _worker(rank: int, tmpdir: str, scenario: str, result_dir: str, backend: str) -> None:
    try:
        if backend == "nccl":
            torch.cuda.set_device(rank)
            device = torch.device("cuda", rank)
        else:
            device = torch.device("cpu")
        torch.distributed.init_process_group(
            backend=backend, init_method=f"file://{tmpdir}/pg_store", rank=rank, world_size=_WORLD
        )
        from megatron.core import parallel_state

        parallel_state.initialize_model_parallel(tensor_model_parallel_size=_WORLD)
        try:
            if scenario in {
                "split_qkv_dgrad",
                "split_fc1_dgrad",
                "split_qkv_sp",
                "split_fc1_sp",
                "split_qkv_fp8_disabled_sp",
                "split_fc1_fp8_disabled_sp",
                *_OUTER_SP_SCENARIOS,
            }:
                from megatron.bridge.orbit.oft.canonical_oft import (
                    OFTLinearSplitFC1UpGate,
                    OFTLinearSplitQKV,
                )

                torch.manual_seed(100 + rank)
                local_weight = torch.randn(8, 8, device=device)
                sequence_parallel = scenario.endswith("_sp")
                base = _base(local_weight, sequence_parallel=sequence_parallel)
                base.tp_size = _WORLD
                base._tp_group = parallel_state.get_tensor_model_parallel_group()
                if "fp8_disabled" in scenario:
                    # Direct-FP8 checkpoints can be materialized into a floating
                    # parameter while the scalar scale remains the format marker.
                    base.weight_scale_inv = torch.ones(1, device=device)
                is_qkv = "qkv" in scenario
                if is_qkv:
                    # TE column-parallel modules expose tp_size/_tp_group but
                    # not MCore's allreduce_dgrad flag.
                    del base.allreduce_dgrad
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
                if scenario.startswith("outer_"):
                    _add_layer_norm_fixture(base, device)
                    wrapper = _outer_wrapper(base, wrapper, qkv=is_qkv)
                if "fp8_disabled" in scenario:
                    wrapper.disable_adapter_layers()
                wrapper.to(device)

                # Both wrapper classes have the same column-parallel dgrad contract.
                torch.manual_seed(9 + rank if sequence_parallel else 9)
                x = torch.randn(3, 8, device=device, requires_grad=True)
                out, _ = wrapper(x)
                if sequence_parallel:
                    assert out.shape[0] == 3 * _WORLD
                    gathered_x = [torch.empty_like(x) for _ in range(_WORLD)]
                    torch.distributed.all_gather(gathered_x, x.detach())
                    gathered_input = torch.cat(gathered_x)
                    if scenario.startswith("outer_"):
                        gathered_input = torch.nn.functional.layer_norm(
                            gathered_input,
                            (8,),
                            base.layer_norm_weight,
                            base.layer_norm_bias,
                            base.eps,
                        )
                    expected_out = torch.nn.functional.linear(gathered_input, local_weight)
                    torch.testing.assert_close(out, expected_out)
                out.sum().backward()

                gathered = [torch.empty_like(local_weight) for _ in range(_WORLD)]
                torch.distributed.all_gather(gathered, local_weight)
                if scenario.startswith("outer_"):
                    reference_input = x.detach().clone().requires_grad_(True)
                    reference_norm = torch.nn.functional.layer_norm(
                        reference_input,
                        (8,),
                        base.layer_norm_weight.detach(),
                        base.layer_norm_bias.detach(),
                        base.eps,
                    )
                    reference_loss = torch.nn.functional.linear(reference_norm, torch.cat(gathered)).sum()
                    expected = torch.autograd.grad(reference_loss, reference_input)[0]
                else:
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


def _spawn(tmp_path: Path, scenario: str, *, backend: str = "gloo") -> None:
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    try:
        mp.spawn(
            _worker,
            args=(str(tmp_path), scenario, str(result_dir), backend),
            nprocs=_WORLD,
            join=True,
        )
    except Exception as spawn_error:
        details = "\n".join(path.read_text() for path in sorted(result_dir.glob("rank*.err")))
        raise AssertionError(f"spawn failed for {scenario}\n{details}") from spawn_error
    assert (result_dir / f"{scenario}_ok").exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    "scenario",
    [
        "split_qkv_dgrad",
        "split_fc1_dgrad",
        "split_qkv_sp",
        "split_fc1_sp",
        "split_qkv_fp8_disabled_sp",
        "split_fc1_fp8_disabled_sp",
        *_OUTER_SP_SCENARIOS,
    ],
)
def test_split_column_parallel_wrappers_reduce_input_gradients(tmp_path: Path, scenario: str) -> None:
    _spawn(tmp_path, scenario)


@pytest.mark.unit
def test_row_parallel_block_share_reduces_replicated_rotation_gradients(tmp_path: Path) -> None:
    _spawn(tmp_path, "block_share")


@pytest.mark.unit
@pytest.mark.skipif(
    not torch.distributed.is_nccl_available() or torch.cuda.device_count() < _WORLD,
    reason="TP2 NCCL regression requires at least two CUDA GPUs and NCCL",
)
@pytest.mark.parametrize("scenario", _OUTER_SP_SCENARIOS)
def test_outer_split_wrappers_own_sequence_gather_under_nccl(tmp_path: Path, scenario: str) -> None:
    _spawn(tmp_path, scenario, backend="nccl")
