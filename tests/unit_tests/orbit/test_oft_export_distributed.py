# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Two-rank Gloo regressions for coordinated OFT export failures."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed
import torch.multiprocessing as mp


_WORLD_SIZE = 2


def _worker(rank: int, tmpdir: str, result_dir: str) -> None:
    try:
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=f"file://{tmpdir}/pg_store",
            rank=rank,
            world_size=_WORLD_SIZE,
            timeout=timedelta(seconds=20),
        )
        try:
            from megatron.bridge.orbit.conversion import oft_export

            task = oft_export.OFTAdapterConversionTask(
                global_base_prefix="decoder.layers.0.self_attention.linear_qkv",
                local_base_prefix="decoder.layers.0.self_attention.linear_qkv",
                is_expert=False,
                input_is_parallel=False,
                block_size=4,
                r=2,
                block_share=False,
                pp_rank=0,
                vp_stage=0,
                tensor_dtype="torch.float32",
                device_type="cpu",
            )

            mixin = oft_export.OrbitOFTExportMixin()
            mixin._megatron_global_oft_adapters_info_all_pp_ranks = (  # type: ignore[method-assign]
                lambda model: [] if rank == 0 else [task]
            )
            original_unwrap_model = oft_export.unwrap_model
            oft_export.unwrap_model = lambda models: models
            try:
                list(
                    mixin.stream_oft_adapter_weights_megatron_to_hf(
                        SimpleNamespace(config=SimpleNamespace(num_moe_experts=0)),
                        show_progress=False,
                    )
                )
            except RuntimeError as exc:
                assert "differs from rank 0" in str(exc)
            else:  # pragma: no cover
                raise AssertionError("stream accepted divergent empty/nonempty task plans")
            finally:
                oft_export.unwrap_model = original_unwrap_model

            try:
                oft_export._distributed_error("rank-local setup", "sentinel" if rank == 1 else None)
            except RuntimeError as exc:
                assert "rank 1: sentinel" in str(exc)
            else:  # pragma: no cover
                raise AssertionError("one-rank setup error was not propagated")

            adapter_name = "model.layers.0.self_attn.q_proj.oft_R.weight"
            original_export = oft_export.export_oft_adapter_weights
            original_peft_version = oft_export._installed_peft_version
            original_publish = oft_export._publish_hf_oft_adapter_directory
            oft_export.export_oft_adapter_weights = lambda *args, **kwargs: iter([(adapter_name, torch.zeros(2, 6))])
            oft_export._installed_peft_version = lambda: "0.19.1"

            def fail_adapter_publication(*args, **kwargs) -> None:
                raise OSError("adapter publication sentinel")

            oft_export._publish_hf_oft_adapter_directory = fail_adapter_publication
            try:
                oft_export.save_hf_oft_adapter(
                    SimpleNamespace(),
                    object(),
                    Path(tmpdir, "adapter"),
                    SimpleNamespace(
                        r=2,
                        block_size=0,
                        block_share=False,
                        coft=False,
                        eps=6e-5,
                        module_dropout=0.0,
                    ),
                    base_model_name_or_path="example/base",
                    show_progress=False,
                )
            except RuntimeError as exc:
                assert "OSError: adapter publication sentinel" in str(exc)
            else:  # pragma: no cover
                raise AssertionError("save did not propagate rank-zero publication failure")
            finally:
                oft_export.export_oft_adapter_weights = original_export
                oft_export._installed_peft_version = original_peft_version
                oft_export._publish_hf_oft_adapter_directory = original_publish

            def fail_publication() -> None:
                if rank != 0:  # pragma: no cover
                    raise AssertionError("publication operation ran outside rank 0")
                raise OSError("publication sentinel")

            try:
                oft_export._run_rank0_stage(
                    "publication",
                    is_distributed=True,
                    is_rank0=rank == 0,
                    operation=fail_publication,
                )
            except RuntimeError as exc:
                assert "OSError: publication sentinel" in str(exc)
            else:  # pragma: no cover
                raise AssertionError("rank-zero publication error was not propagated")

            Path(result_dir, f"rank{rank}.ok").touch()
        finally:
            torch.distributed.destroy_process_group()
    except BaseException:
        import traceback

        Path(result_dir, f"rank{rank}.err").write_text(traceback.format_exc())
        raise


@pytest.mark.unit
def test_oft_export_failures_are_coordinated_across_two_ranks(tmp_path: Path) -> None:
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
        errors = sorted(result_dir.glob("rank*.err"))
        details = "\n".join(f"--- {error.name} ---\n{error.read_text()}" for error in errors)
        raise AssertionError(f"distributed OFT export probe failed\n{details}") from spawn_error

    assert sorted(path.name for path in result_dir.glob("*.ok")) == ["rank0.ok", "rank1.ok"]
