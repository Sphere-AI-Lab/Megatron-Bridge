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

from pathlib import Path

import pytest
import torch

from megatron.bridge.orbit.peft_ext.peft_mixin import OrbitPEFTMixin
from megatron.bridge.orbit.training import peft_reports
from megatron.bridge.orbit.training.peft_reports import _write_peft_parameter_reports


@pytest.mark.unit
def test_peft_parameter_reports_partition_sort_and_summarize_parameters(tmp_path: Path) -> None:
    model = torch.nn.Module()
    model.register_parameter("z_trainable", torch.nn.Parameter(torch.ones(2, dtype=torch.float32)))
    model.register_parameter("a_frozen", torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16), requires_grad=False))
    model.register_parameter("a_trainable", torch.nn.Parameter(torch.ones(1, dtype=torch.float32)))

    paths = _write_peft_parameter_reports(model, report_dir=str(tmp_path))

    assert paths == {
        "trainable": str(tmp_path / "megatron_bridge_peft_trainable_params.txt"),
        "frozen": str(tmp_path / "megatron_bridge_peft_frozen_params.txt"),
        "summary": str(tmp_path / "megatron_bridge_peft_summary.txt"),
    }
    assert (tmp_path / "megatron_bridge_peft_trainable_params.txt").read_text() == (
        "name\tnumel\tdtype\tshape\na_trainable\t1\tfloat32\t(1,)\nz_trainable\t2\tfloat32\t(2,)\n"
    )
    assert (tmp_path / "megatron_bridge_peft_frozen_params.txt").read_text() == (
        "name\tnumel\tdtype\tshape\na_frozen\t3\tbfloat16\t(3,)\n"
    )
    assert (tmp_path / "megatron_bridge_peft_summary.txt").read_text() == (
        "total_parameters\t6\n"
        "trainable_parameters\t3\n"
        "frozen_parameters\t3\n"
        "trainable_percentage\t50.00\n"
        "frozen_percentage\t50.00\n"
    )


@pytest.mark.unit
def test_orbit_peft_reports_include_every_model_chunk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dropping all but chunks[0] would omit the second pipeline chunk."""
    from megatron.bridge.utils import common_utils

    first = torch.nn.Module()
    first.register_parameter("first_adapter", torch.nn.Parameter(torch.ones(2)))
    second = torch.nn.Module()
    second.register_parameter("second_adapter", torch.nn.Parameter(torch.ones(3)))

    real_writer = _write_peft_parameter_reports

    def write_to_test_directory(model, report_dir: str):
        return real_writer(model, report_dir=str(tmp_path))

    monkeypatch.setattr(common_utils, "get_rank_safe", lambda: 0)
    monkeypatch.setattr(common_utils, "print_rank_0", lambda message: None)
    monkeypatch.setattr(peft_reports, "_write_peft_parameter_reports", write_to_test_directory)

    OrbitPEFTMixin()._write_parameter_reports([first, second])

    assert (tmp_path / "megatron_bridge_peft_trainable_params.txt").read_text() == (
        "name\tnumel\tdtype\tshape\n"
        "model_chunks.0.first_adapter\t2\tfloat32\t(2,)\n"
        "model_chunks.1.second_adapter\t3\tfloat32\t(3,)\n"
    )
    assert (tmp_path / "megatron_bridge_peft_summary.txt").read_text() == (
        "total_parameters\t5\n"
        "trainable_parameters\t5\n"
        "frozen_parameters\t0\n"
        "trainable_percentage\t100.00\n"
        "frozen_percentage\t0.00\n"
    )
