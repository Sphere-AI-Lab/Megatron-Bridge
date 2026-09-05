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

import os
import subprocess
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).parents[3]
_PEFT_LAUNCHER = _REPO_ROOT / "scripts" / "orbit" / "run_peft_finetune.sh"
_QOFT_LAUNCHER = _REPO_ROOT / "scripts" / "orbit" / "run_qoft_finetune.sh"


def _install_uv_capture(tmp_path: Path) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture_path = tmp_path / "uv-args.bin"
    uv = bin_dir / "uv"
    uv.write_text('#!/bin/bash\nprintf "%s\\0" "$@" >"$CAPTURE_PATH"\n')
    uv.chmod(0o755)
    return bin_dir, capture_path


def _captured_args(capture_path: Path) -> list[str]:
    return [arg.decode() for arg in capture_path.read_bytes().split(b"\0") if arg]


def _launcher_env(tmp_path: Path, launcher: Path) -> tuple[dict[str, str], Path]:
    bin_dir, capture_path = _install_uv_capture(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CAPTURE_PATH": str(capture_path),
        "NUM_GPUS": "2",
    }
    env.pop("EXTRA_ARGS", None)
    if launcher == _QOFT_LAUNCHER:
        env.update(
            QUANT="int4",
            HF_MODEL_PATH="/models/example",
            MEGATRON_CKPT="/checkpoints/example",
        )
    return env, capture_path


@pytest.mark.unit
@pytest.mark.parametrize(
    ("launcher", "entrypoint"),
    [
        (_PEFT_LAUNCHER, "scripts/orbit/finetune_peft.py"),
        (_QOFT_LAUNCHER, "scripts/orbit/finetune_qoft.py"),
    ],
)
def test_launchers_use_project_uv_and_preserve_positional_arguments(
    tmp_path: Path,
    launcher: Path,
    entrypoint: str,
) -> None:
    env, capture_path = _launcher_env(tmp_path, launcher)

    result = subprocess.run(
        ["bash", str(launcher), "--", "--output-dir", "/runs/two words", "--train-iters", "20"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    forwarded = _captured_args(capture_path)
    assert forwarded[:8] == [
        "run",
        "--project",
        str(_REPO_ROOT),
        "python",
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        entrypoint,
    ]
    assert forwarded[-4:] == ["--output-dir", "/runs/two words", "--train-iters", "20"]


@pytest.mark.unit
@pytest.mark.parametrize("launcher", [_PEFT_LAUNCHER, _QOFT_LAUNCHER])
@pytest.mark.parametrize("num_gpus", ["0", "-2", "two", "1.5"])
def test_launchers_reject_nonpositive_or_nonnumeric_process_counts(
    tmp_path: Path,
    launcher: Path,
    num_gpus: str,
) -> None:
    env, capture_path = _launcher_env(tmp_path, launcher)
    env["NUM_GPUS"] = num_gpus

    result = subprocess.run(
        ["bash", str(launcher)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "NUM_GPUS must be a positive integer" in result.stderr
    assert not capture_path.exists()


@pytest.mark.unit
@pytest.mark.parametrize("launcher", [_PEFT_LAUNCHER, _QOFT_LAUNCHER])
def test_launchers_reject_lossy_extra_args_environment_variable(tmp_path: Path, launcher: Path) -> None:
    env, capture_path = _launcher_env(tmp_path, launcher)
    env["EXTRA_ARGS"] = '--output-dir "/runs/two words"'

    result = subprocess.run(
        ["bash", str(launcher)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "pass extra arguments after --" in result.stderr
    assert not capture_path.exists()
