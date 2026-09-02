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

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest
import torch
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from transformers import Qwen3Config


_REPO_ROOT = Path(__file__).parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "orbit" / "finetune_qoft.py"
_QOFT_LAUNCHER_PATH = _REPO_ROOT / "scripts" / "orbit" / "run_qoft_finetune.sh"

_RETIRED_RECIPE_PATHS = [
    "scripts/orbit/models/gpt_oss/finetune_oft.py",
    "scripts/orbit/models/kimi_k25/finetune_qoft_int4.py",
    "scripts/orbit/models/kimi_k25/finetune_qoft_nvfp4.py",
    "scripts/orbit/models/moonlight_16b/finetune_qoft_int4.py",
    "scripts/orbit/models/qwen3_moe/finetune_qoft.py",
    "scripts/orbit/models/qwen3_moe/finetune_qoft_int4.py",
    "scripts/orbit/tutorials/llama/01_quickstart_finetune_lora.py",
]


def _load_qoft_entrypoint():
    spec = importlib.util.spec_from_file_location("qoft_entrypoint_under_test", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", _RETIRED_RECIPE_PATHS)
def test_superseded_model_recipes_are_removed(relative_path: str) -> None:
    """The generic PEFT/QOFT entrypoints are the only maintained launch surfaces."""
    assert not (_REPO_ROOT / relative_path).exists()


@pytest.mark.unit
def test_build_config_accepts_dense_qwen3_fp8(tmp_path: Path) -> None:
    """A dense Qwen3 FP8 checkpoint gets a one-GPU, non-MoE QOFT config."""
    hf_model_path = tmp_path / "qwen3-4b-fp8"
    hf_model_path.mkdir()
    hf_config = Qwen3Config(
        hidden_size=256,
        intermediate_size=768,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=512,
    )
    hf_config.architectures = ["Qwen3ForCausalLM"]
    hf_config.save_pretrained(hf_model_path)

    entrypoint = _load_qoft_entrypoint()
    args = entrypoint.parse_args(
        [
            "--quant",
            "fp8",
            "--hf-model-path",
            str(hf_model_path),
            "--pretrained-checkpoint",
            str(tmp_path / "converted-checkpoint"),
            "--skip-train",
        ]
    )
    arch_spec = entrypoint.resolve_arch(str(hf_model_path))
    entrypoint._fill_arch_defaults(args, arch_spec)
    entrypoint.build_config(args, arch_spec)

    assert arch_spec["key"] == "qwen3_dense"
    assert args.tp == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("architecture", "expected_group_size"),
    [
        ("Qwen3MoeForCausalLM", 128),
        ("KimiK25ForConditionalGeneration", 32),
        ("DeepseekV3ForCausalLM", 32),
    ],
)
def test_int4_group_size_defaults_per_architecture(
    tmp_path: Path, architecture: str, expected_group_size: int
) -> None:
    """--group-size must default per architecture, not to one fixed value.

    Kimi and Moonlight's converted checkpoints are quantized at group_size=32
    (the legacy default their converter falls back to when the HF config has
    no quantization_config); Qwen3 MoE's are quantized at 128. A single
    global default would silently mismatch one side or the other and produce
    wrong dequantized weights with no error.
    """
    entrypoint = _load_qoft_entrypoint()
    spec = entrypoint.ARCH_SPECS[architecture]
    args = entrypoint.parse_args(
        [
            "--quant",
            "int4",
            "--hf-model-path",
            str(tmp_path / "hf-model"),
            "--pretrained-checkpoint",
            str(tmp_path / "checkpoint"),
        ]
    )

    entrypoint._fill_arch_defaults(args, spec)

    assert args.group_size == expected_group_size


@pytest.mark.unit
def test_int4_group_size_explicit_override_wins_over_architecture_default(tmp_path: Path) -> None:
    """An explicit --group-size must still take priority over the architecture default."""
    entrypoint = _load_qoft_entrypoint()
    spec = entrypoint.ARCH_SPECS["KimiK25ForConditionalGeneration"]
    args = entrypoint.parse_args(
        [
            "--quant",
            "int4",
            "--hf-model-path",
            str(tmp_path / "hf-model"),
            "--pretrained-checkpoint",
            str(tmp_path / "checkpoint"),
            "--group-size",
            "64",
        ]
    )

    entrypoint._fill_arch_defaults(args, spec)

    assert args.group_size == 64


@pytest.mark.unit
def test_fp8_explicit_checkpoint_requests_per_layer_state_dict() -> None:
    """Direct FP8 keys must disable Megatron's homogeneous-layer schema."""
    entrypoint = _load_qoft_entrypoint()
    original_kwargs = {"metadata": {"dp_cp_group": "preserved"}}
    checkpoint_keys = {
        "decoder.layers.0.self_attention.linear_proj.weight_w",
        "decoder.layers.0.self_attention.linear_proj.weight_scale_inv",
    }

    result = entrypoint.fp8_model_state_dict_kwargs_for_checkpoint_keys(
        original_kwargs,
        checkpoint_keys,
    )

    assert result == {"metadata": {"dp_cp_group": "preserved", "non_homogeneous_layers": True}}
    assert original_kwargs == {"metadata": {"dp_cp_group": "preserved"}}


@pytest.mark.unit
def test_fp8_preflight_reports_only_missing_request_keys() -> None:
    """Contract drift should fail with a concise missing-key preview."""
    entrypoint = _load_qoft_entrypoint()
    missing_key = "decoder.layers.0.self_attention.linear_proj.weight_w"
    present_key = "embedding.word_embeddings.weight"
    model_state = {
        "missing": ShardedTensor.from_rank_offsets(missing_key, torch.empty((2, 2))),
        "present": ShardedTensor.from_rank_offsets(present_key, torch.empty((2, 2))),
    }

    with pytest.raises(RuntimeError) as exc_info:
        entrypoint.assert_fp8_request_keys_in_checkpoint(
            model_state,
            frozenset({present_key}),
            "Qwen3 dense",
        )

    message = str(exc_info.value)
    assert "Qwen3 dense FP8 preflight" in message
    assert "1 requested tensor entries are not in the checkpoint index" in message
    assert missing_key in message
    assert present_key not in message
    assert "checkpoint index of 1 entries" in message


@pytest.mark.unit
def test_qoft_cli_rejects_removed_quantized_lora_option(tmp_path: Path) -> None:
    """The quantized entrypoint owns OFT only until QLoRA has an implementation."""
    entrypoint = _load_qoft_entrypoint()

    with pytest.raises(SystemExit):
        entrypoint.parse_args(
            [
                "--quant",
                "int4",
                "--hf-model-path",
                str(tmp_path / "hf-model"),
                "--pretrained-checkpoint",
                str(tmp_path / "checkpoint"),
                "--peft",
                "lora",
            ]
        )


@pytest.mark.unit
def test_moonlight_topology_validation_preserves_supported_layouts(tmp_path: Path) -> None:
    """The generic entrypoint retains the old Moonlight launcher safety checks."""
    entrypoint = _load_qoft_entrypoint()
    spec = entrypoint.ARCH_SPECS["DeepseekV3ForCausalLM"]
    base_argv = [
        "--quant",
        "int4",
        "--hf-model-path",
        str(tmp_path / "hf-model"),
        "--pretrained-checkpoint",
        str(tmp_path / "checkpoint"),
    ]

    supported = entrypoint.parse_args(base_argv + ["--tp", "2", "--ep", "2", "--sp"])
    entrypoint._fill_arch_defaults(supported, spec)
    entrypoint._validate_runtime_topology(supported, spec, world_size=4)

    unsupported = entrypoint.parse_args(base_argv + ["--tp", "1", "--ep", "1"])
    entrypoint._fill_arch_defaults(unsupported, spec)
    with pytest.raises(SystemExit, match="supported GPU/parallel layouts"):
        entrypoint._validate_runtime_topology(unsupported, spec, world_size=8)

    missing_sp = entrypoint.parse_args(base_argv + ["--tp", "2", "--ep", "2", "--no-sp"])
    entrypoint._fill_arch_defaults(missing_sp, spec)
    with pytest.raises(SystemExit, match="sequence parallelism"):
        entrypoint._validate_runtime_topology(missing_sp, spec, world_size=4)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("world_size", "expected_tp", "expected_ep", "expected_sp"),
    [
        (1, 1, 1, False),
        (2, 1, 2, False),
        (4, 2, 2, True),
    ],
)
def test_moonlight_defaults_match_supported_world_size(
    tmp_path: Path,
    world_size: int,
    expected_tp: int,
    expected_ep: int,
    expected_sp: bool,
) -> None:
    """Required-variable-only launches choose a supported Moonlight topology."""
    entrypoint = _load_qoft_entrypoint()
    spec = entrypoint.ARCH_SPECS["DeepseekV3ForCausalLM"]
    args = entrypoint.parse_args(
        [
            "--quant",
            "int4",
            "--hf-model-path",
            str(tmp_path / "hf-model"),
            "--pretrained-checkpoint",
            str(tmp_path / "checkpoint"),
        ]
    )

    entrypoint._fill_arch_defaults(args, spec, world_size=world_size)
    entrypoint._validate_runtime_topology(args, spec, world_size=world_size)

    assert (args.tp, args.ep, args.sp) == (expected_tp, expected_ep, expected_sp)


@pytest.mark.unit
def test_qoft_launcher_forwards_explicit_operational_controls(tmp_path: Path) -> None:
    """The consolidated launcher exposes former model-wrapper controls directly."""
    capture_path = tmp_path / "torchrun-args.txt"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    torchrun = bin_dir / "torchrun"
    torchrun.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >"$CAPTURE_PATH"\n')
    torchrun.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CAPTURE_PATH": str(capture_path),
        "QUANT": "int4",
        "HF_MODEL_PATH": "/models/kimi",
        "MEGATRON_CKPT": "/checkpoints/kimi",
        "NUM_GPUS": "8",
        "SP": "1",
        "DISTRIBUTED_TIMEOUT_MINUTES": "60",
        "TARGET_MODULES": "linear_fc1,linear_fc2",
        "GROUP_SIZE": "128",
        "SAVE_CHECKPOINTS": "1",
        "SAVE_INTERVAL": "250",
        "SKIP_EVAL": "1",
        "PROFILE_MEMORY": "1",
        "PROFILE_MEMORY_STEPS": "2",
    }

    result = subprocess.run(
        ["bash", str(_QOFT_LAUNCHER_PATH)],
        cwd=_REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    forwarded = capture_path.read_text().splitlines()
    assert forwarded[:2] == ["--nproc_per_node=8", "scripts/orbit/finetune_qoft.py"]
    assert "--sp" in forwarded
    assert "--distributed-timeout-minutes" in forwarded
    assert "--target-modules" in forwarded
    assert "--group-size" in forwarded
    assert "--save-checkpoints" in forwarded
    assert "--save-interval" in forwarded
    assert "--skip-eval" in forwarded
    assert "--profile-memory" in forwarded
    assert "--profile-memory-steps" in forwarded
    assert "--peft" not in forwarded


def _oft_argv(tmp_path: Path, quant: str, *extra: str) -> list[str]:
    return [
        "--quant",
        quant,
        "--hf-model-path",
        str(tmp_path / "hf-model"),
        "--pretrained-checkpoint",
        str(tmp_path / "checkpoint"),
        *extra,
    ]


def _build_oft(entrypoint, tmp_path: Path, arch: str, quant: str, *extra: str):
    """parse_args -> arch defaults -> build_oft, the order main() uses."""
    spec = entrypoint.ARCH_SPECS[arch]
    args = entrypoint.parse_args(_oft_argv(tmp_path, quant, *extra))
    entrypoint._fill_arch_defaults(args, spec)
    return entrypoint.build_oft(args, spec)


@pytest.mark.unit
def test_qoft_defaults_to_canonical_oft_with_split_targets(tmp_path: Path) -> None:
    """Without --oft-type the entrypoint builds CanonicalOFT, not legacy shared-R OFT.

    Regression test: both launchers used to hardcode ``OFT``, so Q/K/V shared one
    rotation on the fused linear_qkv and CanonicalOFT was unreachable.
    """
    from megatron.bridge.orbit.oft.canonical_oft import CanonicalOFT

    entrypoint = _load_qoft_entrypoint()
    peft = _build_oft(entrypoint, tmp_path, "Qwen3MoeForCausalLM", "int4")

    assert isinstance(peft, CanonicalOFT)
    # CanonicalOFT rejects the fused leaves outright; assert the resolved list is split.
    assert not any(target.endswith(("linear_qkv", "linear_fc1")) for target in peft.target_modules)
    assert {"linear_q", "linear_k", "linear_v"}.issubset(set(peft.target_modules))


@pytest.mark.unit
def test_qoft_translates_architecture_fused_targets_to_split_names(tmp_path: Path) -> None:
    """Arch defaults are written with fused leaf names; canonical mode expands them."""
    from megatron.bridge.orbit.oft.canonical_oft import CanonicalOFT

    entrypoint = _load_qoft_entrypoint()
    # nvfp4 is the one arch/quant pair carrying an explicit fused target list.
    peft = _build_oft(
        entrypoint,
        tmp_path,
        "Qwen3MoeForCausalLM",
        "nvfp4",
        "--target-modules",
        "linear_qkv,linear_proj",
    )

    assert isinstance(peft, CanonicalOFT)
    assert peft.target_modules == ["linear_q", "linear_k", "linear_v", "linear_proj"]


@pytest.mark.unit
def test_qoft_oft_type_oft_opts_back_into_legacy_shared_r(tmp_path: Path) -> None:
    """--oft-type oft is the explicit opt-in to the legacy one-rotation class."""
    from megatron.bridge.orbit.oft.canonical_oft import CanonicalOFT
    from megatron.bridge.orbit.oft.oft import OFT

    entrypoint = _load_qoft_entrypoint()
    peft = _build_oft(
        entrypoint,
        tmp_path,
        "Qwen3MoeForCausalLM",
        "nvfp4",
        "--oft-type",
        "oft",
    )

    assert isinstance(peft, OFT)
    assert not isinstance(peft, CanonicalOFT)
    # Legacy targets are passed through unchanged, fused leaves included.
    assert peft.target_modules == list(entrypoint.QWEN3_MOE_OFT_TARGET_MODULES)


@pytest.mark.unit
@pytest.mark.parametrize("quant", ["fp8", "nvfp4"])
def test_qoft_canonical_rejects_grouped_expert_fc1_on_fp8_and_nvfp4(tmp_path: Path, quant: str) -> None:
    """OFTLinearGroupedSplitFC1UpGate raises on FP8/NVFP4 in forward, i.e. after the
    checkpoint is loaded. The launcher must reject the combination up front."""
    entrypoint = _load_qoft_entrypoint()

    with pytest.raises(SystemExit, match="cannot train grouped-expert linear_fc1"):
        _build_oft(entrypoint, tmp_path, "Qwen3MoeForCausalLM", quant)


@pytest.mark.unit
def test_qoft_canonical_allowed_on_int4_grouped_experts(tmp_path: Path) -> None:
    """INT4 grouped FC1 is implemented, so canonical stays the default there."""
    from megatron.bridge.orbit.oft.canonical_oft import CanonicalOFT

    entrypoint = _load_qoft_entrypoint()
    assert isinstance(_build_oft(entrypoint, tmp_path, "DeepseekV3ForCausalLM", "int4"), CanonicalOFT)


@pytest.mark.unit
def test_qoft_canonical_allowed_on_nvfp4_when_fc1_is_not_targeted(tmp_path: Path) -> None:
    """The guard is about grouped FC1 specifically, not about NVFP4 as a whole."""
    from megatron.bridge.orbit.oft.canonical_oft import CanonicalOFT

    entrypoint = _load_qoft_entrypoint()
    peft = _build_oft(
        entrypoint,
        tmp_path,
        "Qwen3MoeForCausalLM",
        "nvfp4",
        "--target-modules",
        "linear_qkv,linear_proj,linear_fc2",
    )

    assert isinstance(peft, CanonicalOFT)
    assert "linear_fc2" in peft.target_modules


@pytest.mark.unit
def test_qoft_canonical_allowed_on_dense_architecture(tmp_path: Path) -> None:
    """Qwen3 dense has no grouped experts, so FP8 canonical is not gated."""
    from megatron.bridge.orbit.oft.canonical_oft import CanonicalOFT

    entrypoint = _load_qoft_entrypoint()
    assert isinstance(_build_oft(entrypoint, tmp_path, "Qwen3ForCausalLM", "fp8"), CanonicalOFT)


@pytest.mark.unit
def test_qoft_rejects_unknown_oft_type(tmp_path: Path) -> None:
    entrypoint = _load_qoft_entrypoint()

    with pytest.raises(SystemExit):
        entrypoint.parse_args(_oft_argv(tmp_path, "int4", "--oft-type", "canonical"))


@pytest.mark.unit
def test_qoft_launcher_forwards_oft_type(tmp_path: Path) -> None:
    """OFT_TYPE reaches the entrypoint, so legacy OFT is selectable from the launcher."""
    capture_path = tmp_path / "torchrun-args.txt"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    torchrun = bin_dir / "torchrun"
    torchrun.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >"$CAPTURE_PATH"\n')
    torchrun.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CAPTURE_PATH": str(capture_path),
        "QUANT": "nvfp4",
        "HF_MODEL_PATH": "/models/qwen3-moe",
        "MEGATRON_CKPT": "/checkpoints/qwen3-moe",
        "NUM_GPUS": "4",
        "OFT_TYPE": "oft",
    }

    result = subprocess.run(
        ["bash", str(_QOFT_LAUNCHER_PATH)],
        cwd=_REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    forwarded = capture_path.read_text().splitlines()
    assert "--oft-type" in forwarded
    assert forwarded[forwarded.index("--oft-type") + 1] == "oft"
