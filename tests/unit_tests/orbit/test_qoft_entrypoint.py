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
from pathlib import Path

import pytest
import torch
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from transformers import Qwen3Config


_REPO_ROOT = Path(__file__).parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "orbit" / "finetune_qoft.py"


def _load_qoft_entrypoint():
    spec = importlib.util.spec_from_file_location("qoft_entrypoint_under_test", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
