#!/usr/bin/env python3
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

"""Qwen3-30B-A3B MoE QOFT finetuning with INT4 W4A16 base weights."""

# ruff: noqa: D101, D103  # operational scripts: helpers here are entrypoint plumbing, not API

import argparse
import os
import re
from typing import Any

import torch


INT4_SCALE_DTYPE = torch.bfloat16

_DENSE_INT4_TRIPLET_KEY_RE = re.compile(
    r"^.*\.(?:linear_qkv|linear_proj|linear_fc1|linear_fc2|router)\.weight_(?:packed|scale|shape)$"
)
_EXPERT_INT4_TRIPLET_KEY_RE = re.compile(r"^.*\.experts\.linear_fc[12]\.weight\d+_(?:packed|scale|shape)$")
_EXPECTED_DENSE_INT4_MISSING_KEY_RE = re.compile(
    r"^.*\.(?:linear_qkv|linear_proj|linear_fc1|linear_fc2|router)\."
    r"(?:weight|weight_(?:packed|scale|shape))$"
)
_EXPECTED_EXPERT_INT4_MISSING_KEY_RE = re.compile(
    r"^.*\.experts\.linear_fc[12]\.(?:weight|weight\d+|weight\d+_(?:packed|scale|shape))$"
)


def make_oft(**kwargs):
    from megatron.bridge.orbit.oft.oft import OFT

    return OFT(**kwargs)


def build_qwen3_moe_peft_config(peft_scheme):
    from megatron.bridge.recipes.qwen import qwen3_30b_a3b_peft_config

    return qwen3_30b_a3b_peft_config(peft_scheme=peft_scheme)


def transform_sharded_state_dict_for_int4_experts(*args, **kwargs):
    from megatron.bridge.orbit.quant.int4_utils import transform_sharded_state_dict_for_int4 as _impl

    return _impl(*args, **kwargs)


def register_int4_expert_buffers_after_load(*args, **kwargs):
    from megatron.bridge.orbit.quant.int4_utils import register_int4_buffers_after_load as _impl

    return _impl(*args, **kwargs)


def transform_sharded_state_dict_for_int4_dense(*args, **kwargs):
    from megatron.bridge.orbit.low_precision.int4 import (
        transform_sharded_state_dict_for_int4_dense as _impl,
    )

    return _impl(*args, **kwargs)


def register_int4_dense_buffers_after_load(*args, **kwargs):
    from megatron.bridge.orbit.low_precision.int4 import (
        register_int4_buffers_after_load_dense as _impl,
    )

    return _impl(*args, **kwargs)


def run_finetune(config) -> None:
    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.gpt_step import forward_step

    finetune(config=config, forward_step_func=forward_step)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-30B-A3B MoE QOFT finetuning (INT4 W4A16 base weights)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        required=True,
        help="Path to INT4 Megatron checkpoint from convert_int4_checkpoint_direct.py",
    )
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--ep", type=int, default=2)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--sp", action="store_true", default=True)
    parser.add_argument("--train-iters", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--coft", action="store_true", default=False)
    parser.add_argument("--eps", type=float, default=6e-5)
    parser.add_argument("--block-share", action="store_true", default=False)
    parser.add_argument("--module-dropout", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args(argv)


def build_config(args):
    oft = make_oft(
        block_size=args.block_size,
        coft=args.coft,
        eps=args.eps,
        block_share=args.block_share,
        module_dropout=args.module_dropout,
    )
    config = build_qwen3_moe_peft_config(peft_scheme=oft)

    config.checkpoint.pretrained_checkpoint = args.pretrained_checkpoint
    config.model.tensor_model_parallel_size = args.tp
    config.model.expert_model_parallel_size = args.ep
    config.model.pipeline_model_parallel_size = args.pp
    config.model.sequence_parallel = args.sp
    config.model.expert_tensor_parallel_size = 1
    config.model.init_model_with_meta_device = True
    config.model.perform_initialization = False
    # QOFT freezes the router, so the MoE aux loss cannot rebalance routing here.
    config.model.moe_aux_loss_coeff = 0.0

    output_dir = args.output_dir or os.path.join(os.getcwd(), "nemo_experiments", "qwen3_30b_a3b_qoft_int4")
    config.checkpoint.save = os.path.join(output_dir, "checkpoints")
    config.checkpoint.load = os.path.join(output_dir, "checkpoints")
    config.logger.tensorboard_dir = os.path.join(output_dir, "tb_logs")

    config.train.train_iters = args.train_iters
    config.scheduler.lr_warmup_iters = 2
    config.scheduler.lr_decay_iters = args.train_iters
    config.checkpoint.save_interval = 500

    config.logger.wandb_project = "megatron-bridge-finetuning"
    config.logger.wandb_exp_name = f"qwen3_30b_a3b_qoft_int4_bs{args.block_size}_tp{args.tp}_ep{args.ep}"
    return config


def _drop_extra_state_entries(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state_dict.items() if "._extra_state" not in str(key)}


def _materialize_meta_sharded_tensor_data_to_cpu(state_dict, sharded_tensor_cls) -> None:
    for value in state_dict.values():
        if isinstance(value, list):
            for sharded_tensor in value:
                if not isinstance(sharded_tensor, sharded_tensor_cls):
                    continue
                if sharded_tensor.data is not None and sharded_tensor.data.device.type == "meta":
                    sharded_tensor.data = torch.empty(
                        sharded_tensor.local_shape,
                        dtype=sharded_tensor.dtype,
                        device="cpu",
                    )
        elif isinstance(value, sharded_tensor_cls):
            if value.data is not None and value.data.device.type == "meta":
                value.data = torch.empty(value.local_shape, dtype=value.dtype, device="cpu")


def _is_int4_triplet_key(key: str) -> bool:
    return _DENSE_INT4_TRIPLET_KEY_RE.match(key) is not None or _EXPERT_INT4_TRIPLET_KEY_RE.match(key) is not None


def _is_expected_int4_missing_key(key: str) -> bool:
    return (
        key.endswith("._extra_state")
        or _EXPECTED_DENSE_INT4_MISSING_KEY_RE.match(key) is not None
        or _EXPECTED_EXPERT_INT4_MISSING_KEY_RE.match(key) is not None
    )


def install_int4_checkpoint_load_patches(group_size: int) -> None:
    import megatron.core.dist_checkpointing.strategies.torch as torch_strategy
    from megatron.core.dist_checkpointing.mapping import ShardedTensor as MCoreShardedTensor

    import megatron.bridge.training.checkpointing as checkpointing
    from megatron.bridge.models.common.unimodal import to_empty_if_meta_device

    if getattr(checkpointing, "_qwen3_moe_qoft_int4_checkpoint_patches_installed", False):
        return

    original_generate_model_state_dict = checkpointing._generate_model_state_dict
    original_mcore_to_pyt = torch_strategy.mcore_to_pyt_state_dict

    def _int4_generate_model_state_dict(model, model_sd_kwargs=None, ckpt_format="torch_dist", **kwargs):
        model_sd_kwargs = dict(model_sd_kwargs or {})
        metadata = dict(model_sd_kwargs.get("metadata") or {})
        metadata["non_homogeneous_layers"] = True
        model_sd_kwargs["metadata"] = metadata

        state_dict = original_generate_model_state_dict(model, model_sd_kwargs, ckpt_format, **kwargs)
        if ckpt_format != "torch_dist":
            return state_dict

        for model_key in list(state_dict.keys()):
            if not model_key.startswith("model"):
                continue
            model_state = _drop_extra_state_entries(state_dict[model_key])
            model_state = transform_sharded_state_dict_for_int4_experts(
                model_state,
                group_size=group_size,
                scale_dtype=INT4_SCALE_DTYPE,
            )
            model_state = transform_sharded_state_dict_for_int4_dense(
                model_state,
                group_size=group_size,
                scale_dtype=INT4_SCALE_DTYPE,
            )
            state_dict[model_key] = model_state
        return state_dict

    def _meta_safe_mcore_to_pyt(state_dict, is_loading=False, **kwargs):
        if is_loading:
            _materialize_meta_sharded_tensor_data_to_cpu(state_dict, MCoreShardedTensor)
        return original_mcore_to_pyt(state_dict, is_loading, **kwargs)

    def _int4_load_model_state_dict(module, state_dict, strict=True):
        state_dict = _drop_extra_state_entries(state_dict)
        register_int4_expert_buffers_after_load(module, state_dict)
        register_int4_dense_buffers_after_load(module, state_dict)

        for key in [key for key in state_dict if isinstance(key, str) and _is_int4_triplet_key(key)]:
            del state_dict[key]

        load_return = module.load_state_dict(state_dict, strict=False, assign=True)
        missing = [key for key in load_return.missing_keys if not _is_expected_int4_missing_key(key)]
        unexpected = [key for key in load_return.unexpected_keys if not key.endswith("._extra_state")]

        if missing or unexpected:
            details = []
            if missing:
                details.append("missing=" + ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))
            if unexpected:
                details.append("unexpected=" + ", ".join(unexpected[:20]) + (" ..." if len(unexpected) > 20 else ""))
            raise RuntimeError(
                "Unexpected non-INT4 state_dict mismatch during Qwen3 MoE INT4 load: " + " | ".join(details)
            )

        if torch.cuda.is_available():
            to_empty_if_meta_device(module, device=torch.device("cuda", torch.cuda.current_device()))

    checkpointing._generate_model_state_dict = _int4_generate_model_state_dict
    checkpointing._load_model_state_dict = _int4_load_model_state_dict
    torch_strategy.mcore_to_pyt_state_dict = _meta_safe_mcore_to_pyt
    checkpointing._qwen3_moe_qoft_int4_checkpoint_patches_installed = True


def main() -> None:
    args = parse_args()
    config = build_config(args)
    install_int4_checkpoint_load_patches(group_size=args.group_size)
    run_finetune(config=config)


if __name__ == "__main__":
    main()
