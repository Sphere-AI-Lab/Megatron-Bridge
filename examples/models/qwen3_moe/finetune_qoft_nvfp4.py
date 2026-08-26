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

"""Qwen3-30B-A3B MoE QOFT finetuning with NVFP4 base weights.

Smaller-scale companion to ``examples/models/kimi_k25/finetune_qoft_nvfp4.py``
intended for fast iteration on the NVFP4 + grouped-MoE codepath. Same flow:
``restore_modelopt_state=True`` + ``init_model_with_meta_device=True`` +
``bf16_with_nvfp4_mixed`` mixed precision. The ModelOpt grouped-MoE patches
in ``bridge/training/post_training/checkpointing.py`` apply automatically.

Architecture (Qwen3-30B-A3B): 48 layers, 128 routed experts, hidden=2048,
moe_intermediate=768.

    Prerequisites:
    Convert HF NVFP4 -> Megatron checkpoint:
        bash convert_nvfp4_checkpoint_direct.sh \\
            ${HF_MODEL_ROOT:-${HOME}/hf_models}/Qwen3-30B-A3B-NVFP4 \\
            ./checkpoints/Qwen3-30B-A3B-NVFP4

Usage:
    torchrun --nproc_per_node=8 \\
        examples/models/qwen3_moe/finetune_qoft_nvfp4.py \\
        --pretrained-checkpoint ./checkpoints/Qwen3-30B-A3B-NVFP4 \\
        --tp 2 --ep 4
"""

import argparse
import os
import re

import torch

from megatron.bridge.orbit.oft.oft import OFT
from megatron.bridge.recipes.qwen.qwen3_moe import qwen3_30b_a3b_peft_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.mixed_precision import get_mixed_precision_config


# Standard Qwen3 (non-MLA) target modules — see TARGET_MODULES in
# orbit/scripts/low_precision/run_qwen3_30b_a3b_nvfp4_math_megatron_oft.sh.
QWEN3_MOE_OFT_TARGET_MODULES = [
    "linear_qkv",
    "linear_proj",
    "linear_fc1",
    "linear_fc2",
]


def _parse_target_modules(value: str) -> list[str]:
    return [m for m in re.split(r"[\s,]+", value.strip()) if m]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-30B-A3B MoE QOFT finetuning (NVFP4 base weights)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--pretrained-checkpoint", type=str, required=True)
    parser.add_argument("--hf-model-path", type=str, default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--sp", action="store_true", default=True)
    parser.add_argument("--train-iters", type=int, default=10)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument(
        "--distributed-timeout-minutes",
        type=int,
        default=None,
        help="Override torch.distributed process-group timeout in minutes.",
    )
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--coft", action="store_true", default=False)
    parser.add_argument("--eps", type=float, default=6e-5)
    parser.add_argument("--block-share", action="store_true", default=False)
    parser.add_argument("--module-dropout", type=float, default=0.0)
    parser.add_argument(
        "--target-modules",
        type=_parse_target_modules,
        default=list(QWEN3_MOE_OFT_TARGET_MODULES),
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        default=False,
    )
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--skip-train", action="store_true", default=False)
    parser.add_argument("--skip-eval", action="store_true", default=False)
    return parser.parse_args(argv)


def _disable_evaluation(config) -> None:
    config.validation.eval_iters = 0
    config.validation.eval_interval = None


def build_config(args) -> object:
    oft = OFT(
        target_modules=args.target_modules,
        block_size=args.block_size,
        coft=args.coft,
        eps=args.eps,
        block_share=args.block_share,
        module_dropout=args.module_dropout,
    )
    config = qwen3_30b_a3b_peft_config(peft_scheme=oft)

    if args.distributed_timeout_minutes is not None:
        config.dist.distributed_timeout_minutes = args.distributed_timeout_minutes

    config.checkpoint.pretrained_checkpoint = args.pretrained_checkpoint

    # Parallelism
    config.model.tensor_model_parallel_size = args.tp
    config.model.expert_model_parallel_size = args.ep
    config.model.pipeline_model_parallel_size = args.pp
    config.model.sequence_parallel = args.sp
    config.model.expert_tensor_parallel_size = 1

    # Build the model on meta to keep RAM low (mirror of the kimi_k25 NVFP4
    # path — the framework's ModelOpt-compress patches handle meta-device
    # weights so SwiGLU sharded-state-dict factories see the packed shape).
    config.model.init_model_with_meta_device = True
    config.model.perform_initialization = False

    # NVFP4 path — restore modelopt-quantized layers from the saved modelopt
    # state in the converted checkpoint. The grouped-MoE-safe compress patches
    # in bridge/training/post_training/checkpointing.py apply automatically.
    config.model.restore_modelopt_state = True
    config.model.gradient_accumulation_fusion = False

    # Force GroupedLinear MoE layout (TEColumnParallelGroupedLinear with
    # `weightN_v/_w` per-expert params on a single module). This matches the
    # checkpoint produced by `convert_nvfp4_checkpoint_direct.py`. Without
    # this, `qwen3_30b_a3b_peft_config()` defaults to SequentialMLP
    # (`experts.experts.N.linear_fc1.weight_v/_w`), and the model's
    # sharded_state_dict requests keys that don't exist in the checkpoint.
    config.model.moe_grouped_gemm = True
    config.model.moe_token_dispatcher_type = "alltoall"
    config.model.moe_permute_fusion = True
    config.model.moe_router_fusion = False
    config.model.moe_shared_expert_overlap = True

    # QOFT freezes the router; aux loss has nothing to rebalance against.
    config.model.moe_aux_loss_coeff = 0.0

    # Mixed precision: BF16 activations + grads, NVFP4 base weights.
    config.mixed_precision = get_mixed_precision_config("bf16_with_nvfp4_mixed")

    # Training
    config.train.train_iters = args.train_iters
    config.train.global_batch_size = args.global_batch_size
    config.train.micro_batch_size = args.micro_batch_size

    # Sequence length
    config.model.seq_length = args.seq_length
    if getattr(config, "dataset", None) is not None:
        config.dataset.seq_length = args.seq_length
        packed_specs = getattr(config.dataset, "packed_sequence_specs", None)
        if packed_specs is not None:
            packed_specs.packed_sequence_size = args.seq_length

    # Scheduler
    config.scheduler.lr_warmup_iters = 2
    config.scheduler.lr_decay_iters = args.train_iters

    # Tokenizer — prefer assets saved in the converted checkpoint.
    checkpoint_tokenizer_dir = os.path.join(
        args.pretrained_checkpoint, "iter_{:07d}".format(0), "tokenizer"
    )
    tokenizer_model = (
        checkpoint_tokenizer_dir
        if os.path.isdir(checkpoint_tokenizer_dir)
        else args.hf_model_path
    )
    config.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    config.tokenizer.tokenizer_model = tokenizer_model
    config.tokenizer.hf_tokenizer_kwargs = {"trust_remote_code": True}

    # Checkpoint save
    config.checkpoint.save_interval = args.save_interval if args.save_checkpoints else 0
    config.checkpoint.async_save = False
    config.checkpoint.async_strategy = "mcore"

    if args.skip_eval:
        _disable_evaluation(config)
    if args.skip_train:
        config.validation.skip_train = True
        _disable_evaluation(config)

    output_dir = args.output_dir or os.path.join(
        os.getcwd(), "nemo_experiments", "qwen3_30b_a3b_qoft_nvfp4"
    )
    if args.save_checkpoints:
        config.checkpoint.save = os.path.join(output_dir, "checkpoints")
        config.checkpoint.load = os.path.join(output_dir, "checkpoints")
    else:
        config.checkpoint.save = None
        config.checkpoint.load = None
        config.logger.log_progress = False
        config.logger.wandb_save_dir = os.path.join(output_dir, "wandb")
    config.logger.tensorboard_dir = os.path.join(output_dir, "tb_logs")

    config.logger.log_interval = 1
    config.logger.wandb_project = "megatron-bridge-finetuning"
    config.logger.wandb_exp_name = (
        f"qwen3_30b_a3b_qoft_nvfp4_bs{args.block_size}_tp{args.tp}_ep{args.ep}"
    )

    return config


def main() -> None:
    args = parse_args()
    config = build_config(args)
    finetune(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
