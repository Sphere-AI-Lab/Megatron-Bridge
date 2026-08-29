#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
GPT-OSS 20B MoE OFT finetuning (interactive, no SLURM).

OFT (Orthogonal Fine-Tuning) learns block-diagonal orthogonal rotations
applied to the input of linear layers, preserving angular structure.

Usage:
    torchrun --nproc_per_node=4 scripts/orbit/models/gpt_oss/finetune_oft.py \
        --pretrained-checkpoint /path/to/gpt-oss-20b

    With custom parallelism:
        torchrun --nproc_per_node=4 scripts/orbit/models/gpt_oss/finetune_oft.py \
            --pretrained-checkpoint /path/to/gpt-oss-20b \
            --tp 2 --ep 2

    With Constrained OFT:
        torchrun --nproc_per_node=4 scripts/orbit/models/gpt_oss/finetune_oft.py \
            --pretrained-checkpoint /path/to/gpt-oss-20b \
            --coft --eps 6e-5
"""

import argparse
import os

from megatron.bridge.orbit.oft.oft import OFT
from megatron.bridge.recipes.gpt_oss import gpt_oss_20b_peft_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPT-OSS 20B MoE OFT finetuning",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        required=True,
        help="Path to pretrained checkpoint in Megatron format",
    )
    # Parallelism
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size (default: 2)")
    parser.add_argument("--ep", type=int, default=2, help="Expert parallel size (default: 2)")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size (default: 1)")
    parser.add_argument("--sp", action="store_true", default=True, help="Enable sequence parallel (default: True)")
    parser.add_argument("--train-iters", type=int, default=10, help="Number of training iterations (default: 10)")
    # OFT hyperparameters
    parser.add_argument("--block-size", type=int, default=32, help="OFT block size (default: 32)")
    parser.add_argument("--coft", action="store_true", default=False, help="Use Constrained OFT")
    parser.add_argument("--eps", type=float, default=6e-5, help="COFT epsilon (default: 6e-5)")
    parser.add_argument("--block-share", action="store_true", default=False, help="Share params across blocks")
    parser.add_argument("--module-dropout", type=float, default=0.0, help="Multiplicative dropout (default: 0.0)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for checkpoints and logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    oft = OFT(
        block_size=args.block_size,
        coft=args.coft,
        eps=args.eps,
        block_share=args.block_share,
        module_dropout=args.module_dropout,
    )
    config = gpt_oss_20b_peft_config(peft_scheme=oft)

    # Checkpoint
    config.checkpoint.pretrained_checkpoint = args.pretrained_checkpoint

    # Parallelism for 4 GPUs (default: TP=2, EP=2)
    config.model.tensor_model_parallel_size = args.tp
    config.model.expert_model_parallel_size = args.ep
    config.model.pipeline_model_parallel_size = args.pp
    config.model.sequence_parallel = args.sp
    config.model.expert_tensor_parallel_size = 1  # ETP must be 1 when bias is enabled

    # Save training checkpoints to a separate directory (not the pretrained checkpoint path)
    output_dir = args.output_dir or os.path.join(os.getcwd(), "nemo_experiments", "gpt_oss_20b_oft")
    config.checkpoint.save = os.path.join(output_dir, "checkpoints")
    config.checkpoint.load = os.path.join(output_dir, "checkpoints")
    config.logger.tensorboard_dir = os.path.join(output_dir, "tb_logs")

    # Training
    config.train.train_iters = args.train_iters
    config.scheduler.lr_warmup_iters = 2
    config.checkpoint.save_interval = 500

    # W&B logging
    config.logger.wandb_project = "megatron-bridge-finetuning"
    config.logger.wandb_exp_name = f"gpt_oss_20b_oft_bs{args.block_size}_tp{args.tp}_ep{args.ep}"

    finetune(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
