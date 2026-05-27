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
Qwen3-30B-A3B MoE LoRA finetuning with FP8 (interactive, no SLURM).

Supports two FP8 recipes:
  - "hopper": FP8 current scaling (H100+)
  - "blackwell": MXFP8 (B200+)

Usage:
    torchrun --nproc_per_node=4 examples/models/qwen3_moe/finetune_lora_fp8.py \
        --pretrained-checkpoint /path/to/qwen3-30b-a3b \
        --tp 2 --ep 2 --fp8-recipe hopper

    torchrun --nproc_per_node=4 examples/models/qwen3_moe/finetune_lora_fp8.py \
        --pretrained-checkpoint /path/to/qwen3-30b-a3b \
        --tp 2 --ep 2 --fp8-recipe blackwell
"""

import argparse
import os

from megatron.bridge.recipes.qwen import qwen3_30b_a3b_peft_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.mixed_precision import get_mixed_precision_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-30B-A3B MoE LoRA finetuning with FP8",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        required=True,
        help="Path to pretrained checkpoint in Megatron format",
    )
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size (default: 2)")
    parser.add_argument("--ep", type=int, default=2, help="Expert parallel size (default: 2)")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size (default: 1)")
    parser.add_argument("--sp", action="store_true", default=True, help="Enable sequence parallel (default: True)")
    parser.add_argument("--train-iters", type=int, default=10, help="Number of training iterations (default: 10)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for checkpoints and logs")
    parser.add_argument(
        "--fp8-recipe",
        type=str,
        default="hopper",
        choices=["hopper", "blackwell"],
        help="FP8 recipe: 'hopper' for FP8 current scaling (H100+), 'blackwell' for MXFP8 (B200+)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Start from the Qwen3-30B-A3B PEFT config (BF16 by default)
    config = qwen3_30b_a3b_peft_config(peft_scheme="lora")

    # Enable FP8 mixed precision
    if args.fp8_recipe == "hopper":
        config.mixed_precision = get_mixed_precision_config("bf16_with_fp8_current_scaling_mixed")
    elif args.fp8_recipe == "blackwell":
        config.mixed_precision = get_mixed_precision_config("bf16_with_mxfp8_mixed")
    config.model.moe_router_padding_for_fp8 = True

    # Disable CUDA graphs and torch.compile for easier debugging
    config.model.cuda_graph_impl = "none"

    # Checkpoint
    config.checkpoint.pretrained_checkpoint = args.pretrained_checkpoint

    # Parallelism
    config.model.tensor_model_parallel_size = args.tp
    config.model.expert_model_parallel_size = args.ep
    config.model.pipeline_model_parallel_size = args.pp
    config.model.sequence_parallel = args.sp
    config.model.expert_tensor_parallel_size = 1

    # Output directory
    output_dir = args.output_dir or os.path.join(os.getcwd(), "nemo_experiments", "qwen3_30b_a3b_lora_fp8")
    config.checkpoint.save = os.path.join(output_dir, "checkpoints")
    config.checkpoint.load = os.path.join(output_dir, "checkpoints")
    config.logger.tensorboard_dir = os.path.join(output_dir, "tb_logs")

    # Training
    config.train.train_iters = args.train_iters
    config.scheduler.lr_warmup_iters = 2
    config.checkpoint.save_interval = 500

    # W&B logging
    config.logger.wandb_project = "megatron-bridge-finetuning"
    config.logger.wandb_exp_name = f"qwen3_30b_a3b_lora_fp8_{args.fp8_recipe}_tp{args.tp}_ep{args.ep}"

    finetune(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
