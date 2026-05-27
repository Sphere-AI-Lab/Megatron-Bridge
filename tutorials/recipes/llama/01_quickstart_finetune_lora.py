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
Quickstart: Finetune Llama 3.2 1B with LoRA and W&B logging

Usage:
    Single GPU with LoRA:
        torchrun --nproc_per_node=1 01_quickstart_finetune_lora.py \
            --pretrained-checkpoint /path/to/megatron/checkpoint

    Multiple GPUs (automatic data parallelism):
        torchrun --nproc_per_node=8 01_quickstart_finetune_lora.py \
            --pretrained-checkpoint /path/to/megatron/checkpoint

Prerequisites:
    You need a checkpoint in Megatron format. You can either:
    1. Convert HF checkpoint to Megatron format:
       python examples/conversion/convert_checkpoints.py import \
           --hf-model meta-llama/Llama-3.2-1B \
           --megatron-path ./checkpoints/llama32_1b
    2. Use a checkpoint from pretraining (see 00_quickstart_pretrain.py)

The script uses SQuAD dataset by default. See inline comments for:
- Using your own dataset
- Adjusting LoRA hyperparameters
- Switching to full supervised finetuning
"""

import argparse
import os

from megatron.bridge.models.gpt_provider import local_layer_spec
from megatron.bridge.recipes.llama import llama32_1b_peft_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Finetune Llama 3.2 1B with LoRA",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        required=True,
        help="Path to pretrained checkpoint in Megatron format",
    )
    return parser.parse_args()


def main() -> None:
    """Run Llama 3.2 1B finetuning with LoRA."""
    args = parse_args()

    # Load the base finetune configuration with LoRA
    config = llama32_1b_peft_config(peft_scheme="lora")

    # Load from the pretrained checkpoint
    config.checkpoint.pretrained_checkpoint = args.pretrained_checkpoint

    # Use local (non-TE) layer spec to avoid fused LN+Linear layers
    config.model.transformer_layer_spec = local_layer_spec
    config.model.persist_layer_norm = False
    config.dataset.packed_sequence_specs = None  # local DotProductAttention doesn't support packed seq

    # Save training checkpoints to a separate directory (not the pretrained checkpoint path)
    output_dir = os.path.join(os.getcwd(), "nemo_experiments", "llama32_1b_lora")
    config.checkpoint.save = os.path.join(output_dir, "checkpoints")
    config.checkpoint.load = os.path.join(output_dir, "checkpoints")
    config.logger.tensorboard_dir = os.path.join(output_dir, "tb_logs")

    # === Quick test run ===
    config.train.train_iters = 10
    config.scheduler.lr_warmup_iters = 2

    # W&B logging
    config.logger.wandb_project = "megatron-bridge-finetuning"
    config.logger.wandb_exp_name = "llama32_1b_lora"

    # ===== OPTIONAL CUSTOMIZATIONS =====
    # Uncomment and modify as needed:

    # === Use your own dataset ===
    # config.dataset.dataset_root = "/path/to/your/dataset"

    # === Adjust learning rate ===
    # config.optimizer.lr = 5e-5

    # === Change checkpoint save frequency ===
    # config.train.save_interval = 100

    # === Adjust LoRA hyperparameters ===
    # Higher rank = more trainable parameters, potentially better quality but slower
    # config.peft.dim = 16  # LoRA rank
    # config.peft.alpha = 32  # LoRA alpha scaling

    # === Full supervised finetuning (no LoRA) ===
    # config = llama32_1b_peft_config(peft_scheme=None)
    # config.checkpoint.pretrained_checkpoint = args.pretrained_checkpoint

    # Start finetuning
    finetune(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
