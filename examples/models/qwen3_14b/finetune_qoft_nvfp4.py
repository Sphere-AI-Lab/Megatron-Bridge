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

import argparse
import os

from megatron.bridge.sphere.oft.oft import OFT
from megatron.bridge.recipes.qwen import qwen3_14b_peft_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.mixed_precision import get_mixed_precision_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Qwen3-14B NVFP4 dense QOFT finetuning")
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--train-iters", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--coft", action="store_true", default=False)
    parser.add_argument("--eps", type=float, default=6e-5)
    parser.add_argument("--block-share", action="store_true", default=False)
    return parser.parse_args(argv)


def build_config(args):
    oft = OFT(
        block_size=args.block_size,
        coft=args.coft,
        eps=args.eps,
        block_share=args.block_share,
    )
    cfg = qwen3_14b_peft_config(peft_scheme=oft)
    cfg.checkpoint.pretrained_checkpoint = args.pretrained_checkpoint
    # Direct NVFP4 checkpoints are expected to carry ModelOpt quantization state.
    cfg.model.restore_modelopt_state = True
    cfg.model.use_arbitrary_attention_mask = False
    cfg.model.gradient_accumulation_fusion = False
    cfg.model.tensor_model_parallel_size = args.tp
    cfg.model.pipeline_model_parallel_size = args.pp
    cfg.train.train_iters = args.train_iters
    cfg.mixed_precision = get_mixed_precision_config("bf16_with_nvfp4_mixed")
    cfg.checkpoint.async_strategy = "mcore"
    output_dir = args.output_dir or os.path.join(os.getcwd(), "nemo_experiments", "qwen3_14b_qoft_nvfp4")
    cfg.checkpoint.save = os.path.join(output_dir, "checkpoints")
    cfg.checkpoint.load = os.path.join(output_dir, "checkpoints")
    return cfg


def main():
    args = parse_args()
    cfg = build_config(args)
    finetune(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
