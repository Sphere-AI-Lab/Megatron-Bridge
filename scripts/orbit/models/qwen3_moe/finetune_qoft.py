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
Qwen3-30B-A3B MoE QOFT finetuning — FP8 base weights + BF16 OFT adapters.

QOFT (Quantized OFT) keeps the base model weights in FP8 permanently
(analogous to QLoRA keeping weights in NF4).  Only the OFT rotation
parameters are in BF16 and receive gradient updates.

Prerequisites:
    Convert the HF FP8 checkpoint to Megatron format (preserving FP8):

        python scripts/orbit/conversion/convert_fp8_checkpoint.py \\
            --hf-model-path /path/to/Qwen3-30B-A3B-FP8 \\
            --megatron-path ./checkpoints/Qwen3-30B-A3B-FP8

Usage:
    torchrun --nproc_per_node=4 scripts/orbit/models/qwen3_moe/finetune_qoft.py \\
        --pretrained-checkpoint ./checkpoints/Qwen3-30B-A3B-FP8 \\
        --tp 2 --ep 2
"""

import argparse
import math
import os

import torch


FP8_WEIGHT_BLOCK_SIZE = 128


def make_oft(**kwargs):
    from megatron.bridge.orbit.oft.oft import OFT

    return OFT(**kwargs)


def build_qwen3_moe_peft_config(peft_scheme):
    from megatron.bridge.recipes.qwen import qwen3_30b_a3b_peft_config

    return qwen3_30b_a3b_peft_config(peft_scheme=peft_scheme)


def transform_sharded_state_dict_for_fp8(*args, **kwargs):
    from megatron.bridge.orbit.quant.fp8_utils import transform_sharded_state_dict_for_fp8 as _impl

    return _impl(*args, **kwargs)


def register_fp8_scale_inv_buffers_after_load(*args, **kwargs):
    from megatron.bridge.orbit.quant.fp8_utils import register_fp8_scale_inv_buffers_after_load as _impl

    return _impl(*args, **kwargs)


def run_finetune(config) -> None:
    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.gpt_step import forward_step

    finetune(config=config, forward_step_func=forward_step)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-30B-A3B MoE QOFT finetuning (FP8 base weights)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        required=True,
        help="Path to FP8 Megatron checkpoint (converted via convert_fp8_checkpoint.py)",
    )
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--ep", type=int, default=2)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--sp", action="store_true", default=True)
    parser.add_argument("--train-iters", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--coft", action="store_true", default=False)
    parser.add_argument("--eps", type=float, default=6e-5)
    parser.add_argument("--block-share", action="store_true", default=False)
    parser.add_argument("--module-dropout", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args(argv)


def infer_fp8_scale_inv_shape(
    weight: torch.Tensor,
    *,
    block_size: int = FP8_WEIGHT_BLOCK_SIZE,
) -> tuple[int, ...]:
    if weight.ndim == 1:
        return (max(1, math.ceil(weight.shape[0] / block_size)),)
    if weight.ndim >= 2:
        leading_dims = tuple(weight.shape[:-2])
        out_blocks = max(1, math.ceil(weight.shape[-2] / block_size))
        in_blocks = max(1, math.ceil(weight.shape[-1] / block_size))
        return leading_dims + (out_blocks, in_blocks)
    raise ValueError(f"Unsupported FP8 weight rank {weight.ndim}")


def ensure_fp8_scale_inv_buffers(model):
    if not isinstance(model, list):
        model = [model]

    for root in model:
        for module in root.modules():
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor):
                continue
            if weight.dtype != torch.float8_e4m3fn or weight.numel() == 0:
                continue

            expected_shape = infer_fp8_scale_inv_shape(weight)
            existing = getattr(module, "weight_scale_inv", None)
            if (
                isinstance(existing, torch.Tensor)
                and tuple(existing.shape) == expected_shape
                and existing.dtype == torch.float32
                and existing.device == weight.device
            ):
                continue

            new_buffer = torch.ones(expected_shape, dtype=torch.float32, device=weight.device)
            if "weight_scale_inv" in module._buffers:
                module._buffers["weight_scale_inv"] = new_buffer
            else:
                module.register_buffer("weight_scale_inv", new_buffer, persistent=True)

    return model


def build_config(args):
    oft = make_oft(
        block_size=args.block_size,
        coft=args.coft,
        eps=args.eps,
        block_share=args.block_share,
        module_dropout=args.module_dropout,
    )
    config = build_qwen3_moe_peft_config(peft_scheme=oft)
    config.model.register_pre_wrap_hook(ensure_fp8_scale_inv_buffers)

    config.checkpoint.pretrained_checkpoint = args.pretrained_checkpoint
    config.model.tensor_model_parallel_size = args.tp
    config.model.expert_model_parallel_size = args.ep
    config.model.pipeline_model_parallel_size = args.pp
    config.model.sequence_parallel = args.sp
    config.model.expert_tensor_parallel_size = 1

    output_dir = args.output_dir or os.path.join(os.getcwd(), "nemo_experiments", "qwen3_30b_a3b_qoft")
    config.checkpoint.save = os.path.join(output_dir, "checkpoints")
    config.checkpoint.load = os.path.join(output_dir, "checkpoints")
    config.logger.tensorboard_dir = os.path.join(output_dir, "tb_logs")

    config.train.train_iters = args.train_iters
    config.scheduler.lr_warmup_iters = 2
    config.checkpoint.save_interval = 500

    config.logger.wandb_project = "megatron-bridge-finetuning"
    config.logger.wandb_exp_name = (
        f"qwen3_30b_a3b_qoft_bs{args.block_size}_tp{args.tp}_ep{args.ep}"
    )
    return config


def install_fp8_checkpoint_load_patches() -> None:
    import megatron.bridge.training.checkpointing as _ckpt_mod

    if getattr(_ckpt_mod, "_qwen3_moe_qoft_fp8_checkpoint_patches_installed", False):
        return

    _orig_gen_model_sd = _ckpt_mod._generate_model_state_dict
    _orig_load_model_sd = _ckpt_mod._load_model_state_dict

    def _fp8_generate_model_state_dict(model, model_sd_kwargs=None, ckpt_format="torch_dist", **kwargs):
        state_dict = _orig_gen_model_sd(model, model_sd_kwargs, ckpt_format, **kwargs)
        for model_key in list(state_dict.keys()):
            if model_key.startswith("model"):
                state_dict[model_key] = transform_sharded_state_dict_for_fp8(state_dict[model_key])
        return state_dict

    def _fp8_load_model_state_dict(model_module, state_dict, strict=True):
        register_fp8_scale_inv_buffers_after_load(model_module, state_dict)
        _orig_load_model_sd(model_module, state_dict, strict)

    _ckpt_mod._generate_model_state_dict = _fp8_generate_model_state_dict
    _ckpt_mod._load_model_state_dict = _fp8_load_model_state_dict
    _ckpt_mod._qwen3_moe_qoft_fp8_checkpoint_patches_installed = True


def main() -> None:
    args = parse_args()
    config = build_config(args)
    install_fp8_checkpoint_load_patches()
    run_finetune(config=config)


if __name__ == "__main__":
    main()
