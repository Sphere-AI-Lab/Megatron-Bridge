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

"""Generic PEFT finetuning entrypoint, driven by an HF model path.

Replaces the per-model ``finetune_{oft,qoft,qoft_fp8,qoft_nvfp4}.py`` scripts for
every model whose recipe is "``_peft_common()`` plus an HF path". The upstream
per-model recipes (``qwen3_14b_peft_config``, ``gpt_oss_20b_peft_config``, ...)
are ~92 lines each that differ only in the HF path string and a couple of tuning
numbers, and ``_peft_common`` documents the contract directly: the caller MUST
set ``cfg.model`` and ``cfg.tokenizer.tokenizer_model``. That is what this does.

    torchrun --nproc_per_node=8 scripts/orbit/finetune_peft.py \
        --model-path Qwen/Qwen3-14B \
        --pretrained-checkpoint ./checkpoints/Qwen3-14B-NVFP4 \
        --peft oft --quant nvfp4 --tp 1 --pp 1

Scope: ``--quant`` in {none, fp8, mxfp8, nvfp4}. INT4 is deliberately NOT handled
here — the INT4 path additionally installs a checkpoint monkey-patch stack (see
``scripts/orbit/models/qwen3_moe/finetune_qoft_int4.py`` and the kimi/moonlight
entrypoints), which is real machinery rather than a config flag. Use those
entrypoints for INT4 until that stack is extracted into the orbit package.
"""

import argparse
import os

from megatron.bridge import AutoBridge
from megatron.bridge.orbit.oft.oft import OFT
from megatron.bridge.peft.dora import DoRA
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.recipes.common import _peft_common
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.mixed_precision import get_mixed_precision_config


# Quantization presets. Each entry is applied on top of the _peft_common base.
# Settings are taken from the per-model entrypoints they replace, not invented:
#   fp8    <- scripts/orbit/models/qwen3_moe/finetune_oft_fp8.py
#   nvfp4  <- scripts/orbit/models/qwen3_14b/finetune_qoft_nvfp4.py
#             scripts/orbit/models/qwen3_moe/finetune_qoft_nvfp4.py
QUANT_PRESETS: dict[str, dict] = {
    "none": {
        "mixed_precision": None,
        "model": {},
    },
    "fp8": {
        "mixed_precision": "bf16_with_fp8_current_scaling_mixed",
        "model": {"moe_router_padding_for_fp8": True},
    },
    "mxfp8": {
        "mixed_precision": "bf16_with_mxfp8_mixed",
        "model": {"moe_router_padding_for_fp8": True},
    },
    "nvfp4": {
        "mixed_precision": "bf16_with_nvfp4_mixed",
        "model": {
            "restore_modelopt_state": True,
            "init_model_with_meta_device": True,
            "gradient_accumulation_fusion": False,
            "use_arbitrary_attention_mask": False,
        },
        "async_strategy": "mcore",
    },
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model-path", required=True, help="HF model id or local path (architecture + tokenizer)")
    p.add_argument("--pretrained-checkpoint", required=True, help="Megatron checkpoint to finetune from")
    p.add_argument("--peft", choices=["oft", "lora", "dora", "none"], default="oft")
    p.add_argument("--quant", choices=sorted(QUANT_PRESETS), default="none")
    p.add_argument("--output-dir", default=None)

    par = p.add_argument_group("parallelism")
    par.add_argument("--tp", type=int, default=1)
    par.add_argument("--pp", type=int, default=1)
    par.add_argument("--ep", type=int, default=1)
    par.add_argument("--cp", type=int, default=1)

    tr = p.add_argument_group("training")
    tr.add_argument("--train-iters", type=int, default=None)
    tr.add_argument("--seq-length", type=int, default=None)
    tr.add_argument("--global-batch-size", type=int, default=None)
    tr.add_argument("--micro-batch-size", type=int, default=None)
    tr.add_argument("--recompute", action="store_true", help="Full uniform recompute (large models)")

    oft = p.add_argument_group("OFT")
    oft.add_argument("--block-size", type=int, default=32)
    oft.add_argument("--coft", action="store_true", default=False)
    oft.add_argument("--eps", type=float, default=6e-5)
    oft.add_argument("--block-share", action="store_true", default=False)

    lora = p.add_argument_group("LoRA / DoRA")
    lora.add_argument("--dim", type=int, default=32)
    lora.add_argument("--alpha", type=int, default=32)
    lora.add_argument("--dropout", type=float, default=0.0)

    return p.parse_args(argv)


def build_peft(args):
    """Return the PEFT object, or None for full finetuning.

    All 68 upstream ``*_peft_config`` recipes accept ``peft_scheme: str | PEFT``,
    so a PEFT instance is a first-class argument -- lora/oft need no separate
    code path.
    """
    if args.peft == "none":
        return None
    if args.peft == "oft":
        return OFT(
            block_size=args.block_size,
            coft=args.coft,
            eps=args.eps,
            block_share=args.block_share,
        )
    cls = DoRA if args.peft == "dora" else LoRA
    return cls(dim=args.dim, alpha=args.alpha, dropout=args.dropout)


def validate_parallelism(args) -> None:
    """Reject the ModelOpt quantized-MoE layout that cannot work.

    ModelOpt's QuantSequentialMLP forbids TP>1 and EP>1 simultaneously for
    quantized MoE. INT4 escapes this (it does not route through
    QuantSequentialMLP) but the ModelOpt-backed formats do not.
    """
    preset = QUANT_PRESETS[args.quant]
    if preset["model"].get("restore_modelopt_state") and args.tp > 1 and args.ep > 1:
        raise SystemExit(
            f"--quant {args.quant} cannot use TP>1 and EP>1 at the same time "
            f"(got TP={args.tp}, EP={args.ep}): ModelOpt's QuantSequentialMLP "
            f"forbids it for quantized MoE. Set one of them to 1."
        )


def build_config(args):
    validate_parallelism(args)

    cfg = _peft_common()

    # _peft_common docstring: "The caller MUST set cfg.model and
    # cfg.tokenizer.tokenizer_model before use." The HF path supplies both.
    cfg.model = AutoBridge.from_hf_pretrained(args.model_path).to_megatron_provider(load_weights=False)
    cfg.tokenizer.tokenizer_model = args.model_path

    cfg.peft = build_peft(args)
    cfg.checkpoint.pretrained_checkpoint = args.pretrained_checkpoint

    cfg.model.tensor_model_parallel_size = args.tp
    cfg.model.pipeline_model_parallel_size = args.pp
    cfg.model.expert_model_parallel_size = args.ep
    cfg.model.context_parallel_size = args.cp
    if args.cp > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = args.cp * 2

    preset = QUANT_PRESETS[args.quant]
    for key, value in preset["model"].items():
        setattr(cfg.model, key, value)
    if preset["mixed_precision"] is not None:
        cfg.mixed_precision = get_mixed_precision_config(preset["mixed_precision"])
    if "async_strategy" in preset:
        cfg.checkpoint.async_strategy = preset["async_strategy"]

    if args.seq_length is not None:
        cfg.model.seq_length = args.seq_length
        cfg.dataset.seq_length = args.seq_length
        cfg.dataset.packed_sequence_specs.packed_sequence_size = args.seq_length
    if args.train_iters is not None:
        cfg.train.train_iters = args.train_iters
    if args.global_batch_size is not None:
        cfg.train.global_batch_size = args.global_batch_size
    if args.micro_batch_size is not None:
        cfg.train.micro_batch_size = args.micro_batch_size

    if args.recompute:
        cfg.model.recompute_granularity = "full"
        cfg.model.recompute_method = "uniform"
        cfg.model.recompute_num_layers = 1

    slug = f"{os.path.basename(args.model_path.rstrip('/'))}_{args.peft}_{args.quant}".lower()
    output_dir = args.output_dir or os.path.join(os.getcwd(), "nemo_experiments", slug)
    cfg.checkpoint.save = os.path.join(output_dir, "checkpoints")
    cfg.checkpoint.load = os.path.join(output_dir, "checkpoints")

    return cfg


def main() -> None:
    args = parse_args()
    finetune(config=build_config(args), forward_step_func=forward_step)


if __name__ == "__main__":
    main()
