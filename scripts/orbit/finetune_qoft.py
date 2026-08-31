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

"""Generic quantized-base QOFT finetuning entrypoint.

The base model weights stay in their quantized format permanently; only the
OFT parameters are BF16 and receive gradient updates. Requires a converted
Megatron checkpoint from ``scripts/orbit/conversion/`` (see scripts/orbit/README.md).

Replaces the per-model ``models/*/finetune_qoft*.py`` entrypoints. The
architecture is detected from the HF config and each architecture keeps the
exact recipe base and settings its retired entrypoint used:

    Qwen3MoeForCausalLM             fp8 | int4 | nvfp4   (qwen3_30b_a3b recipe)
    Qwen3ForCausalLM                fp8                  (_peft_common + AutoBridge)
    KimiK25ForConditionalGeneration int4 | nvfp4         (_peft_common + AutoBridge)
    DeepseekV3ForCausalLM           int4                 (moonlight_16b recipe)

Usage:
    torchrun --nproc_per_node=4 scripts/orbit/finetune_qoft.py \\
        --quant fp8 \\
        --hf-model-path /path/to/Qwen3-30B-A3B-FP8 \\
        --pretrained-checkpoint ./checkpoints/Qwen3-30B-A3B-FP8-mcore \\
        --tp 2 --ep 2

Unlike the retired entrypoints, run-checkpoint saving is opt-in for every
architecture: pass ``--save-checkpoints`` to write checkpoints under
``--output-dir`` (the old fp8/int4/moonlight scripts always saved).
"""

import argparse
import json
import os
import sys

import torch


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models._qoft_common import (  # noqa: E402
    AdapterInitCallback,
    MemoryProfileCallback,
    NanTraceCallback,
    assert_fp8_request_keys_in_checkpoint,  # noqa: F401
    disable_evaluation,
    ensure_fp8_scale_inv_buffers,
    fp8_model_state_dict_kwargs_for_checkpoint_keys,  # noqa: F401
    install_fp8_checkpoint_load_patches,
    install_int4_checkpoint_load_patches,
    install_nvfp4_checkpoint_load_patches,
    log_model_storage_summary,
    normalize_hf_dataset_source,
    parse_target_modules,
    patch_moonshot_build_tokenizer,
    set_sequence_length,
    tokenizer_model_from_checkpoint,
)


KIMI_K25_ALL_LINEAR_OFT_TARGET_MODULES = [
    "linear_q_down_proj",
    "linear_q_up_proj",
    "linear_kv_down_proj",
    "linear_kv_up_proj",
    "linear_proj",
    "linear_fc1",
    "linear_fc2",
]

QWEN3_MOE_OFT_TARGET_MODULES = [
    "linear_qkv",
    "linear_proj",
    "linear_fc1",
    "linear_fc2",
]

# Per-architecture behavior, carried over verbatim from the retired
# per-model entrypoints. ``defaults`` fills any CLI argument left unset;
# ``quant_defaults`` refines them per quantization format.
ARCH_SPECS = {
    "Qwen3ForCausalLM": {
        "key": "qwen3_dense",
        "label": "Qwen3 dense",
        "slug": "qwen3_dense",
        "trust_remote_code": False,
        "quants": ("fp8",),
        "big_block": False,
        "int4_scope": "all",
        "validate_nonfinite": False,
        "defaults": {
            "tp": 1,
            "ep": 1,
            "sp": False,
            "train_iters": 10,
            "global_batch_size": 8,
            "micro_batch_size": 1,
            "seq_length": 2048,
            "group_size": 128,
        },
        "quant_defaults": {},
        "target_modules": {"fp8": None},
        "tokenizer_from_ckpt": {"fp8": False},
    },
    "Qwen3MoeForCausalLM": {
        "key": "qwen3_moe",
        "label": "Qwen3 MoE",
        "slug": "qwen3_30b_a3b",
        "trust_remote_code": False,
        "quants": ("fp8", "int4", "nvfp4"),
        "big_block": False,
        "int4_scope": "all",
        "validate_nonfinite": False,
        "defaults": {
            "tp": 2,
            "ep": 2,
            "sp": True,
            "train_iters": 10,
            "global_batch_size": 32,
            "micro_batch_size": 1,
            "seq_length": 2048,
            "group_size": 128,
        },
        "quant_defaults": {"nvfp4": {"ep": 4}},
        "target_modules": {"fp8": None, "int4": None, "nvfp4": list(QWEN3_MOE_OFT_TARGET_MODULES)},
        "tokenizer_from_ckpt": {"fp8": False, "int4": False, "nvfp4": True},
    },
    "KimiK25ForConditionalGeneration": {
        "key": "kimi_k25",
        "label": "Kimi",
        "slug": "kimi_k25",
        "trust_remote_code": True,
        "quants": ("int4", "nvfp4"),
        "big_block": True,
        "int4_scope": "experts",
        "validate_nonfinite": False,
        "defaults": {
            "tp": 2,
            "ep": 4,
            "sp": True,
            "train_iters": 10,
            "global_batch_size": 32,
            "micro_batch_size": 1,
            "seq_length": 2048,
            "log_interval": 1,
            "group_size": 32,
        },
        "quant_defaults": {},
        "target_modules": {
            "int4": list(KIMI_K25_ALL_LINEAR_OFT_TARGET_MODULES),
            "nvfp4": list(KIMI_K25_ALL_LINEAR_OFT_TARGET_MODULES),
        },
        "tokenizer_from_ckpt": {"int4": True, "nvfp4": True},
    },
    "DeepseekV3ForCausalLM": {
        "key": "moonlight",
        "label": "Moonlight",
        "slug": "moonlight_16b",
        "trust_remote_code": True,
        "quants": ("int4",),
        "big_block": True,
        "int4_scope": "experts",
        "validate_nonfinite": True,
        "defaults": {
            "tp": 1,
            "ep": 1,
            "sp": False,
            "train_iters": 100,
            "global_batch_size": 32,
            "micro_batch_size": 1,
            "seq_length": 2048,
            "log_interval": 10,
            "group_size": 32,
        },
        "quant_defaults": {"int4": {"int4_active_expert_chunk_size": 4, "int4_grouped_chunk_backend": "python"}},
        "target_modules": {"int4": None},
        "tokenizer_from_ckpt": {"int4": True},
    },
}


def resolve_arch(hf_model_path: str) -> dict:
    """Detect the model architecture from the HF config and return its spec."""
    config_path = os.path.join(hf_model_path, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as fh:
            hf_config = json.load(fh)
        architectures = hf_config.get("architectures") or (hf_config.get("text_config") or {}).get("architectures")
    else:
        from transformers import AutoConfig

        try:
            architectures = AutoConfig.from_pretrained(hf_model_path).architectures
        except Exception as exc:
            raise SystemExit(
                f"Could not detect the architecture of {hf_model_path!r} without trust_remote_code "
                f"({exc}). Pass a local model directory instead."
            ) from exc

    if not architectures:
        raise SystemExit(f"No architectures entry in the HF config of {hf_model_path!r}")
    arch = architectures[0]
    spec = ARCH_SPECS.get(arch)
    if spec is None:
        raise SystemExit(
            f"Architecture {arch!r} has no QOFT support. Supported: {sorted(ARCH_SPECS)}. "
            "Add an entry to ARCH_SPECS mirroring the closest existing one."
        )
    return spec


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments; per-architecture defaults fill unset values later."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--quant",
        required=True,
        choices=["fp8", "int4", "nvfp4"],
        help="Quantized-base format of the converted checkpoint",
    )
    p.add_argument(
        "--hf-model-path", required=True, help="HF model id or local path (architecture detection, config, tokenizer)"
    )
    p.add_argument(
        "--pretrained-checkpoint", required=True, help="Converted Megatron checkpoint (see scripts/orbit/conversion/)"
    )
    p.add_argument("--output-dir", default=None)

    par = p.add_argument_group("parallelism (defaults per architecture)")
    par.add_argument("--tp", type=int, default=None)
    par.add_argument("--ep", type=int, default=None)
    par.add_argument("--pp", type=int, default=1)
    sp_group = par.add_mutually_exclusive_group()
    sp_group.add_argument("--sp", dest="sp", action="store_true", default=None)
    sp_group.add_argument("--no-sp", dest="sp", action="store_false")
    par.add_argument("--distributed-timeout-minutes", type=int, default=None)

    tr = p.add_argument_group("training (defaults per architecture)")
    tr.add_argument("--train-iters", type=int, default=None)
    tr.add_argument("--global-batch-size", type=int, default=None)
    tr.add_argument("--micro-batch-size", type=int, default=None)
    tr.add_argument("--seq-length", type=int, default=None)
    tr.add_argument("--log-interval", type=int, default=None)

    oft = p.add_argument_group("OFT")
    oft.add_argument("--block-size", type=int, default=32)
    oft.add_argument("--coft", action="store_true", default=False)
    oft.add_argument("--eps", type=float, default=6e-5)
    oft.add_argument("--block-share", action="store_true", default=False)
    oft.add_argument("--module-dropout", type=float, default=0.0)
    oft.add_argument(
        "--target-modules",
        type=parse_target_modules,
        default=None,
        help="Comma/whitespace separated module names to wrap (default per architecture)",
    )

    i4 = p.add_argument_group("INT4")
    i4.add_argument(
        "--group-size",
        type=int,
        default=None,
        help="INT4 quantization group size (default per architecture -- getting this wrong "
        "silently changes checkpoint-load precision, not a crash)",
    )
    i4.add_argument(
        "--int4-active-expert-chunk-size",
        type=int,
        default=None,
        help="Active experts per INT4 chunk in grouped expert linears (0 disables; default per architecture)",
    )
    i4.add_argument("--int4-grouped-chunk-backend", choices=("python", "te"), default=None)

    modes = p.add_argument_group("modes")
    modes.add_argument(
        "--save-checkpoints",
        action="store_true",
        default=False,
        help="Save/load run checkpoints under output-dir/checkpoints",
    )
    modes.add_argument("--save-interval", type=int, default=500)
    modes.add_argument(
        "--skip-train",
        action="store_true",
        default=False,
        help="Load the checkpoint and initialize PEFT, then exit without training",
    )
    modes.add_argument("--skip-eval", action="store_true", default=False)
    modes.add_argument("--profile-memory", action="store_true", default=False)
    modes.add_argument("--profile-memory-steps", type=int, default=1)
    modes.add_argument("--debug-nan", action="store_true", default=False)
    modes.add_argument("--debug-nan-steps", type=int, default=3)
    modes.add_argument(
        "--memory-smoke-test",
        action="store_true",
        default=False,
        help="One training step with memory profiling, no save, no evaluation",
    )

    return p.parse_args(argv)


def _fill_arch_defaults(args, spec: dict, *, world_size: int | None = None) -> None:
    defaults = dict(spec["defaults"])
    defaults.update(spec["quant_defaults"].get(args.quant, {}))
    if spec["key"] == "moonlight" and args.quant == "int4" and world_size is not None:
        topology_defaults = {
            1: {"tp": 1, "ep": 1, "sp": False},
            2: {"tp": 1, "ep": 2, "sp": False},
            4: {"tp": 2, "ep": 2, "sp": True},
        }
        defaults.update(topology_defaults.get(world_size, {}))
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.target_modules is None:
        args.target_modules = spec["target_modules"][args.quant]


def build_oft(args):
    """Build the OFT adapter for a quantized base model."""
    from megatron.bridge.orbit.oft.oft import OFT

    kwargs = {
        "block_size": args.block_size,
        "coft": args.coft,
        "eps": args.eps,
        "block_share": args.block_share,
        "module_dropout": args.module_dropout,
    }
    if args.target_modules:
        kwargs["target_modules"] = args.target_modules
    return OFT(**kwargs)


def _validate_runtime_topology(args, spec: dict, *, world_size: int) -> None:
    """Retain model-specific launch constraints after launcher consolidation."""
    if spec["key"] != "moonlight" or args.quant != "int4":
        return

    supported = {
        (1, 1, 1, 1),
        (2, 1, 2, 1),
        (4, 2, 2, 1),
    }
    topology = (world_size, args.tp, args.ep, args.pp)
    if topology not in supported:
        raise SystemExit(
            "Moonlight INT4 supported GPU/parallel layouts are: "
            "WORLD_SIZE=1 TP=1 EP=1 PP=1; WORLD_SIZE=2 TP=1 EP=2 PP=1; "
            "WORLD_SIZE=4 TP=2 EP=2 PP=1. "
            f"Received WORLD_SIZE={world_size} TP={args.tp} EP={args.ep} PP={args.pp}."
        )
    if args.tp > 1 and not args.sp:
        raise SystemExit("Moonlight INT4 requires sequence parallelism when TP is greater than one; pass --sp.")


def _apply_common_model_overrides(config, args) -> None:
    config.checkpoint.pretrained_checkpoint = args.pretrained_checkpoint
    config.model.tensor_model_parallel_size = args.tp
    config.model.expert_model_parallel_size = args.ep
    config.model.pipeline_model_parallel_size = args.pp
    config.model.sequence_parallel = args.sp
    config.model.expert_tensor_parallel_size = 1
    if args.distributed_timeout_minutes is not None:
        config.dist.distributed_timeout_minutes = args.distributed_timeout_minutes


def _apply_big_model_block(config, args, spec: dict) -> None:
    """The deep Kimi/Moonlight configuration block, shared verbatim."""
    from megatron.bridge.training.comm_overlap import CommOverlapConfig
    from megatron.bridge.training.mixed_precision import MixedPrecisionConfig

    config.model.virtual_pipeline_model_parallel_size = None
    config.model.context_parallel_size = 1
    config.model.pipeline_dtype = torch.bfloat16 if args.pp > 1 else None

    if args.pp > 1:
        if spec["key"] == "kimi_k25":
            from megatron.bridge.recipes.kimi.kimi_k2 import _get_kimi_k2_pipeline_layout

            config.model.pipeline_model_parallel_layout = _get_kimi_k2_pipeline_layout(args.pp, 1)
        else:
            from megatron.bridge.recipes.moonlight.moonlight_16b import _get_moonlight_pipeline_layout

            config.model.pipeline_model_parallel_layout = _get_moonlight_pipeline_layout(args.pp, 1)
    else:
        config.model.pipeline_model_parallel_layout = None

    config.model.account_for_embedding_in_pipeline_split = False
    config.model.account_for_loss_in_pipeline_split = False

    config.train.manual_gc = True
    config.train.manual_gc_interval = 5
    config.train.manual_gc_eval = 5

    config.model.moe_token_dispatcher_type = "alltoall"
    config.model.moe_flex_dispatcher_backend = "deepep"
    config.model.moe_hybridep_num_sms = 16
    config.model.moe_router_fusion = False
    config.model.moe_permute_fusion = True
    config.model.moe_grouped_gemm = True
    config.model.moe_shared_expert_overlap = True

    config.model.transformer_impl = "transformer_engine"
    config.model.cuda_graph_impl = "none"
    config.model.cuda_graph_scope = "full"
    config.model.cuda_graph_warmup_steps = 3

    config.model.attention_backend = None
    config.model.cross_entropy_loss_fusion = True
    config.model.cross_entropy_fusion_impl = "te"

    config.mixed_precision = MixedPrecisionConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=True,
    )
    config.model.moe_router_padding_for_fp8 = False

    config.optimizer.use_precision_aware_optimizer = False
    config.optimizer.main_grads_dtype = torch.float32
    config.optimizer.main_params_dtype = torch.float32
    config.optimizer.exp_avg_dtype = torch.float32
    config.optimizer.exp_avg_sq_dtype = torch.float32

    config.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    config.comm_overlap.delay_wgrad_compute = False
    config.comm_overlap.overlap_moe_expert_parallel_comm = False

    config.ddp.use_distributed_optimizer = False
    config.ddp.overlap_param_gather = False
    config.ddp.grad_reduce_in_fp32 = True
    config.ddp.overlap_grad_reduce = True
    config.ddp.check_for_nan_in_grad = True

    config.model.recompute_granularity = "full"
    config.model.recompute_method = "uniform"
    config.model.recompute_num_layers = 1


def _apply_quant_mode(config, args, spec: dict) -> None:
    from megatron.bridge.training.mixed_precision import get_mixed_precision_config

    if args.quant == "fp8":
        config.model.register_pre_wrap_hook(ensure_fp8_scale_inv_buffers)
        return

    # int4 and nvfp4 both build on meta and materialize during load.
    config.model.init_model_with_meta_device = True
    config.model.perform_initialization = False

    if args.quant == "int4":
        if spec["key"] == "qwen3_moe":
            # QOFT freezes the router, so the MoE aux loss cannot rebalance routing.
            config.model.moe_aux_loss_coeff = 0.0
            config.scheduler.lr_decay_iters = args.train_iters
        return

    # nvfp4: plain bf16 module structure + packed NVFP4 buffers at runtime
    # (orbit's low_precision_bootstrap mechanism). The load patches installed
    # by the entrypoint rewrite checkpoint requests into the converter's NVFP4
    # entry families, register packed expert buffers for the OFT grouped
    # kernels, and dequantize dense weights to bf16. ModelOpt is not used at
    # runtime: restoring its state would rebuild the expert modules into a
    # per-expert sharded layout whose keys do not exist in the checkpoint.
    config.model.gradient_accumulation_fusion = False
    if not spec["big_block"]:
        config.mixed_precision = get_mixed_precision_config("bf16_mixed")
    config.checkpoint.async_strategy = "mcore"
    if spec["key"] == "qwen3_moe":
        config.scheduler.lr_decay_iters = args.train_iters
        # Force the GroupedLinear MoE layout that convert_nvfp4_checkpoint_direct
        # produced; the recipe default (SequentialMLP) requests keys that do not
        # exist in the checkpoint.
        config.model.moe_grouped_gemm = True
        config.model.moe_token_dispatcher_type = "alltoall"
        config.model.moe_permute_fusion = True
        config.model.moe_router_fusion = False
        config.model.moe_shared_expert_overlap = True
        config.model.moe_aux_loss_coeff = 0.0


def build_config(args, spec: dict):
    """Assemble the full finetuning config for the detected architecture."""
    peft = build_oft(args)

    if spec["key"] == "qwen3_moe":
        from megatron.bridge import AutoBridge
        from megatron.bridge.recipes.qwen import qwen3_30b_a3b_peft_config

        config = qwen3_30b_a3b_peft_config(peft_scheme=peft)
        # Keep the recipe's training/dataset/logger/PEFT defaults but build the
        # model provider from the HF config so the runtime architecture matches
        # the checkpoint conversion path exactly, the same as kimi_k25/moonlight
        # below. The recipe otherwise hardcodes Qwen/Qwen3-30B-A3B, ignoring
        # --hf-model-path / --pretrained-checkpoint entirely.
        config.model = AutoBridge.from_hf_pretrained(args.hf_model_path, trust_remote_code=True).to_megatron_provider(
            load_weights=False
        )
    elif spec["key"] == "qwen3_dense":
        from megatron.bridge import AutoBridge
        from megatron.bridge.recipes.common import _peft_common
        from megatron.bridge.recipes.utils.finetune_utils import default_peft_config

        config = _peft_common()
        config.model = AutoBridge.from_hf_pretrained(
            args.hf_model_path,
            trust_remote_code=spec["trust_remote_code"],
        ).to_megatron_provider(load_weights=False)
        config.peft = default_peft_config(peft)
        config.tokenizer.tokenizer_model = args.hf_model_path
    elif spec["key"] == "kimi_k25":
        from megatron.bridge import AutoBridge
        from megatron.bridge.recipes.common import _peft_common
        from megatron.bridge.recipes.utils.finetune_utils import default_peft_config

        config = _peft_common()
        config.model = AutoBridge.from_hf_pretrained(args.hf_model_path, trust_remote_code=True).to_megatron_provider(
            load_weights=False
        )
        config.peft = default_peft_config(peft)
    else:  # moonlight
        from megatron.bridge import AutoBridge
        from megatron.bridge.recipes.moonlight.moonlight_16b import moonlight_16b_peft_config

        config = moonlight_16b_peft_config(peft)
        # Keep the recipe's training/dataset/logger defaults but build the model
        # provider from the HF config so the runtime architecture matches the
        # checkpoint conversion path exactly.
        config.model = AutoBridge.from_hf_pretrained(args.hf_model_path, trust_remote_code=True).to_megatron_provider(
            load_weights=False
        )
        patch_moonshot_build_tokenizer(config.model.vocab_size)

    _apply_common_model_overrides(config, args)
    if spec["big_block"]:
        _apply_big_model_block(config, args, spec)

    set_sequence_length(config, args.seq_length)

    if spec["tokenizer_from_ckpt"][args.quant]:
        config.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
        config.tokenizer.tokenizer_model = tokenizer_model_from_checkpoint(
            args.pretrained_checkpoint, args.hf_model_path
        )
        config.tokenizer.hf_tokenizer_kwargs = {"trust_remote_code": True}

    config.train.train_iters = args.train_iters
    config.train.global_batch_size = args.global_batch_size
    config.train.micro_batch_size = args.micro_batch_size
    config.scheduler.lr_warmup_iters = 2
    if spec["big_block"]:
        config.scheduler.lr_decay_iters = args.train_iters

    _apply_quant_mode(config, args, spec)
    normalize_hf_dataset_source(config)

    config.checkpoint.save_interval = args.save_interval if args.save_checkpoints else 0
    config.checkpoint.async_save = False

    if args.memory_smoke_test:
        args.profile_memory = True
        config.train.train_iters = 1
        disable_evaluation(config)
    if args.skip_eval:
        disable_evaluation(config)
    if args.skip_train:
        config.validation.skip_train = True
        disable_evaluation(config)

    slug = f"{spec['slug']}_qoft_{args.quant}"
    output_dir = args.output_dir or os.path.join(os.getcwd(), "nemo_experiments", slug)
    if args.save_checkpoints:
        config.checkpoint.save = os.path.join(output_dir, "checkpoints")
        config.checkpoint.load = os.path.join(output_dir, "checkpoints")
    else:
        config.checkpoint.save = None
        config.checkpoint.load = None
        config.logger.log_progress = False
        config.logger.wandb_save_dir = os.path.join(output_dir, "wandb")
    config.logger.tensorboard_dir = os.path.join(output_dir, "tb_logs")

    if args.log_interval is not None:
        config.logger.log_interval = args.log_interval
    if args.memory_smoke_test:
        config.logger.wandb_project = None
        config.logger.wandb_exp_name = None
    else:
        config.logger.wandb_project = "megatron-bridge-finetuning"
        config.logger.wandb_exp_name = f"{slug}_bs{args.block_size}_tp{args.tp}_ep{args.ep}"

    return config


def _debug_flag(name: str) -> bool:
    """Whether an opt-in debug environment variable is set."""
    return os.environ.get(name, "0").lower() in ("1", "true", "yes")


def main() -> None:
    """Detect the architecture, build the config, install patches, finetune."""
    args = parse_args()
    spec = resolve_arch(args.hf_model_path)
    if args.quant not in spec["quants"]:
        raise SystemExit(
            f"--quant {args.quant} is not supported for {spec['label']} (supported: {', '.join(spec['quants'])})"
        )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    _fill_arch_defaults(args, spec, world_size=world_size)
    _validate_runtime_topology(args, spec, world_size=world_size)

    if args.quant == "int4" and (
        args.int4_active_expert_chunk_size is not None or args.int4_grouped_chunk_backend is not None
    ):
        from megatron.bridge.orbit.oft.oft_layers import (
            set_int4_active_expert_chunk_size,
            set_int4_grouped_chunk_backend,
        )

        if args.int4_active_expert_chunk_size is not None:
            set_int4_active_expert_chunk_size(args.int4_active_expert_chunk_size)
        if args.int4_grouped_chunk_backend is not None:
            set_int4_grouped_chunk_backend(args.int4_grouped_chunk_backend)

    config = build_config(args, spec)

    if _debug_flag("QOFT_DEBUG_ZERO_LR"):
        # Pin every learning rate to zero so the optimizer cannot change a
        # single parameter. Any failure that still appears on a later step is
        # therefore independent of the update -- it comes from the data or from
        # reused device memory, not from a bad step.
        config.optimizer.lr = 0.0
        config.optimizer.min_lr = 0.0
        config.scheduler.lr_warmup_init = 0.0
        config.scheduler.lr_warmup_iters = 0
        print("[qoft-debug] learning rate pinned to 0: parameters are identical on every step")

    if _debug_flag("QOFT_DEBUG_ANOMALY"):
        # Names the forward operation whose backward produced the first
        # non-finite value, which module-level hooks cannot do.
        torch.autograd.set_detect_anomaly(True)
        print("[qoft-debug] autograd anomaly detection enabled (slow)")

    if args.quant == "int4":
        after_load_hook = None
        if args.profile_memory and args.skip_train:

            def after_load_hook(model_module):
                log_model_storage_summary([model_module], "int4")
                torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())

        install_int4_checkpoint_load_patches(
            scope=spec["int4_scope"],
            group_size=args.group_size,
            arch_label=spec["label"],
            validate_nonfinite=spec["validate_nonfinite"],
            after_load_hook=after_load_hook,
        )
    elif args.quant == "fp8":
        install_fp8_checkpoint_load_patches(
            pretrained_checkpoint=args.pretrained_checkpoint,
            arch_label=spec["label"],
        )
    elif args.quant == "nvfp4":
        install_nvfp4_checkpoint_load_patches(
            pretrained_checkpoint=args.pretrained_checkpoint,
            arch_label=spec["label"],
            validate_nonfinite=spec["validate_nonfinite"],
        )

    # Opt-in adapter diagnostics: QOFT_ADAPTER_INIT_CHECK=1 reports the OFT
    # rotation parameters at train start (see AdapterInitCallback).
    callbacks = []
    if os.environ.get("QOFT_ADAPTER_INIT_CHECK", "0").lower() in ("1", "true", "yes"):
        callbacks.append(AdapterInitCallback())
    if args.profile_memory and not args.skip_train:
        callbacks.append(MemoryProfileCallback(args.profile_memory_steps, args.quant))
    if args.debug_nan:
        callbacks.append(NanTraceCallback(args.debug_nan_steps))

    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.gpt_step import forward_step

    finetune(config=config, forward_step_func=forward_step, callbacks=callbacks or None)


if __name__ == "__main__":
    main()
