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

"""
Kimi-K2.5 QOFT finetuning — NVFP4 base weights + BF16 OFT adapters.

QOFT (Quantized OFT) keeps the base model expert weights in NVFP4 permanently
(analogous to QLoRA keeping weights in NF4).  Only the OFT rotation parameters
are in BF16 and receive gradient updates.

Architecture: 61 layers, 384 routed experts + 1 shared, hidden=7168, 64 heads,
Multi-Latent Attention (MLA), ~1T parameters.

Difference vs finetune_qoft_int4.py: NVFP4 base weights live in modelopt
quantizer modules (one weight + per-block FP8 scale + amax tensors) rather
than INT4 triplets (`*_packed`, `*_scale`, `*_shape`). NVFP4 conversion saves
ModelOpt quantization state alongside the checkpoint, so we just enable
`restore_modelopt_state=True` and the standard checkpoint loader rebuilds
the quantized layers — no INT4-style sharded-state-dict monkey-patching.

Prerequisites:
    Convert the HF NVFP4 checkpoint to Megatron format:

        python scripts/orbit/conversion/convert_nvfp4_checkpoint_direct.py \\
            --hf-model-path ${HF_MODEL_ROOT:-${HOME}/hf_models}/Kimi-K2.5-NVFP4 \\
            --megatron-path ./checkpoints/Kimi-K2.5-NVFP4

Usage:
    torchrun --nproc_per_node=8 scripts/orbit/models/kimi_k25/finetune_qoft_nvfp4.py \\
        --pretrained-checkpoint ./checkpoints/Kimi-K2.5-NVFP4 \\
        --tp 2 --ep 4
"""

# ruff: noqa: D101, D103  # operational scripts: helpers here are entrypoint plumbing, not API

import argparse
import os
import re
from collections import defaultdict

import torch
import torch.distributed as dist

from megatron.bridge import AutoBridge
from megatron.bridge.orbit.oft.oft import OFT
from megatron.bridge.recipes.common import _peft_common
from megatron.bridge.recipes.kimi.kimi_k2 import _get_kimi_k2_pipeline_layout
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config
from megatron.bridge.training.callbacks import Callback
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.mixed_precision import get_mixed_precision_config
from megatron.bridge.utils.common_utils import print_rank_0


KIMI_K25_ALL_LINEAR_OFT_TARGET_MODULES = [
    "linear_q_down_proj",
    "linear_q_up_proj",
    "linear_kv_down_proj",
    "linear_kv_up_proj",
    "linear_proj",
    "linear_fc1",
    "linear_fc2",
]


def _parse_target_modules(value: str) -> list[str]:
    return [module for module in re.split(r"[\s,]+", value.strip()) if module]


def _format_nbytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _tensor_storage_nbytes(tensor: torch.Tensor) -> int:
    if tensor is None or tensor.device.type == "meta":
        return 0
    try:
        return tensor.untyped_storage().nbytes()
    except RuntimeError:
        return 0


def _add_tensor_storage(summary: dict[str, int], seen_storages: set, bucket: str, tensor: torch.Tensor) -> None:
    nbytes = _tensor_storage_nbytes(tensor)
    if nbytes == 0:
        return

    try:
        storage = tensor.untyped_storage()
        key = (tensor.device.type, tensor.device.index, storage.data_ptr(), nbytes)
    except RuntimeError:
        return

    if key in seen_storages:
        return
    seen_storages.add(key)
    summary[bucket] += nbytes
    summary["total_unique_storage_bytes"] += nbytes


def _classify_nvfp4_buffer(name: str) -> str:
    """Bucket modelopt NVFP4 quantizer buffers by role for memory accounting."""
    if name.endswith(".weight_quantizer._scale") or "._weight_quantizer._scale" in name:
        return "nvfp4_block_scale_storage_bytes"
    if name.endswith(".weight_quantizer._amax") or "._weight_quantizer._amax" in name:
        return "nvfp4_weight_amax_storage_bytes"
    if name.endswith(".input_quantizer._amax") or "._input_quantizer._amax" in name:
        return "nvfp4_input_amax_storage_bytes"
    if name.endswith(".weight_quantizer._double_scale"):
        return "nvfp4_double_scale_storage_bytes"
    return "other_buffer_storage_bytes"


def _summarize_model_storage(model_chunks) -> dict[str, int]:
    summary = defaultdict(int)
    seen_storages = set()

    for model_chunk in model_chunks:
        for _, param in model_chunk.named_parameters():
            if param is None:
                continue

            summary["parameter_tensors"] += 1
            summary["parameter_numel"] += param.numel()
            if param.requires_grad:
                summary["trainable_parameter_numel"] += param.numel()
            else:
                summary["frozen_parameter_numel"] += param.numel()

            if _tensor_storage_nbytes(param) == 0:
                summary["empty_parameter_tensors"] += 1
                continue

            bucket = "trainable_parameter_storage_bytes" if param.requires_grad else "frozen_parameter_storage_bytes"
            _add_tensor_storage(summary, seen_storages, bucket, param)

        for name, buffer in model_chunk.named_buffers():
            if buffer is None:
                continue

            summary["buffer_tensors"] += 1
            summary["buffer_numel"] += buffer.numel()

            bucket = _classify_nvfp4_buffer(name)
            _add_tensor_storage(summary, seen_storages, bucket, buffer)

    return dict(summary)


def _collect_parameter_entries(model_chunks) -> tuple[list[dict], list[dict]]:
    frozen_entries = []
    trainable_entries = []

    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if param is None:
                continue

            nbytes = _tensor_storage_nbytes(param)
            if nbytes == 0:
                continue

            entry = {
                "name": name,
                "nbytes": nbytes,
                "shape": tuple(param.shape),
                "dtype": str(param.dtype).replace("torch.", ""),
            }
            if param.requires_grad:
                trainable_entries.append(entry)
            else:
                frozen_entries.append(entry)

    frozen_entries.sort(key=lambda item: item["nbytes"], reverse=True)
    trainable_entries.sort(key=lambda item: item["nbytes"], reverse=True)
    return frozen_entries, trainable_entries


def _collect_buffer_entries(model_chunks) -> tuple[list[dict], list[dict]]:
    nvfp4_entries = []
    other_entries = []

    for model_chunk in model_chunks:
        for name, buffer in model_chunk.named_buffers():
            if buffer is None:
                continue

            nbytes = _tensor_storage_nbytes(buffer)
            if nbytes == 0:
                continue

            bucket = _classify_nvfp4_buffer(name)
            entry = {
                "name": name,
                "nbytes": nbytes,
                "shape": tuple(buffer.shape),
                "dtype": str(buffer.dtype).replace("torch.", ""),
                "kind": bucket.replace("_storage_bytes", "").replace("nvfp4_", ""),
            }

            if bucket.startswith("nvfp4_"):
                nvfp4_entries.append(entry)
            else:
                other_entries.append(entry)

    nvfp4_entries.sort(key=lambda item: item["nbytes"], reverse=True)
    other_entries.sort(key=lambda item: item["nbytes"], reverse=True)
    return nvfp4_entries, other_entries


def _log_parameter_table(title: str, entries: list[dict], limit: int = 12) -> None:
    if not entries:
        print_rank_0(f"[memory:{title}] none")
        return

    print_rank_0(f"[memory:{title}] top_{min(limit, len(entries))}")
    for idx, entry in enumerate(entries[:limit], start=1):
        print_rank_0(
            "[memory:{title}] {idx:02d} {size:>10}  {dtype:<8}  {shape}  {name}".format(
                title=title,
                idx=idx,
                size=_format_nbytes(entry["nbytes"]),
                dtype=entry["dtype"],
                shape=entry["shape"],
                name=entry["name"],
            )
        )


def _log_buffer_table(title: str, entries: list[dict], limit: int = 12) -> None:
    if not entries:
        print_rank_0(f"[memory:{title}] none")
        return

    print_rank_0(f"[memory:{title}] top_{min(limit, len(entries))}")
    for idx, entry in enumerate(entries[:limit], start=1):
        print_rank_0(
            "[memory:{title}] {idx:02d} {size:>10}  {dtype:<8}  {kind:<18}  {shape}  {name}".format(
                title=title,
                idx=idx,
                size=_format_nbytes(entry["nbytes"]),
                dtype=entry["dtype"],
                kind=entry["kind"],
                shape=entry["shape"],
                name=entry["name"],
            )
        )


def _cuda_memory_snapshot() -> dict[str, int] | None:
    if not torch.cuda.is_available():
        return None

    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.synchronize(device)
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _log_cuda_memory(tag: str) -> None:
    snapshot = _cuda_memory_snapshot()
    if snapshot is None:
        print_rank_0(f"[memory:{tag}] CUDA unavailable")
        return

    device = torch.device("cuda", torch.cuda.current_device())
    values = torch.tensor(
        [
            snapshot["allocated_bytes"],
            snapshot["reserved_bytes"],
            snapshot["max_allocated_bytes"],
            snapshot["max_reserved_bytes"],
        ],
        device=device,
        dtype=torch.int64,
    )
    min_values = values.clone()
    max_values = values.clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(min_values, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_values, op=dist.ReduceOp.MAX)

    labels = [
        "allocated",
        "reserved",
        "peak_allocated",
        "peak_reserved",
    ]
    parts = []
    for idx, label in enumerate(labels):
        min_val = int(min_values[idx].item())
        max_val = int(max_values[idx].item())
        if min_val == max_val:
            parts.append(f"{label}={_format_nbytes(max_val)}")
        else:
            parts.append(f"{label}=min {_format_nbytes(min_val)}, max {_format_nbytes(max_val)}")

    print_rank_0(f"[memory:{tag}] " + ", ".join(parts))


def _log_model_storage_summary(model_chunks) -> None:
    summary = _summarize_model_storage(model_chunks)
    frozen_entries, trainable_entries = _collect_parameter_entries(model_chunks)
    nvfp4_buffer_entries, other_buffer_entries = _collect_buffer_entries(model_chunks)
    lines = [
        "[memory:model-storage] unique_storage={}".format(
            _format_nbytes(summary.get("total_unique_storage_bytes", 0))
        ),
        "[memory:model-storage] frozen_params={} trainable_params={} empty_param_tensors={}".format(
            _format_nbytes(summary.get("frozen_parameter_storage_bytes", 0)),
            _format_nbytes(summary.get("trainable_parameter_storage_bytes", 0)),
            summary.get("empty_parameter_tensors", 0),
        ),
        "[memory:model-storage] nvfp4_block_scale={} nvfp4_double_scale={} nvfp4_weight_amax={} nvfp4_input_amax={} other_buffers={}".format(
            _format_nbytes(summary.get("nvfp4_block_scale_storage_bytes", 0)),
            _format_nbytes(summary.get("nvfp4_double_scale_storage_bytes", 0)),
            _format_nbytes(summary.get("nvfp4_weight_amax_storage_bytes", 0)),
            _format_nbytes(summary.get("nvfp4_input_amax_storage_bytes", 0)),
            _format_nbytes(summary.get("other_buffer_storage_bytes", 0)),
        ),
        "[memory:model-storage] parameter_numel={} trainable_parameter_numel={} buffer_numel={}".format(
            summary.get("parameter_numel", 0),
            summary.get("trainable_parameter_numel", 0),
            summary.get("buffer_numel", 0),
        ),
    ]
    for line in lines:
        print_rank_0(line)
    _log_parameter_table("frozen-params", frozen_entries)
    _log_parameter_table("trainable-params", trainable_entries)
    _log_buffer_table("nvfp4-buffers", nvfp4_buffer_entries)
    _log_buffer_table("other-buffers", other_buffer_entries)


class MemoryProfileCallback(Callback):
    def __init__(self, profile_steps: int):
        self.profile_steps = max(1, profile_steps)

    def on_train_start(self, context) -> None:
        if not torch.cuda.is_available():
            return
        _log_model_storage_summary(context.model)
        torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
        _log_cuda_memory("after-load-before-step0")

    def on_train_step_start(self, context) -> None:
        if not torch.cuda.is_available():
            return
        step = int(context.state.train_state.step)
        if step < self.profile_steps:
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
            _log_cuda_memory(f"step{step}-start")

    def on_train_step_end(self, context) -> None:
        if not torch.cuda.is_available():
            return
        step = int(context.state.train_state.step)
        if step < self.profile_steps:
            _log_cuda_memory(f"step{step}-end")


def _set_sequence_length(config, seq_length: int) -> None:
    config.model.seq_length = seq_length
    if getattr(config, "dataset", None) is not None:
        config.dataset.seq_length = seq_length
        packed_sequence_specs = getattr(config.dataset, "packed_sequence_specs", None)
        if packed_sequence_specs is not None:
            packed_sequence_specs.packed_sequence_size = seq_length


def _disable_evaluation(config) -> None:
    config.validation.eval_iters = 0
    config.validation.eval_interval = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kimi-K2.5 QOFT finetuning (NVFP4 base weights)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        required=True,
        help="Path to NVFP4 Megatron checkpoint (converted via convert_nvfp4_checkpoint_direct.py)",
    )
    parser.add_argument(
        "--hf-model-path", type=str, default="moonshotai/Kimi-K2.5", help="HF model path for config/tokenizer"
    )
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
        default=list(KIMI_K25_ALL_LINEAR_OFT_TARGET_MODULES),
        help=(
            "Comma or whitespace separated Megatron module names to wrap with OFT. "
            "Defaults to all Kimi MLA attention projections plus MLP/expert linears."
        ),
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        default=False,
        help="Enable saving/loading run checkpoints under output-dir/checkpoints.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=500,
        help="Checkpoint save interval when --save-checkpoints is enabled.",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        default=False,
        help="Print model storage and CUDA allocator memory around training steps.",
    )
    parser.add_argument(
        "--profile-memory-steps",
        type=int,
        default=1,
        help="Number of initial training steps to profile when --profile-memory is enabled.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        default=False,
        help="Load the checkpoint, restore NVFP4 modelopt state, initialize OFT, then exit without training.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        default=False,
        help="Disable validation/evaluation after training. Useful for long-sequence memory smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # The grouped-MoE-safe ModelOpt compress patch lives in
    # `bridge.training.post_training.checkpointing._patch_modelopt_pack_for_grouped_moe`
    # and is applied automatically inside `_maybe_compress_restored_modelopt_model`
    # whenever a packed-weight checkpoint is restored — no per-script patching needed.

    # OFT adapter config — rotation parameters are BF16, base weights stay NVFP4
    oft = OFT(
        target_modules=args.target_modules,
        block_size=args.block_size,
        coft=args.coft,
        eps=args.eps,
        block_share=args.block_share,
        module_dropout=args.module_dropout,
    )

    # Start from PEFT common config
    config = _peft_common()
    if args.distributed_timeout_minutes is not None:
        config.dist.distributed_timeout_minutes = args.distributed_timeout_minutes

    # Model config — Kimi K2.5 (LLM backbone, not VL)
    # init_model_with_meta_device=True: build model on meta device (zero memory),
    # then materialize during checkpoint load. This avoids the ~2TB BF16 allocation
    # that would OOM on both GPU and CPU for a 1T-param model.
    config.model = AutoBridge.from_hf_pretrained(
        args.hf_model_path,
        trust_remote_code=True,
    ).to_megatron_provider(load_weights=False)
    config.model.init_model_with_meta_device = True
    config.model.perform_initialization = False

    # NVFP4 path: rebuild modelopt-quantized layers from the saved modelopt state
    # in the converted checkpoint. This replaces the INT4 monkey-patching path
    # (sharded-state-dict transform + register_int4_buffers_after_load); the
    # standard checkpoint loader handles NVFP4 buffers via modelopt restore.
    config.model.restore_modelopt_state = True
    # Required when restore_modelopt_state=True — fused gradient accumulation
    # is incompatible with ModelOpt-quantized layers (validator enforces this
    # in training/config.py:validate). Same setting as qwen3_14b NVFP4.
    config.model.gradient_accumulation_fusion = False

    # PEFT
    config.peft = default_peft_config(oft)

    # Checkpoint
    config.checkpoint.pretrained_checkpoint = args.pretrained_checkpoint

    # Parallelism
    config.model.tensor_model_parallel_size = args.tp
    config.model.expert_model_parallel_size = args.ep
    config.model.pipeline_model_parallel_size = args.pp
    config.model.sequence_parallel = args.sp
    config.model.expert_tensor_parallel_size = 1
    config.model.virtual_pipeline_model_parallel_size = None
    config.model.context_parallel_size = 1
    config.model.pipeline_dtype = torch.bfloat16 if args.pp > 1 else None

    # Pipeline layout for PP > 1
    if args.pp > 1:
        config.model.pipeline_model_parallel_layout = _get_kimi_k2_pipeline_layout(args.pp, 1)
    else:
        config.model.pipeline_model_parallel_layout = None

    # Pipeline split settings
    config.model.account_for_embedding_in_pipeline_split = False
    config.model.account_for_loss_in_pipeline_split = False

    # Sequence length
    _set_sequence_length(config, args.seq_length)

    # Tokenizer
    # This is an LLM finetune path built on `_peft_common()`, so we need a real
    # text tokenizer for dataset preprocessing. Prefer the tokenizer assets saved
    # into the converted Megatron checkpoint, and fall back to the HF model path.
    checkpoint_tokenizer_dir = os.path.join(args.pretrained_checkpoint, "iter_{:07d}".format(0), "tokenizer")
    tokenizer_model = checkpoint_tokenizer_dir if os.path.isdir(checkpoint_tokenizer_dir) else args.hf_model_path
    config.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    config.tokenizer.tokenizer_model = tokenizer_model
    config.tokenizer.hf_tokenizer_kwargs = {"trust_remote_code": True}

    # Training
    config.train.train_iters = args.train_iters
    config.train.global_batch_size = args.global_batch_size
    config.train.micro_batch_size = args.micro_batch_size
    config.train.manual_gc = True
    config.train.manual_gc_interval = 5
    config.train.manual_gc_eval = 5

    # Scheduler
    config.scheduler.lr_warmup_iters = 2
    config.scheduler.lr_decay_iters = args.train_iters

    # MoE settings
    config.model.moe_token_dispatcher_type = "alltoall"
    config.model.moe_flex_dispatcher_backend = "deepep"
    config.model.moe_hybridep_num_sms = 16
    config.model.moe_router_fusion = False
    config.model.moe_permute_fusion = True
    config.model.moe_grouped_gemm = True
    config.model.moe_shared_expert_overlap = True

    # TE
    config.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    config.model.cuda_graph_impl = "none"
    config.model.cuda_graph_scope = "full"
    config.model.cuda_graph_warmup_steps = 3

    # Kernels
    config.model.attention_backend = None
    config.model.cross_entropy_loss_fusion = True
    config.model.cross_entropy_fusion_impl = "te"

    # Mixed precision — BF16 activations + grads, NVFP4 base weights, BF16
    # OFT params. The recipe registers FP4 quantizer state alongside the
    # standard BF16 mixed-precision config.
    config.mixed_precision = get_mixed_precision_config("bf16_with_nvfp4_mixed")
    config.model.moe_router_padding_for_fp8 = False

    # Optimizer precision
    config.optimizer.use_precision_aware_optimizer = False
    config.optimizer.main_grads_dtype = torch.float32
    config.optimizer.main_params_dtype = torch.float32
    config.optimizer.exp_avg_dtype = torch.float32
    config.optimizer.exp_avg_sq_dtype = torch.float32

    # Communication
    config.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    config.comm_overlap.delay_wgrad_compute = False
    config.comm_overlap.overlap_moe_expert_parallel_comm = False

    # DDP — PEFT typically doesn't need distributed optimizer
    config.ddp.use_distributed_optimizer = False
    config.ddp.overlap_param_gather = False
    config.ddp.grad_reduce_in_fp32 = True
    config.ddp.overlap_grad_reduce = True
    config.ddp.check_for_nan_in_grad = True

    # Memory
    config.model.recompute_granularity = "full"
    config.model.recompute_method = "uniform"
    config.model.recompute_num_layers = 1

    # Checkpoint save
    config.checkpoint.save_interval = args.save_interval if args.save_checkpoints else 0
    config.checkpoint.async_save = False
    # Direct NVFP4 conversion writes via mcore async strategy; mirror that on save.
    config.checkpoint.async_strategy = "mcore"

    # Optional load-only smoke mode: exercise setup/checkpoint load and then exit.
    if args.skip_eval:
        _disable_evaluation(config)

    if args.skip_train:
        config.validation.skip_train = True
        _disable_evaluation(config)

    # Output
    output_dir = args.output_dir or os.path.join(os.getcwd(), "nemo_experiments", "kimi_k25_qoft_nvfp4")
    if args.save_checkpoints:
        config.checkpoint.save = os.path.join(output_dir, "checkpoints")
        config.checkpoint.load = os.path.join(output_dir, "checkpoints")
    else:
        config.checkpoint.save = None
        config.checkpoint.load = None
        config.logger.log_progress = False
        config.logger.wandb_save_dir = os.path.join(output_dir, "wandb")
    config.logger.tensorboard_dir = os.path.join(output_dir, "tb_logs")

    # Logger
    config.logger.log_interval = 1
    config.logger.wandb_project = "megatron-bridge-finetuning"
    config.logger.wandb_exp_name = f"kimi_k25_qoft_nvfp4_bs{args.block_size}_tp{args.tp}_ep{args.ep}"

    callbacks = []
    if args.profile_memory and not args.skip_train:
        callbacks.append(MemoryProfileCallback(args.profile_memory_steps))

    finetune(config=config, forward_step_func=forward_step, callbacks=callbacks or None)


if __name__ == "__main__":
    main()
