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
Moonlight-16B-A3B QOFT finetuning - INT4 base weights + BF16 OFT adapters.

Expected flow:
    1. Quantize BF16 HF experts to Kimi-style INT4 triplets:
         python examples/conversion/quantize_to_int4.py --input ... --output ...
    2. Convert HF INT4 checkpoint to Megatron INT4:
         bash convert_int4_checkpoint_direct.sh /path/to/Moonlight-16B-A3B-INT4 ...
    3. Finetune with this script:
         torchrun --nproc_per_node=2 examples/models/moonlight_16b/finetune_qoft_int4.py \
             --pretrained-checkpoint ./checkpoints/Moonlight-16B-A3B-INT4 \
             --tp 1 --ep 2
"""

import argparse
from collections import defaultdict
import logging
import os
import re
from typing import Any, Iterable

import torch
import torch.distributed as dist

from megatron.bridge import AutoBridge
from megatron.bridge.models.common.unimodal import to_empty_if_meta_device
from megatron.bridge.peft.int4_utils import (
    register_int4_buffers_after_load,
    transform_sharded_state_dict_for_int4,
)
from megatron.bridge.peft.oft import OFT
from megatron.bridge.peft.oft_layers import (
    OFTLinear,
    OFTRotationModule,
    OFTTopKRouter,
    set_int4_active_expert_chunk_size,
    set_int4_grouped_chunk_backend,
)
from megatron.bridge.recipes.moonlight.moonlight_16b import (
    _get_moonlight_pipeline_layout,
    moonlight_16b_peft_config,
)
from megatron.bridge.training.callbacks import Callback
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig
from megatron.bridge.utils.common_utils import print_rank_0


logger = logging.getLogger(__name__)


class _TokenizerLenProxy:
    """Delegate tokenizer operations while forcing a smaller reported length."""

    def __init__(self, tokenizer, forced_len: int):
        self._tokenizer = tokenizer
        self._forced_len = forced_len

    def __len__(self):
        return self._forced_len

    def __call__(self, *args, **kwargs):
        return self._tokenizer(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


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

            if name.endswith("_packed"):
                bucket = "int4_packed_storage_bytes"
            elif name.endswith("_scale"):
                bucket = "int4_scale_storage_bytes"
            elif name.endswith("_shape"):
                bucket = "int4_shape_storage_bytes"
            else:
                bucket = "other_buffer_storage_bytes"

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
    int4_entries = []
    other_entries = []

    for model_chunk in model_chunks:
        for name, buffer in model_chunk.named_buffers():
            if buffer is None:
                continue

            nbytes = _tensor_storage_nbytes(buffer)
            if nbytes == 0:
                continue

            if name.endswith("_packed"):
                buffer_kind = "packed"
            elif name.endswith("_scale"):
                buffer_kind = "scale"
            elif name.endswith("_shape"):
                buffer_kind = "shape"
            else:
                buffer_kind = "other"

            entry = {
                "name": name,
                "nbytes": nbytes,
                "shape": tuple(buffer.shape),
                "dtype": str(buffer.dtype).replace("torch.", ""),
                "kind": buffer_kind,
            }

            if buffer_kind == "other":
                other_entries.append(entry)
            else:
                int4_entries.append(entry)

    int4_entries.sort(key=lambda item: item["nbytes"], reverse=True)
    other_entries.sort(key=lambda item: item["nbytes"], reverse=True)
    return int4_entries, other_entries


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
            "[memory:{title}] {idx:02d} {size:>10}  {dtype:<8}  {kind:<6}  {shape}  {name}".format(
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
    int4_buffer_entries, other_buffer_entries = _collect_buffer_entries(model_chunks)
    lines = [
        "[memory:model-storage] unique_storage={}".format(
            _format_nbytes(summary.get("total_unique_storage_bytes", 0))
        ),
        "[memory:model-storage] frozen_params={} trainable_params={} empty_param_tensors={}".format(
            _format_nbytes(summary.get("frozen_parameter_storage_bytes", 0)),
            _format_nbytes(summary.get("trainable_parameter_storage_bytes", 0)),
            summary.get("empty_parameter_tensors", 0),
        ),
        "[memory:model-storage] int4_packed={} int4_scale={} int4_shape={} other_buffers={}".format(
            _format_nbytes(summary.get("int4_packed_storage_bytes", 0)),
            _format_nbytes(summary.get("int4_scale_storage_bytes", 0)),
            _format_nbytes(summary.get("int4_shape_storage_bytes", 0)),
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
    _log_buffer_table("int4-buffers", int4_buffer_entries)
    _log_buffer_table("other-buffers", other_buffer_entries)


def _iter_tensor_paths(obj: Any, prefix: str = "value") -> Iterable[tuple[str, torch.Tensor]]:
    if isinstance(obj, torch.Tensor):
        yield prefix, obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from _iter_tensor_paths(value, f"{prefix}.{key}")
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            yield from _iter_tensor_paths(value, f"{prefix}[{idx}]")


def _tensor_nonfinite_stats(tensor: torch.Tensor) -> dict[str, Any] | None:
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return None
    if tensor.device.type == "meta":
        return None
    if torch.isfinite(tensor).all():
        return None

    nan_count = int(torch.isnan(tensor).sum().item())
    inf_count = int(torch.isinf(tensor).sum().item())
    finite_mask = torch.isfinite(tensor)
    finite_abs_max = 0.0
    if finite_mask.any():
        finite_abs_max = float(tensor[finite_mask].abs().max().item())

    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "finite_abs_max": finite_abs_max,
    }


def _describe_nonfinite_tensors(obj: Any, prefix: str) -> dict[str, Any] | None:
    for path, tensor in _iter_tensor_paths(obj, prefix):
        stats = _tensor_nonfinite_stats(tensor)
        if stats is not None:
            stats["path"] = path
            return stats
    return None


def _first_nonfinite_named_parameter(model_chunks, include_grad: bool = False) -> dict[str, Any] | None:
    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if param is None:
                continue

            tensor = param.grad if include_grad else param
            if tensor is None:
                continue

            stats = _tensor_nonfinite_stats(tensor)
            if stats is None:
                continue

            stats["name"] = name
            stats["kind"] = "grad" if include_grad else "param"
            return stats
    return None


def _current_learning_rate(optimizer) -> float | None:
    if optimizer is None:
        return None
    for group in getattr(optimizer, "param_groups", []):
        if len(group) == 0:
            continue
        if not group.get("is_decoupled_lr", False):
            return float(group["lr"])
    return None


_EXPECTED_INT4_MISSING_KEY_RE = re.compile(
    r".*\.experts\.linear_fc[12]\.(?:weight\d+|weight|weight\d+_(?:packed|scale|shape))$"
)


def _is_expected_int4_missing_key(key: str) -> bool:
    return key.endswith("._extra_state") or _EXPECTED_INT4_MISSING_KEY_RE.fullmatch(key) is not None


def _validate_loaded_model_tensors(model_module: torch.nn.Module) -> None:
    bad_entries = []

    for name, param in model_module.named_parameters():
        if param is None or param.device.type == "meta":
            continue
        stats = _tensor_nonfinite_stats(param)
        if stats is not None:
            bad_entries.append(("param", name, stats))
            break

    if not bad_entries:
        for name, buffer in model_module.named_buffers():
            if buffer is None or buffer.device.type == "meta":
                continue
            stats = _tensor_nonfinite_stats(buffer)
            if stats is not None:
                bad_entries.append(("buffer", name, stats))
                break

    if bad_entries:
        kind, name, stats = bad_entries[0]
        raise RuntimeError(
            "Loaded model contains non-finite tensor before training: "
            f"{kind}={name}, shape={stats['shape']}, dtype={stats['dtype']}, "
            f"device={stats['device']}, nan={stats['nan_count']}, inf={stats['inf_count']}, "
            f"finite_abs_max={stats['finite_abs_max']:.6g}"
        )


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


class NanTraceCallback(Callback):
    def __init__(self, trace_steps: int):
        self.trace_steps = max(1, trace_steps)
        self._current_step: int | None = None
        self._forward_hit = False
        self._backward_hit = False
        self._hook_handles = []

    def _should_trace(self, step: int) -> bool:
        return step < self.trace_steps

    def _should_hook_module(self, module: torch.nn.Module) -> bool:
        if isinstance(module, (OFTLinear, OFTRotationModule, OFTTopKRouter)):
            return True
        if any(True for _ in module.children()):
            return False
        has_state = any(True for _ in module.parameters(recurse=False)) or any(
            True for _ in module.buffers(recurse=False)
        )
        return has_state

    def _format_issue(self, issue: dict[str, Any]) -> str:
        return (
            f"path={issue.get('path', '-')}, shape={issue['shape']}, dtype={issue['dtype']}, "
            f"device={issue['device']}, nan={issue['nan_count']}, inf={issue['inf_count']}, "
            f"finite_abs_max={issue['finite_abs_max']:.6g}"
        )

    def _make_forward_hook(self, name: str):
        def _hook(module, inputs, output):
            if self._current_step is None or self._forward_hit or not self._should_trace(self._current_step):
                return

            issue = _describe_nonfinite_tensors(inputs, "input")
            source = "input"
            if issue is None:
                issue = _describe_nonfinite_tensors(output, "output")
                source = "output"
            if issue is None:
                return

            self._forward_hit = True
            print_rank_0(
                f"[nan-debug] first forward non-finite at step {self._current_step}: "
                f"module={name} type={type(module).__name__} source={source} {self._format_issue(issue)}"
            )

        return _hook

    def _make_backward_hook(self, name: str):
        def _hook(module, grad_input, grad_output):
            if self._current_step is None or self._backward_hit or not self._should_trace(self._current_step):
                return

            issue = _describe_nonfinite_tensors(grad_output, "grad_output")
            source = "grad_output"
            if issue is None:
                issue = _describe_nonfinite_tensors(grad_input, "grad_input")
                source = "grad_input"
            if issue is None:
                return

            self._backward_hit = True
            print_rank_0(
                f"[nan-debug] first backward non-finite at step {self._current_step}: "
                f"module={name} type={type(module).__name__} source={source} {self._format_issue(issue)}"
            )

        return _hook

    def on_train_start(self, context) -> None:
        hooked_modules = 0
        for model_chunk in context.model:
            for name, module in model_chunk.named_modules():
                if not name or not self._should_hook_module(module):
                    continue
                self._hook_handles.append(module.register_forward_hook(self._make_forward_hook(name)))
                try:
                    self._hook_handles.append(module.register_full_backward_hook(self._make_backward_hook(name)))
                except RuntimeError:
                    pass
                hooked_modules += 1

        print_rank_0(f"[nan-debug] registered hooks on {hooked_modules} modules")

    def on_train_step_start(self, context) -> None:
        step = int(context.state.train_state.step)
        self._current_step = step
        self._forward_hit = False
        self._backward_hit = False

        if not self._should_trace(step):
            return

        lr = _current_learning_rate(context.optimizer)
        lr_text = f"{lr:.6g}" if lr is not None else "n/a"
        print_rank_0(f"[nan-debug] step {step} start lr={lr_text}")

        param_issue = _first_nonfinite_named_parameter(context.model, include_grad=False)
        if param_issue is not None:
            print_rank_0(
                f"[nan-debug] non-finite trainable/base parameter before forward at step {step}: "
                f"name={param_issue['name']} kind={param_issue['kind']} shape={param_issue['shape']} "
                f"dtype={param_issue['dtype']} device={param_issue['device']} "
                f"nan={param_issue['nan_count']} inf={param_issue['inf_count']} "
                f"finite_abs_max={param_issue['finite_abs_max']:.6g}"
            )

    def on_train_step_end(self, context) -> None:
        step = int(context.state.train_state.step)
        if not self._should_trace(step):
            return

        lr = _current_learning_rate(context.optimizer)
        lr_text = f"{lr:.6g}" if lr is not None else "n/a"

        loss_text = "n/a"
        if context.loss_dict and "lm loss" in context.loss_dict:
            loss_val = context.loss_dict["lm loss"]
            if isinstance(loss_val, torch.Tensor):
                loss_text = f"{float(loss_val.item()):.6g}"
            else:
                loss_text = str(loss_val)

        grad_norm_text = "n/a" if context.grad_norm is None else f"{float(context.grad_norm):.6g}"
        print_rank_0(
            f"[nan-debug] step {step} end loss={loss_text} grad_norm={grad_norm_text} "
            f"skipped={context.skipped_iter} lr={lr_text}"
        )

        param_issue = _first_nonfinite_named_parameter(context.model, include_grad=False)
        if param_issue is not None:
            print_rank_0(
                f"[nan-debug] non-finite parameter after optimizer step at step {step}: "
                f"name={param_issue['name']} kind={param_issue['kind']} shape={param_issue['shape']} "
                f"dtype={param_issue['dtype']} device={param_issue['device']} "
                f"nan={param_issue['nan_count']} inf={param_issue['inf_count']} "
                f"finite_abs_max={param_issue['finite_abs_max']:.6g}"
            )

        grad_issue = _first_nonfinite_named_parameter(context.model, include_grad=True)
        if grad_issue is not None:
            print_rank_0(
                f"[nan-debug] non-finite gradient observed at step {step}: "
                f"name={grad_issue['name']} kind={grad_issue['kind']} shape={grad_issue['shape']} "
                f"dtype={grad_issue['dtype']} device={grad_issue['device']} "
                f"nan={grad_issue['nan_count']} inf={grad_issue['inf_count']} "
                f"finite_abs_max={grad_issue['finite_abs_max']:.6g}"
            )

        if not self._forward_hit:
            print_rank_0(f"[nan-debug] no non-finite activation seen in traced forward modules at step {step}")
        if not self._backward_hit:
            print_rank_0(f"[nan-debug] no non-finite gradient seen in traced backward modules at step {step}")

    def on_train_end(self, context) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()


def _is_moonshot_tokenizer(tokenizer) -> bool:
    if tokenizer is None:
        return False

    name_or_path = str(getattr(tokenizer, "name_or_path", "") or "").lower()
    if "moonlight" in name_or_path or "moonshot" in name_or_path:
        return True

    tokenizer_type = type(tokenizer).__name__.lower()
    tokenizer_module = type(tokenizer).__module__.lower()
    return "moonshot" in tokenizer_module or tokenizer_type == "tiktokentokenizer"


def _patch_moonlight_build_tokenizer(model_vocab_size: int) -> None:
    """Clamp Moonshot HF tokenizer length to the model vocab during training setup.

    Moonlight's HF tokenizer reports two extra added tokens via ``len(tokenizer)``,
    but the model config and converted checkpoint use ``vocab_size=163840``.
    Megatron validates against ``tokenizer.vocab_size`` during setup, so patch the
    tokenizer construction path itself to keep both sides aligned.
    """
    import megatron.bridge.training.model_load_save as _model_load_save_mod
    import megatron.bridge.training.setup as _setup_mod
    import megatron.bridge.training.state as _state_mod
    import megatron.bridge.training.tokenizers.tokenizer as _tok_mod

    if getattr(_tok_mod, "_moonlight_vocab_patch_applied", False):
        return

    original_build_tokenizer = _tok_mod.build_tokenizer

    def _patched_build_tokenizer(config, **kwargs):
        tokenizer = original_build_tokenizer(config, **kwargs)
        outer_tokenizer = getattr(tokenizer, "_tokenizer", None)
        hf_tokenizer = getattr(outer_tokenizer, "tokenizer", None)

        if _is_moonshot_tokenizer(hf_tokenizer) and tokenizer.vocab_size > model_vocab_size:
            actual_vocab_size = tokenizer.vocab_size
            outer_tokenizer.tokenizer = _TokenizerLenProxy(hf_tokenizer, model_vocab_size)
            outer_tokenizer.original_vocab_size = model_vocab_size
            logger.warning(
                "Clamping Moonshot tokenizer vocab from %s to model vocab_size %s for Megatron setup",
                actual_vocab_size,
                model_vocab_size,
            )
        return tokenizer

    _tok_mod.build_tokenizer = _patched_build_tokenizer
    _setup_mod.build_tokenizer = _patched_build_tokenizer
    _state_mod.build_tokenizer = _patched_build_tokenizer
    _model_load_save_mod.build_tokenizer = _patched_build_tokenizer
    _tok_mod._moonlight_vocab_patch_applied = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Moonlight-16B-A3B QOFT finetuning (INT4 base weights)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        required=True,
        help="Path to INT4 Megatron checkpoint (converted from HF INT4)",
    )
    parser.add_argument(
        "--hf-model-path",
        type=str,
        default="moonshotai/Moonlight-16B-A3B",
        help="HF model path for tokenizer/config fallback",
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--sp", action="store_true", default=False)
    parser.add_argument("--train-iters", type=int, default=100)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="How often to print training logs.",
    )
    parser.add_argument("--coft", action="store_true", default=False)
    parser.add_argument("--eps", type=float, default=6e-5)
    parser.add_argument("--block-share", action="store_true", default=False)
    parser.add_argument("--module-dropout", type=float, default=0.0)
    parser.add_argument(
        "--int4-active-expert-chunk-size",
        type=int,
        default=4,
        help="Number of active experts to process per INT4 chunk in grouped expert linears. "
             "Set to 0 to disable the chunked sparse fallback.",
    )
    parser.add_argument(
        "--int4-grouped-chunk-backend",
        type=str,
        choices=("python", "te"),
        default="python",
        help="Execution backend inside grouped INT4 active-expert chunks. "
             "'python' uses per-expert F.linear, 'te' uses chunk-local TE grouped GEMM.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        default=False,
        help="Print model storage and CUDA allocator memory around the first training steps.",
    )
    parser.add_argument(
        "--profile-memory-steps",
        type=int,
        default=1,
        help="Number of initial training steps to profile when --profile-memory is enabled.",
    )
    parser.add_argument(
        "--debug-nan",
        action="store_true",
        default=False,
        help="Trace where non-finite values first appear in parameters, activations, or gradients.",
    )
    parser.add_argument(
        "--debug-nan-steps",
        type=int,
        default=3,
        help="Number of initial training steps to inspect when --debug-nan is enabled.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        default=False,
        help="Load the checkpoint and initialize OFT, then exit without training.",
    )
    parser.add_argument(
        "--memory-smoke-test",
        action="store_true",
        default=False,
        help="Run exactly one training step with memory profiling, then skip checkpoint save and evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_int4_active_expert_chunk_size(args.int4_active_expert_chunk_size)
    set_int4_grouped_chunk_backend(args.int4_grouped_chunk_backend)

    oft = OFT(
        block_size=args.block_size,
        coft=args.coft,
        eps=args.eps,
        block_share=args.block_share,
        module_dropout=args.module_dropout,
    )

    config = moonlight_16b_peft_config(oft)

    # Keep Moonlight recipe defaults for training/dataset/logger settings,
    # but build the model provider directly from the HF config so the runtime
    # architecture matches the checkpoint conversion path exactly.
    config.model = AutoBridge.from_hf_pretrained(
        args.hf_model_path, trust_remote_code=True,
    ).to_megatron_provider(load_weights=False)
    _patch_moonlight_build_tokenizer(config.model.vocab_size)

    # Build on meta device and only materialize during checkpoint load.
    config.model.init_model_with_meta_device = True
    config.model.perform_initialization = False

    config.checkpoint.pretrained_checkpoint = args.pretrained_checkpoint

    config.model.tensor_model_parallel_size = args.tp
    config.model.expert_model_parallel_size = args.ep
    config.model.pipeline_model_parallel_size = args.pp
    config.model.sequence_parallel = args.sp
    config.model.expert_tensor_parallel_size = 1
    config.model.virtual_pipeline_model_parallel_size = None
    config.model.context_parallel_size = 1
    config.model.pipeline_dtype = torch.bfloat16 if args.pp > 1 else None

    if args.pp > 1:
        config.model.pipeline_model_parallel_layout = _get_moonlight_pipeline_layout(args.pp, 1)
    else:
        config.model.pipeline_model_parallel_layout = None

    config.model.account_for_embedding_in_pipeline_split = False
    config.model.account_for_loss_in_pipeline_split = False

    config.model.seq_length = args.seq_length
    config.dataset.seq_length = args.seq_length
    config.dataset.packed_sequence_specs.packed_sequence_size = args.seq_length

    checkpoint_tokenizer_dir = os.path.join(
        args.pretrained_checkpoint, "iter_{:07d}".format(0), "tokenizer"
    )
    tokenizer_model = (
        checkpoint_tokenizer_dir if os.path.isdir(checkpoint_tokenizer_dir) else args.hf_model_path
    )
    config.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    config.tokenizer.tokenizer_model = tokenizer_model
    config.tokenizer.hf_tokenizer_kwargs = {"trust_remote_code": True}

    config.train.train_iters = args.train_iters
    config.train.global_batch_size = args.global_batch_size
    config.train.micro_batch_size = args.micro_batch_size
    config.train.manual_gc = True
    config.train.manual_gc_interval = 5
    config.train.manual_gc_eval = 5

    config.scheduler.lr_warmup_iters = 2
    config.scheduler.lr_decay_iters = args.train_iters

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

    config.checkpoint.save_interval = 500
    config.checkpoint.async_save = False

    if args.memory_smoke_test:
        args.profile_memory = True
        args.profile_memory_steps = max(1, args.profile_memory_steps)
        config.train.train_iters = 1
        config.validation.eval_iters = 0
        config.validation.eval_interval = None
        config.checkpoint.save_interval = 2
        config.logger.wandb_project = None
        config.logger.wandb_exp_name = None
        print_rank_0(
            "Running Moonlight INT4 memory smoke test: 1 training step, no checkpoint save, no evaluation."
        )

    if args.skip_train:
        config.validation.skip_train = True
        config.validation.eval_iters = 0
        config.validation.eval_interval = args.train_iters + 1

    output_dir = args.output_dir or os.path.join(
        os.getcwd(), "nemo_experiments", "moonlight_16b_qoft_int4"
    )
    config.checkpoint.save = os.path.join(output_dir, "checkpoints")
    config.checkpoint.load = os.path.join(output_dir, "checkpoints")
    config.logger.tensorboard_dir = os.path.join(output_dir, "tb_logs")

    config.logger.log_interval = args.log_interval
    config.logger.wandb_project = "megatron-bridge-finetuning"
    config.logger.wandb_exp_name = (
        f"moonlight_16b_qoft_int4_bs{args.block_size}_tp{args.tp}_ep{args.ep}"
    )

    if args.memory_smoke_test:
        config.logger.wandb_project = None
        config.logger.wandb_exp_name = None

    # Patch Megatron checkpoint loading to read INT4 expert triplets directly.
    # Current validated multi-rank targets are small Moonlight smoke configs:
    # EP-only resharded loading (TP=1, EP>1, PP=1) and mixed TP+EP smoke
    # tests (for example TP=2, EP=2, PP=1). Broader TP/PP reshard coverage
    # still needs explicit validation.
    import megatron.bridge.training.checkpointing as _ckpt_mod
    import megatron.core.dist_checkpointing.strategies.torch as _torch_strat
    import megatron.core.transformer.moe.experts as _moe_experts
    from megatron.core.dist_checkpointing.mapping import ShardedTensor as MCoreShardedTensor

    _orig_gen_model_sd = _ckpt_mod._generate_model_state_dict
    _orig_apply_swiglu_sharded_factory = _moe_experts.apply_swiglu_sharded_factory

    def _safe_apply_swiglu_sharded_factory(
        original_sh_ten, sharded_offsets, singleton_local_shards=False
    ):
        local_shape = getattr(original_sh_ten, "local_shape", None)
        if local_shape is not None and len(local_shape) > 0 and local_shape[0] == 0:
            return original_sh_ten
        return _orig_apply_swiglu_sharded_factory(
            original_sh_ten, sharded_offsets, singleton_local_shards
        )

    _moe_experts.apply_swiglu_sharded_factory = _safe_apply_swiglu_sharded_factory

    def _int4_generate_model_state_dict(model, model_sd_kwargs=None, ckpt_format="torch_dist", **kwargs):
        state_dict = _orig_gen_model_sd(model, model_sd_kwargs, ckpt_format, **kwargs)
        for model_key in list(state_dict.keys()):
            if model_key.startswith("model"):
                state_dict[model_key] = transform_sharded_state_dict_for_int4(state_dict[model_key])
        return state_dict

    _ckpt_mod._generate_model_state_dict = _int4_generate_model_state_dict

    _orig_mcore_to_pyt = _torch_strat.mcore_to_pyt_state_dict

    def _meta_safe_mcore_to_pyt(state_dict, is_loading=False, **kwargs):
        if is_loading:
            for _, val in state_dict.items():
                if isinstance(val, list):
                    for sh_ten in val:
                        if isinstance(sh_ten, MCoreShardedTensor):
                            if sh_ten.data is not None and sh_ten.data.device.type == "meta":
                                sh_ten.data = torch.empty(
                                    sh_ten.local_shape, dtype=sh_ten.dtype, device="cpu"
                                )
                elif isinstance(val, MCoreShardedTensor):
                    if val.data is not None and val.data.device.type == "meta":
                        val.data = torch.empty(val.local_shape, dtype=val.dtype, device="cpu")
        return _orig_mcore_to_pyt(state_dict, is_loading, **kwargs)

    _torch_strat.mcore_to_pyt_state_dict = _meta_safe_mcore_to_pyt

    _orig_load_model_sd = _ckpt_mod._load_model_state_dict

    def _int4_load_model_state_dict(model_module, state_dict, strict=True):
        register_int4_buffers_after_load(model_module, state_dict)
        int4_keys = [
            k for k in state_dict
            if k.endswith(("_packed", "_scale", "_shape")) and ".experts.linear_fc" in k
        ]
        for key in int4_keys:
            del state_dict[key]

        load_return = model_module.load_state_dict(state_dict, strict=False, assign=True)
        missing = [key for key in load_return.missing_keys if not _is_expected_int4_missing_key(key)]
        unexpected = [key for key in load_return.unexpected_keys if not key.endswith("._extra_state")]

        if missing or unexpected:
            details = []
            if missing:
                details.append(
                    "missing="
                    + ", ".join(missing[:20])
                    + (" ..." if len(missing) > 20 else "")
                )
            if unexpected:
                details.append(
                    "unexpected="
                    + ", ".join(unexpected[:20])
                    + (" ..." if len(unexpected) > 20 else "")
                )
            raise RuntimeError(
                "Unexpected non-INT4 state_dict mismatch during Moonlight INT4 load: "
                + " | ".join(details)
            )

        if torch.cuda.is_available():
            to_empty_if_meta_device(
                model_module, device=torch.device("cuda", torch.cuda.current_device())
            )
        _validate_loaded_model_tensors(model_module)

    _ckpt_mod._load_model_state_dict = _int4_load_model_state_dict

    callbacks = []
    if args.profile_memory:
        callbacks.append(MemoryProfileCallback(args.profile_memory_steps))
    if args.debug_nan:
        callbacks.append(NanTraceCallback(args.debug_nan_steps))

    finetune(config=config, forward_step_func=forward_step, callbacks=callbacks or None)


if __name__ == "__main__":
    main()
