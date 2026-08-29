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

"""Shared machinery for quantized-base (QOFT) finetuning entrypoints.

Everything here was consolidated verbatim from the retired per-model
entrypoints (kimi_k25 int4/nvfp4, moonlight_16b int4, qwen3_moe fp8/int4):
the INT4 and FP8 checkpoint-load monkey-patch stacks, the Moonshot tokenizer
vocab clamp, the memory-profile and NaN-trace diagnostics, and small config
helpers. ``scripts/orbit/finetune_qoft.py`` is the only intended consumer.
"""

import logging
import re
from collections import defaultdict
from typing import Any, Iterable

import torch
import torch.distributed as dist

from megatron.bridge.training.callbacks import Callback
from megatron.bridge.utils.common_utils import print_rank_0


logger = logging.getLogger(__name__)

INT4_SCALE_DTYPE = torch.bfloat16
FP8_WEIGHT_BLOCK_SIZE = 128


# --------------------------------------------------------------------------- #
# Small config helpers
# --------------------------------------------------------------------------- #


def parse_target_modules(value: str) -> list[str]:
    """Split a comma/whitespace separated module list argument."""
    return [module for module in re.split(r"[\s,]+", value.strip()) if module]


def set_sequence_length(config, seq_length: int) -> None:
    """Apply one sequence length to the model, dataset, and packing specs."""
    config.model.seq_length = seq_length
    if getattr(config, "dataset", None) is not None:
        config.dataset.seq_length = seq_length
        packed_sequence_specs = getattr(config.dataset, "packed_sequence_specs", None)
        if packed_sequence_specs is not None:
            packed_sequence_specs.packed_sequence_size = seq_length


def disable_evaluation(config) -> None:
    """Turn off validation/evaluation entirely."""
    config.validation.eval_iters = 0
    config.validation.eval_interval = None


def normalize_hf_dataset_source(config) -> None:
    """Namespace bare HF dataset ids that huggingface_hub >= 1.0 rejects.

    The upstream recipes hard-code ``dataset_name="squad"``; hub 1.x requires
    ``namespace/name`` repository ids, so loading fails with HfUriError before
    training starts. Rewrite the known bare ids to their canonical namespaced
    form (the identical dataset).
    """
    source = getattr(getattr(config, "dataset", None), "source", None)
    name = getattr(source, "dataset_name", None)
    if name == "squad":
        source.dataset_name = "rajpurkar/squad"


def tokenizer_model_from_checkpoint(pretrained_checkpoint: str, hf_model_path: str) -> str:
    """Prefer the tokenizer assets saved into the converted checkpoint."""
    import os

    checkpoint_tokenizer_dir = os.path.join(pretrained_checkpoint, "iter_{:07d}".format(0), "tokenizer")
    return checkpoint_tokenizer_dir if os.path.isdir(checkpoint_tokenizer_dir) else hf_model_path


# --------------------------------------------------------------------------- #
# Moonshot tokenizer vocab clamp (Moonlight)
# --------------------------------------------------------------------------- #


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


def _is_moonshot_tokenizer(tokenizer) -> bool:
    if tokenizer is None:
        return False

    name_or_path = str(getattr(tokenizer, "name_or_path", "") or "").lower()
    if "moonlight" in name_or_path or "moonshot" in name_or_path:
        return True

    tokenizer_type = type(tokenizer).__name__.lower()
    tokenizer_module = type(tokenizer).__module__.lower()
    return "moonshot" in tokenizer_module or tokenizer_type == "tiktokentokenizer"


def patch_moonshot_build_tokenizer(model_vocab_size: int) -> None:
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


# --------------------------------------------------------------------------- #
# Memory diagnostics
# --------------------------------------------------------------------------- #


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


def _classify_quant_buffer(name: str, quant: str) -> str:
    """Bucket quantized-base buffers by role for memory accounting."""
    if quant == "int4":
        if name.endswith("_packed"):
            return "int4_packed_storage_bytes"
        if name.endswith("_scale"):
            return "int4_scale_storage_bytes"
        if name.endswith("_shape"):
            return "int4_shape_storage_bytes"
        return "other_buffer_storage_bytes"
    if quant == "nvfp4":
        if name.endswith(".weight_quantizer._scale") or "._weight_quantizer._scale" in name:
            return "nvfp4_block_scale_storage_bytes"
        if name.endswith(".weight_quantizer._amax") or "._weight_quantizer._amax" in name:
            return "nvfp4_weight_amax_storage_bytes"
        if name.endswith(".input_quantizer._amax") or "._input_quantizer._amax" in name:
            return "nvfp4_input_amax_storage_bytes"
        if name.endswith(".weight_quantizer._double_scale"):
            return "nvfp4_double_scale_storage_bytes"
        return "other_buffer_storage_bytes"
    if quant == "fp8" and name.endswith("weight_scale_inv"):
        return "fp8_scale_inv_storage_bytes"
    return "other_buffer_storage_bytes"


def _summarize_model_storage(model_chunks, quant: str) -> dict[str, int]:
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
            _add_tensor_storage(summary, seen_storages, _classify_quant_buffer(name, quant), buffer)

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


def _collect_buffer_entries(model_chunks, quant: str) -> tuple[list[dict], list[dict]]:
    quant_entries = []
    other_entries = []

    for model_chunk in model_chunks:
        for name, buffer in model_chunk.named_buffers():
            if buffer is None:
                continue

            nbytes = _tensor_storage_nbytes(buffer)
            if nbytes == 0:
                continue

            bucket = _classify_quant_buffer(name, quant)
            entry = {
                "name": name,
                "nbytes": nbytes,
                "shape": tuple(buffer.shape),
                "dtype": str(buffer.dtype).replace("torch.", ""),
                "kind": bucket.replace("_storage_bytes", ""),
            }

            if bucket == "other_buffer_storage_bytes":
                other_entries.append(entry)
            else:
                quant_entries.append(entry)

    quant_entries.sort(key=lambda item: item["nbytes"], reverse=True)
    other_entries.sort(key=lambda item: item["nbytes"], reverse=True)
    return quant_entries, other_entries


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
            "[memory:{title}] {idx:02d} {size:>10}  {dtype:<8}  {kind:<24}  {shape}  {name}".format(
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


def log_model_storage_summary(model_chunks, quant: str) -> None:
    """Print unique-storage totals and the largest parameters/buffers."""
    summary = _summarize_model_storage(model_chunks, quant)
    frozen_entries, trainable_entries = _collect_parameter_entries(model_chunks)
    quant_buffer_entries, other_buffer_entries = _collect_buffer_entries(model_chunks, quant)
    quant_buckets = sorted(key for key in summary if key.startswith(("int4_", "nvfp4_", "fp8_")))
    quant_text = " ".join(
        "{}={}".format(key.replace("_storage_bytes", ""), _format_nbytes(summary[key])) for key in quant_buckets
    )
    lines = [
        "[memory:model-storage] unique_storage={}".format(
            _format_nbytes(summary.get("total_unique_storage_bytes", 0))
        ),
        "[memory:model-storage] frozen_params={} trainable_params={} empty_param_tensors={}".format(
            _format_nbytes(summary.get("frozen_parameter_storage_bytes", 0)),
            _format_nbytes(summary.get("trainable_parameter_storage_bytes", 0)),
            summary.get("empty_parameter_tensors", 0),
        ),
        "[memory:model-storage] {} other_buffers={}".format(
            quant_text or "quant_buffers=0.00 B",
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
    _log_buffer_table(f"{quant}-buffers", quant_buffer_entries)
    _log_buffer_table("other-buffers", other_buffer_entries)


class MemoryProfileCallback(Callback):
    """Log model storage and CUDA allocator stats around the first steps."""

    def __init__(self, profile_steps: int, quant: str):
        self.profile_steps = max(1, profile_steps)
        self.quant = quant

    def on_train_start(self, context) -> None:
        if not torch.cuda.is_available():
            return
        log_model_storage_summary(context.model, self.quant)
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


# --------------------------------------------------------------------------- #
# NaN tracing
# --------------------------------------------------------------------------- #


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


def validate_loaded_model_tensors(model_module: torch.nn.Module) -> None:
    """Fail fast if any loaded parameter or buffer is non-finite."""
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


class NanTraceCallback(Callback):
    """Trace where non-finite values first appear in params, activations, grads."""

    def __init__(self, trace_steps: int):
        self.trace_steps = max(1, trace_steps)
        self._current_step: int | None = None
        self._forward_hit = False
        self._backward_hit = False
        self._hook_handles = []

    def _should_trace(self, step: int) -> bool:
        return step < self.trace_steps

    def _should_hook_module(self, module: torch.nn.Module) -> bool:
        from megatron.bridge.orbit.oft.oft_layers import OFTLinear, OFTRotationModule, OFTTopKRouter

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


# --------------------------------------------------------------------------- #
# INT4 checkpoint-load patches
# --------------------------------------------------------------------------- #

_EXPERT_INT4_MISSING_KEY_RE = re.compile(
    r".*\.experts\.linear_fc[12]\.(?:weight\d+|weight|weight\d+_(?:packed|scale|shape))$"
)
_DENSE_INT4_TRIPLET_KEY_RE = re.compile(
    r"^.*\.(?:linear_qkv|linear_proj|linear_fc1|linear_fc2|router)\.weight_(?:packed|scale|shape)$"
)
_EXPERT_INT4_TRIPLET_KEY_RE = re.compile(r"^.*\.experts\.linear_fc[12]\.weight\d+_(?:packed|scale|shape)$")
_EXPECTED_DENSE_INT4_MISSING_KEY_RE = re.compile(
    r"^.*\.(?:linear_qkv|linear_proj|linear_fc1|linear_fc2|router)\.(?:weight|weight_(?:packed|scale|shape))$"
)


def _is_expected_int4_missing_key(key: str, scope: str) -> bool:
    if key.endswith("._extra_state"):
        return True
    if _EXPERT_INT4_MISSING_KEY_RE.fullmatch(key) is not None:
        return True
    return scope == "all" and _EXPECTED_DENSE_INT4_MISSING_KEY_RE.match(key) is not None


def _is_int4_triplet_key(key: str, scope: str) -> bool:
    if _EXPERT_INT4_TRIPLET_KEY_RE.match(key) is not None:
        return True
    return scope == "all" and _DENSE_INT4_TRIPLET_KEY_RE.match(key) is not None


def _drop_extra_state_entries(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state_dict.items() if "._extra_state" not in str(key)}


def _materialize_meta_sharded_tensors_to_cpu(state_dict, sharded_tensor_cls) -> None:
    for value in state_dict.values():
        if isinstance(value, list):
            for sharded_tensor in value:
                if not isinstance(sharded_tensor, sharded_tensor_cls):
                    continue
                if sharded_tensor.data is not None and sharded_tensor.data.device.type == "meta":
                    sharded_tensor.data = torch.empty(
                        sharded_tensor.local_shape, dtype=sharded_tensor.dtype, device="cpu"
                    )
        elif isinstance(value, sharded_tensor_cls):
            if value.data is not None and value.data.device.type == "meta":
                value.data = torch.empty(value.local_shape, dtype=value.dtype, device="cpu")


def install_int4_checkpoint_load_patches(
    *,
    scope: str,
    group_size: int,
    arch_label: str,
    validate_nonfinite: bool = False,
    after_load_hook=None,
) -> None:
    """Patch Megatron checkpoint loading to read INT4 triplets directly.

    ``scope`` selects the checkpoint flavor: ``"experts"`` for expert-only INT4
    triplets (Kimi / Moonlight converted checkpoints) and ``"all"`` for
    checkpoints that additionally quantize the dense linears and router and
    were saved with ``non_homogeneous_layers`` metadata (Qwen3 MoE).

    The three patches, consolidated from the retired per-model entrypoints:

    1. ``_generate_model_state_dict``: rewrite expert (and, for ``scope="all"``,
       dense) BF16 weight entries into INT4 triplet entries so
       ``dist_checkpointing.load()`` reads INT4 data directly.
    2. ``mcore_to_pyt_state_dict``: materialize meta-device sharded tensors to
       CPU right before PyTorch's checkpoint planner sees them.
    3. ``_load_model_state_dict``: register INT4 triplets as module buffers,
       assign-load the rest, validate the remaining missing/unexpected keys,
       and materialize any leftover meta tensors onto the local CUDA device.

    A zero-local-shard guard on ``apply_swiglu_sharded_factory`` keeps EP
    resharding safe when a rank holds no experts for a layer.
    """
    if scope not in ("experts", "all"):
        raise ValueError(f"install_int4_checkpoint_load_patches: unknown scope {scope!r}")

    import megatron.core.dist_checkpointing.strategies.torch as _torch_strat
    import megatron.core.transformer.moe.experts as _moe_experts
    from megatron.core.dist_checkpointing.mapping import ShardedTensor as MCoreShardedTensor

    import megatron.bridge.training.checkpointing as _ckpt_mod
    from megatron.bridge.models.common.unimodal import to_empty_if_meta_device
    from megatron.bridge.orbit.quant.int4_utils import (
        register_int4_buffers_after_load,
        transform_sharded_state_dict_for_int4,
    )

    if getattr(_ckpt_mod, "_qoft_int4_checkpoint_patches_installed", False):
        return

    if scope == "all":
        from megatron.bridge.orbit.low_precision.int4 import (
            register_int4_buffers_after_load_dense,
            transform_sharded_state_dict_for_int4_dense,
        )

    original_generate_model_sd = _ckpt_mod._generate_model_state_dict
    original_apply_swiglu_sharded_factory = _moe_experts.apply_swiglu_sharded_factory
    original_mcore_to_pyt = _torch_strat.mcore_to_pyt_state_dict

    def _safe_apply_swiglu_sharded_factory(original_sh_ten, *args, **kwargs):
        # Signature-agnostic: megatron.core has grown keyword arguments here
        # (e.g. tp_group), and the retired entrypoints' fixed three-argument
        # copy raised TypeError against the current pin.
        local_shape = getattr(original_sh_ten, "local_shape", None)
        if local_shape is not None and len(local_shape) > 0 and local_shape[0] == 0:
            return original_sh_ten
        return original_apply_swiglu_sharded_factory(original_sh_ten, *args, **kwargs)

    def _int4_generate_model_state_dict(model, model_sd_kwargs=None, ckpt_format="torch_dist", **kwargs):
        if scope == "all":
            model_sd_kwargs = dict(model_sd_kwargs or {})
            metadata = dict(model_sd_kwargs.get("metadata") or {})
            metadata["non_homogeneous_layers"] = True
            model_sd_kwargs["metadata"] = metadata

        state_dict = original_generate_model_sd(model, model_sd_kwargs, ckpt_format, **kwargs)
        if scope == "all" and ckpt_format != "torch_dist":
            return state_dict

        for model_key in list(state_dict.keys()):
            if not model_key.startswith("model"):
                continue
            model_state = state_dict[model_key]
            if scope == "all":
                model_state = _drop_extra_state_entries(model_state)
                model_state = transform_sharded_state_dict_for_int4(
                    model_state, group_size=group_size, scale_dtype=INT4_SCALE_DTYPE
                )
                model_state = transform_sharded_state_dict_for_int4_dense(
                    model_state, group_size=group_size, scale_dtype=INT4_SCALE_DTYPE
                )
            else:
                model_state = transform_sharded_state_dict_for_int4(model_state)
            state_dict[model_key] = model_state
        return state_dict

    def _meta_safe_mcore_to_pyt(state_dict, is_loading=False, **kwargs):
        if is_loading:
            _materialize_meta_sharded_tensors_to_cpu(state_dict, MCoreShardedTensor)
        return original_mcore_to_pyt(state_dict, is_loading, **kwargs)

    def _int4_load_model_state_dict(model_module, state_dict, strict=True):
        if scope == "all":
            state_dict = _drop_extra_state_entries(state_dict)
        register_int4_buffers_after_load(model_module, state_dict)
        if scope == "all":
            register_int4_buffers_after_load_dense(model_module, state_dict)

        for key in [key for key in state_dict if isinstance(key, str) and _is_int4_triplet_key(key, scope)]:
            del state_dict[key]

        load_return = model_module.load_state_dict(state_dict, strict=False, assign=True)
        missing = [key for key in load_return.missing_keys if not _is_expected_int4_missing_key(key, scope)]
        unexpected = [key for key in load_return.unexpected_keys if not key.endswith("._extra_state")]

        if missing or unexpected:
            details = []
            if missing:
                details.append("missing=" + ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))
            if unexpected:
                details.append("unexpected=" + ", ".join(unexpected[:20]) + (" ..." if len(unexpected) > 20 else ""))
            raise RuntimeError(
                f"Unexpected non-INT4 state_dict mismatch during {arch_label} INT4 load: " + " | ".join(details)
            )

        if torch.cuda.is_available():
            to_empty_if_meta_device(model_module, device=torch.device("cuda", torch.cuda.current_device()))
            if after_load_hook is not None:
                after_load_hook(model_module)
        if validate_nonfinite:
            validate_loaded_model_tensors(model_module)

    _moe_experts.apply_swiglu_sharded_factory = _safe_apply_swiglu_sharded_factory
    _ckpt_mod._generate_model_state_dict = _int4_generate_model_state_dict
    _torch_strat.mcore_to_pyt_state_dict = _meta_safe_mcore_to_pyt
    _ckpt_mod._load_model_state_dict = _int4_load_model_state_dict
    _ckpt_mod._qoft_int4_checkpoint_patches_installed = True


# --------------------------------------------------------------------------- #
# FP8 (quantized base) checkpoint-load patches
# --------------------------------------------------------------------------- #


def infer_fp8_scale_inv_shape(weight: torch.Tensor, *, block_size: int = FP8_WEIGHT_BLOCK_SIZE) -> tuple[int, ...]:
    """Return the block-wise ``weight_scale_inv`` shape for an FP8 weight."""
    import math

    if weight.ndim == 1:
        return (max(1, math.ceil(weight.shape[0] / block_size)),)
    if weight.ndim >= 2:
        leading_dims = tuple(weight.shape[:-2])
        out_blocks = max(1, math.ceil(weight.shape[-2] / block_size))
        in_blocks = max(1, math.ceil(weight.shape[-1] / block_size))
        return leading_dims + (out_blocks, in_blocks)
    raise ValueError(f"Unsupported FP8 weight rank {weight.ndim}")


def ensure_fp8_scale_inv_buffers(model):
    """Pre-wrap hook: give every FP8 weight a correctly shaped scale_inv buffer."""
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


def install_fp8_checkpoint_load_patches() -> None:
    """Patch checkpoint loading to read FP8 base weights with their block scales."""
    import megatron.bridge.training.checkpointing as _ckpt_mod
    from megatron.bridge.orbit.quant.fp8_utils import (
        register_fp8_scale_inv_buffers_after_load,
        transform_sharded_state_dict_for_fp8,
    )

    if getattr(_ckpt_mod, "_qoft_fp8_checkpoint_patches_installed", False):
        return

    original_generate_model_sd = _ckpt_mod._generate_model_state_dict
    original_load_model_sd = _ckpt_mod._load_model_state_dict

    def _fp8_generate_model_state_dict(model, model_sd_kwargs=None, ckpt_format="torch_dist", **kwargs):
        state_dict = original_generate_model_sd(model, model_sd_kwargs, ckpt_format, **kwargs)
        for model_key in list(state_dict.keys()):
            if model_key.startswith("model"):
                state_dict[model_key] = transform_sharded_state_dict_for_fp8(state_dict[model_key])
        return state_dict

    def _fp8_load_model_state_dict(model_module, state_dict, strict=True):
        register_fp8_scale_inv_buffers_after_load(model_module, state_dict)
        original_load_model_sd(model_module, state_dict, strict)

    _ckpt_mod._generate_model_state_dict = _fp8_generate_model_state_dict
    _ckpt_mod._load_model_state_dict = _fp8_load_model_state_dict
    _ckpt_mod._qoft_fp8_checkpoint_patches_installed = True
