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

"""Multi-GPU direct-write HuggingFace NVFP4 -> Megatron checkpoint conversion.

This is the distributed counterpart to ``convert_nvfp4_checkpoint_direct.py``.
It preserves the NVFP4-specific direct-conversion path (avoid full BF16
materialization via meta-device build + ModelOpt NVFP4 transform + ``TensorSpillManager``)
while spreading the model across multiple ranks via TP/PP/EP, so trillion-class
NVFP4 checkpoints (e.g. Kimi-K2.5 NVFP4) can be converted without the rank-0 OOM
that the single-rank script hits at ``to_empty_if_meta_device(... cuda)``.

Default layout: ``--tp 1 --ep 8`` — matches the typical Kimi / DeepSeek-V3 training
and rollout layout (DP attention with experts EP-partitioned across 8 ranks).

Usage:

    uv run python -m torch.distributed.run --nproc_per_node=8 \\
        scripts/orbit/conversion/convert_nvfp4_checkpoint_direct_multigpu.py \\
        --hf-model-path /path/to/Kimi-K2.5-NVFP4 \\
        --megatron-path /path/to/output \\
        --tp 1 --ep 8

    # Mixed TP and EP (e.g. 4-way TP attention, 2-way EP experts on 8 GPUs)
    uv run python -m torch.distributed.run --nproc_per_node=8 \\
        scripts/orbit/conversion/convert_nvfp4_checkpoint_direct_multigpu.py \\
        --hf-model-path /path/to/Kimi-K2.5-NVFP4 \\
        --megatron-path /path/to/output \\
        --tp 4 --ep 2

    # Multi-node via Slurm srun
    srun --ntasks-per-node=8 ... python \\
        scripts/orbit/conversion/convert_nvfp4_checkpoint_direct_multigpu.py \\
        --hf-model-path /path/to/Kimi-K2.5-NVFP4 \\
        --megatron-path /path/to/output \\
        --tp 1 --ep 8

Environment variables (carry over from the single-rank script):
  MEGATRON_BRIDGE_DIRECT_SAVE_PROGRESS=1     enable rank-0 save-progress monitor
  MEGATRON_BRIDGE_DIRECT_SAVE_PROGRESS_INTERVAL=10  monitor interval seconds
  MEGATRON_BRIDGE_DIRECT_SPILL_DIR=<path>    directory for tensor spill files
  MEGATRON_BRIDGE_DIRECT_STORAGE_WRITERS_PER_RANK=16  parallel storage writers
"""

import argparse
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

# Force mcore async strategy everywhere (modelopt's save_sharded_modelopt_state
# calls dist_checkpointing.save without passing async_strategy, which defaults
# to "nvrx" and fails when nvidia-resiliency-ext is unavailable/mismatched).
from megatron.core import dist_checkpointing as _dc

_orig_dc_save = _dc.save


def _dc_save_with_mcore(*args, **kwargs):
    kwargs.setdefault("async_strategy", "mcore")
    return _orig_dc_save(*args, **kwargs)


_dc.save = _dc_save_with_mcore

import torch

# Force the meta -> empty materialization to land on CPU, not CUDA. The default
# `to_empty_if_meta_device(model, device="cuda")` in model_provider.get_model
# tries to allocate the full per-rank parameter footprint as BF16 on the GPU
# *before* the NVFP4 transform replaces dense linears with quantized buffers.
# For Kimi-K2.5 with TP=1/EP=8, attention + embeddings + dense MLP are
# replicated on every rank, which OOMs a 178GB B200 immediately. The whole
# direct-NVFP4 flow is CPU + spill-to-disk, so CPU materialization is correct
# and matches what `use_cpu_initialization=True` already implies.
import megatron.bridge.models.common.unimodal as _unimodal
import megatron.bridge.models.model_provider as _mp

_orig_to_empty_if_meta_device = _unimodal.to_empty_if_meta_device


def _to_empty_if_meta_device_cpu(module, *, device=None, recurse=True):
    return _orig_to_empty_if_meta_device(
        module, device=torch.device("cpu"), recurse=recurse
    )


_unimodal.to_empty_if_meta_device = _to_empty_if_meta_device_cpu
_mp.to_empty_if_meta_device = _to_empty_if_meta_device_cpu

from megatron.core.optimizer import OptimizerConfig

from megatron.bridge import AutoBridge
from megatron.bridge.orbit.low_precision.common import (
    TensorSpillManager,
    patch_meta_init_for_te_modules,
)


# Bucketed spill: one mmap-shared file per decoder layer (plus a "misc" bucket
# for non-layer tensors like embeddings/lm_head/router) instead of one file per
# tensor. The original per-tensor flow created ~24K files per rank (8 ranks ×
# ~3K NVFP4 tensors per rank for Kimi-K2.5 with EP=8/384 experts), which
# saturated the BeeGFS metadata server and surfaced as `Remote I/O error (121)`
# at `torch.from_file(...).resize` (ftruncate). Bucketing by layer drops that
# to ~60 ftruncates per rank — a >100x reduction in MDS load.
#
# Implementation: each bucket is a `torch.UntypedStorage.from_file(shared=True,
# nbytes=PREALLOC)` — one mmap, one ftruncate per bucket, sparse on disk so
# the prealloc only consumes virtual address space + actually-written bytes.
# Per-tensor `spill_tensor` calls advance a byte cursor and return a
# Tensor view (`.set_(storage, offset, shape, stride)`) into the shared mmap,
# preserving the same "mmap-backed, low-RAM" semantics as the original.
import re as _re
import threading as _threading

_LAYER_RE = _re.compile(r"decoder\.layers\.(\d+)\.")
_BUCKET_PREALLOC = 256 * 1024 * 1024  # 256 MiB per bucket — sparse on disk


class _SpillBucket:
    __slots__ = ("path", "storage", "cursor", "lock")

    def __init__(self, path, prealloc_bytes):
        self.path = path
        # Single ftruncate-via-mmap. Errors are still possible here, but only
        # once per bucket — if BeeGFS fails this we have bigger problems.
        self.storage = torch.UntypedStorage.from_file(
            str(path), shared=True, nbytes=prealloc_bytes
        )
        self.cursor = 0  # next free byte offset
        self.lock = _threading.Lock()


def _bucketed_spill_tensor(self, key, tensor):
    tensor = tensor.detach().cpu().contiguous()
    if tensor.numel() == 0:
        return tensor

    if not hasattr(self, "_buckets"):
        self._buckets = {}
        self._buckets_lock = _threading.Lock()

    match = _LAYER_RE.search(key)
    bucket_name = (
        f"layer_{int(match.group(1)):03d}" if match is not None else "misc"
    )

    with self._buckets_lock:
        bucket = self._buckets.get(bucket_name)
        if bucket is None:
            bpath = self.root_dir / f"{bucket_name}.bin"
            bucket = _SpillBucket(bpath, _BUCKET_PREALLOC)
            self._buckets[bucket_name] = bucket
            self._paths.append(bpath)

    nbytes = tensor.numel() * tensor.element_size()
    elem_size = tensor.element_size()
    with bucket.lock:
        # Align byte cursor up to the tensor's element size so set_'s
        # element-offset arithmetic is exact.
        cursor = (bucket.cursor + elem_size - 1) // elem_size * elem_size
        if cursor + nbytes > bucket.storage.nbytes():
            raise RuntimeError(
                f"Spill bucket {bucket.path} would overflow at "
                f"cursor={cursor} + nbytes={nbytes} > "
                f"prealloc={bucket.storage.nbytes()}. "
                f"Increase _BUCKET_PREALLOC."
            )
        elem_offset = cursor // elem_size
        view = torch.empty(0, dtype=tensor.dtype).set_(
            bucket.storage, elem_offset, tensor.shape, tensor.stride()
        )
        view.copy_(tensor)
        bucket.cursor = cursor + nbytes

    return view


TensorSpillManager.spill_tensor = _bucketed_spill_tensor
from megatron.bridge.orbit.low_precision.nvfp4 import (
    apply_modelopt_nvfp4_to_meta_model,
    build_nvfp4_direct_model_state_dict,
    collect_nvfp4_target_module_names,
    is_nvfp4_source,
)
from megatron.bridge.models.decorators import torchrun_main
from megatron.bridge.orbit.model_bridges.kimi_k25_vl_nvfp4_bridge import KimiK25VLNVFP4Bridge
from megatron.bridge.training.checkpointing import (
    get_checkpoint_name,
    save_checkpoint,
    save_tokenizer_assets,
)
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    LoggerConfig,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.bridge.utils.common_utils import print_rank_0


_DECODER_LAYER_RE = re.compile(r"decoder\.layers\.(\d+)\.")


# ---------------------------------------------------------------------------
# Stage timing / logging helpers (rank-0 only via print_rank_0)
# ---------------------------------------------------------------------------


def _format_elapsed(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def _log_stage_start(message: str) -> float:
    print_rank_0(message)
    return time.monotonic()


def _log_stage_done(
    message: str, start_time: float, *, extra: Optional[str] = None
) -> None:
    suffix = f" | {extra}" if extra else ""
    print_rank_0(
        f"{message} in {_format_elapsed(time.monotonic() - start_time)}{suffix}"
    )


# ---------------------------------------------------------------------------
# Debug layer-range filter (mirror of single-rank script)
# ---------------------------------------------------------------------------


def _parse_debug_layer_range(value: str) -> tuple[int, int]:
    try:
        start_str, end_str = value.split(":", 1)
        start = int(start_str)
        end = int(end_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid debug layer range {value!r}. Expected START:END, e.g. 0:4."
        ) from exc

    if start < 0 or end < 0:
        raise argparse.ArgumentTypeError(
            "Debug layer range indices must be non-negative."
        )
    if end <= start:
        raise argparse.ArgumentTypeError(
            f"Invalid debug layer range {value!r}. END must be greater than START."
        )

    return start, end


def _bridge_name(architecture: Any) -> str:
    if isinstance(architecture, str):
        return architecture
    return architecture.__name__


def _select_nvfp4_bridge(auto_bridge: Any) -> Any:
    architecture_name = _bridge_name(auto_bridge._causal_lm_architecture)
    if architecture_name == "KimiK25ForConditionalGeneration":
        bridge = KimiK25VLNVFP4Bridge()
        if hasattr(auto_bridge.hf_pretrained, "config"):
            bridge.hf_config = auto_bridge.hf_pretrained.config
        return bridge
    return auto_bridge._model_bridge


def _task_decoder_layer_idx(task: Any) -> Optional[int]:
    if task is None:
        return None
    param_name = getattr(task, "param_name", None)
    if not isinstance(param_name, str):
        return None
    match = _DECODER_LAYER_RE.search(param_name)
    return int(match.group(1)) if match else None


def _filter_conversion_tasks_for_debug_layer_range(
    conversion_tasks: list[Any],
    layer_range: tuple[int, int],
) -> list[Any]:
    start, end = layer_range
    filtered = []
    for task in conversion_tasks:
        layer_idx = _task_decoder_layer_idx(task)
        if layer_idx is None:
            continue
        if start <= layer_idx < end:
            filtered.append(task)
    return filtered


# ---------------------------------------------------------------------------
# Save-progress monitor (rank-0 only — directory inspection is local)
# ---------------------------------------------------------------------------


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _format_num_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if value < 1024.0 or unit == "PiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


class _SaveProgressMonitor:
    def __init__(self, checkpoint_root: str | Path, interval_sec: float = 10.0):
        self.checkpoint_root = Path(checkpoint_root)
        self.interval_sec = max(interval_sec, 1.0)
        self._start_time = 0.0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="nvfp4-direct-save-progress",
            daemon=True,
        )
        self._last_snapshot: Optional[tuple[int, int]] = None

    def start(self) -> None:
        self._start_time = time.monotonic()
        print_rank_0(
            f"Save progress monitor enabled for {self.checkpoint_root} "
            f"(interval={self.interval_sec:.1f}s)"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self.interval_sec + 1.0)
        self._emit_progress(final=True)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_sec):
            self._emit_progress()

    def _snapshot(self) -> tuple[int, int]:
        root = self.checkpoint_root
        if not root.exists():
            return 0, 0

        file_count = 0
        total_bytes = 0
        for path in root.rglob("*"):
            try:
                if not path.is_file():
                    continue
                stat_result = path.stat()
            except FileNotFoundError:
                continue
            file_count += 1
            total_bytes += stat_result.st_size
        return file_count, total_bytes

    def _emit_progress(self, *, final: bool = False) -> None:
        snapshot = self._snapshot()
        if not final and snapshot == self._last_snapshot:
            return

        self._last_snapshot = snapshot
        file_count, total_bytes = snapshot
        elapsed = _format_elapsed(time.monotonic() - self._start_time)
        label = "final save progress" if final else "save progress"
        print_rank_0(
            f"[{label}] elapsed {elapsed} | files {file_count} | "
            f"written {_format_num_bytes(total_bytes)}"
        )


def _maybe_create_save_progress_monitor(
    path: str | Path,
) -> Optional[_SaveProgressMonitor]:
    # Only rank 0 inspects the local checkpoint directory; other ranks would
    # see incomplete views (each writes its own shard via dist_checkpointing).
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return None
    if not _env_flag_enabled("MEGATRON_BRIDGE_DIRECT_SAVE_PROGRESS"):
        return None

    try:
        interval_sec = float(
            os.environ.get("MEGATRON_BRIDGE_DIRECT_SAVE_PROGRESS_INTERVAL", "10")
        )
    except ValueError:
        interval_sec = 10.0

    return _SaveProgressMonitor(path, interval_sec=interval_sec)


# ---------------------------------------------------------------------------
# Multi-rank meta provider (mirror of build_single_rank_meta_provider but with
# the user's TP/PP/EP/ETP layout instead of forcing 1/1/1)
# ---------------------------------------------------------------------------


def _build_multi_rank_meta_provider(
    hf_model_path: str,
    *,
    tp: int,
    pp: int,
    ep: int,
    etp: int,
    trust_remote_code: bool,
):
    """Build a multi-rank provider for distributed NVFP4 conversion.

    Same shape as ``build_single_rank_meta_provider`` (no weight load,
    ``perform_initialization=False``) but configured for the user's requested
    parallelism layout. Each rank then materializes only its own shard during
    ``provide_distributed_model``, so the per-rank GPU footprint scales as
    ``model_size / (tp * pp * ep * etp)``.
    """
    auto_bridge = AutoBridge.from_hf_pretrained(
        hf_model_path,
        trust_remote_code=trust_remote_code,
    )
    provider = auto_bridge.to_megatron_provider(load_weights=False)
    provider.perform_initialization = False

    for attr, value in (
        ("tensor_model_parallel_size", tp),
        ("pipeline_model_parallel_size", pp),
        ("context_parallel_size", 1),
        ("expert_model_parallel_size", ep),
        ("expert_tensor_parallel_size", etp),
        ("virtual_pipeline_model_parallel_size", None),
    ):
        if hasattr(provider, attr):
            setattr(provider, attr, value)

    return auto_bridge, provider


# ---------------------------------------------------------------------------
# Direct sharded checkpoint writer (mirror of single-rank, with rank-0-only
# tokenizer asset save and rank-aware progress monitor)
# ---------------------------------------------------------------------------


def _save_direct_checkpoint(
    provider: Any,
    path: str,
    model_state: dict[str, Any],
    *,
    model_list: list[Any],
    pg_collection: Any,
    hf_tokenizer_path: Optional[str],
    hf_tokenizer_kwargs: Optional[dict[str, Any]],
) -> None:
    storage_writers_per_rank = int(
        os.environ.get("MEGATRON_BRIDGE_DIRECT_STORAGE_WRITERS_PER_RANK", "16")
    )
    tokenizer_config = None
    if hf_tokenizer_path is not None:
        tokenizer_config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=str(hf_tokenizer_path),
            hf_tokenizer_kwargs=hf_tokenizer_kwargs or {},
        )

    state = GlobalState()
    if hasattr(state, "train_state") and hasattr(state.train_state, "step"):
        state.train_state.step = 0

    state.cfg = ConfigContainer(
        model=provider,
        train=None,
        optimizer=OptimizerConfig(use_distributed_optimizer=False),
        ddp=None,
        scheduler=None,
        dataset=None,
        logger=LoggerConfig(),
        tokenizer=tokenizer_config,
        checkpoint=CheckpointConfig(
            async_save=False,
            async_strategy="mcore",
            save=str(path),
            save_optim=False,
            save_rng=False,
            ckpt_format="torch_dist",
            dist_ckpt_optim_fully_reshardable=True,
            fully_parallel_save=False,
            storage_writers_per_rank=storage_writers_per_rank,
        ),
        dist=None,
    )

    prebuilt_state_dict = {
        "checkpoint_version": 3.0,
        "iteration": 0,
        "model": model_state,
    }

    t0 = time.monotonic()
    progress_monitor = _maybe_create_save_progress_monitor(path)
    print_rank_0("Saving checkpoint...")
    try:
        if progress_monitor is not None:
            progress_monitor.start()
        save_checkpoint(
            state=state,
            model=model_list,
            optimizer=None,
            opt_param_scheduler=None,
            num_floating_point_operations_so_far=0,
            prebuilt_state_dict=prebuilt_state_dict,
            pg_collection=pg_collection,
        )
    finally:
        if progress_monitor is not None:
            progress_monitor.stop()
    print_rank_0(f"Checkpoint saved in {_format_elapsed(time.monotonic() - t0)}")

    # Tokenizer assets are global metadata; only rank 0 writes them.
    if tokenizer_config is not None:
        is_rank_zero = (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )
        if is_rank_zero:
            tokenizer = build_tokenizer(tokenizer_config)
            checkpoint_name = get_checkpoint_name(str(path), 0, release=False)
            save_tokenizer_assets(tokenizer, tokenizer_config, checkpoint_name)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Multi-GPU direct-write HuggingFace NVFP4 -> Megatron checkpoint "
            "conversion. Defaults to TP=1, EP=8 (the Kimi-K2.5 / DeepSeek-V3 "
            "training+rollout layout). Launch with torchrun --nproc_per_node=N."
        ),
    )
    parser.add_argument("--hf-model-path", required=True)
    parser.add_argument("--megatron-path", required=True)
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallelism size (default: 1)",
    )
    parser.add_argument(
        "--pp",
        type=int,
        default=1,
        help="Pipeline parallelism size (default: 1)",
    )
    parser.add_argument(
        "--ep",
        type=int,
        default=8,
        help="Expert parallelism size (default: 8)",
    )
    parser.add_argument(
        "--etp",
        type=int,
        default=1,
        help="Expert tensor parallelism size (default: 1)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Allow custom model code execution (default: True; required for "
        "Kimi-K2.5 / DeepSeek-V3 family).",
    )
    parser.add_argument(
        "--debug-layer-range",
        type=_parse_debug_layer_range,
        help=(
            "Optional half-open decoder layer range START:END for debug-only "
            "partial conversion, e.g. 0:4 converts decoder.layers.0-3 only. "
            "The resulting checkpoint is intended for save-path debugging, "
            "not loading."
        ),
    )
    return parser.parse_args(argv)


def _check_distributed():
    if os.environ.get("WORLD_SIZE") is None:
        print(
            "This script must be launched with torchrun or srun. Example:\n"
            f"  torchrun --nproc_per_node 8 {sys.argv[0]} "
            "--hf-model-path <hf> --megatron-path <out> --tp 1 --ep 8",
            flush=True,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Worker (runs on every torchrun-spawned rank)
# ---------------------------------------------------------------------------


@torchrun_main
def convert_nvfp4_multigpu(
    hf_model_path: str,
    megatron_path: str,
    *,
    tp: int = 1,
    pp: int = 1,
    ep: int = 8,
    etp: int = 1,
    trust_remote_code: bool = True,
    debug_layer_range: Optional[tuple[int, int]] = None,
) -> int:
    """Direct-write HF NVFP4 -> Megatron, sharded across torchrun ranks."""
    _check_distributed()

    print_rank_0(
        f"Multi-GPU NVFP4 conversion: {hf_model_path} -> {megatron_path}"
    )
    print_rank_0(
        f"  TP={tp}  PP={pp}  EP={ep}  ETP={etp}  "
        f"world_size={os.environ.get('WORLD_SIZE')}"
    )

    auto_bridge, provider = _build_multi_rank_meta_provider(
        hf_model_path,
        tp=tp,
        pp=pp,
        ep=ep,
        etp=etp,
        trust_remote_code=trust_remote_code,
    )
    auto_bridge._model_bridge = _select_nvfp4_bridge(auto_bridge)
    if not is_nvfp4_source(auto_bridge.hf_pretrained.config):
        raise ValueError("Source model is not an NVFP4 HuggingFace checkpoint")

    if hasattr(provider, "finalize"):
        provider.finalize()
    # Set up Megatron-side TP/PP/EP/ETP process groups. @torchrun_main has
    # already initialized torch.distributed; this layer creates the model-
    # parallel sub-groups.
    provider.initialize_model_parallel(seed=0)

    patch_meta_init_for_te_modules()

    trust_remote_code_resolved = getattr(
        auto_bridge.hf_pretrained, "trust_remote_code", False
    )
    tokenizer_kwargs = (
        {"trust_remote_code": True} if trust_remote_code_resolved else None
    )

    stage_start = _log_stage_start("Building Megatron meta model...")
    megatron_model = provider.provide_distributed_model(
        wrap_with_ddp=False,
        use_cpu_initialization=True,
        init_model_with_meta_device=True,
        mixed_precision_wrapper=None,
    )
    _log_stage_done("Built Megatron meta model", stage_start)

    stage_start = _log_stage_start("Building conversion tasks...")
    conversion_tasks = auto_bridge._model_bridge.build_conversion_tasks(
        auto_bridge.hf_pretrained,
        megatron_model,
    )
    _log_stage_done(
        "Built conversion tasks",
        stage_start,
        extra=f"{len(conversion_tasks)} tasks",
    )

    if debug_layer_range is not None:
        start, end = debug_layer_range
        stage_start = _log_stage_start(
            f"Filtering conversion tasks for debug decoder layer range "
            f"[{start}, {end})..."
        )
        original_task_count = len(conversion_tasks)
        conversion_tasks = _filter_conversion_tasks_for_debug_layer_range(
            conversion_tasks,
            debug_layer_range,
        )
        if not conversion_tasks:
            raise ValueError(
                f"Debug layer range [{start}, {end}) selected no conversion tasks."
            )
        _log_stage_done(
            "Filtered conversion tasks for debug layer range",
            stage_start,
            extra=f"{len(conversion_tasks)}/{original_task_count} tasks kept",
        )
        print_rank_0(
            "Debug partial-conversion mode is enabled. The output checkpoint "
            "is only meant for conversion/save validation and is not expected "
            "to be loadable."
        )

    stage_start = _log_stage_start("Collecting NVFP4 target modules...")
    module_names = collect_nvfp4_target_module_names(
        conversion_tasks,
        auto_bridge.hf_pretrained.state,
        # Only rank 0 prints the tqdm bar; other ranks compute silently.
        show_progress=(
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        ),
    )
    _log_stage_done(
        "Collected NVFP4 target modules",
        stage_start,
        extra=f"{len(module_names)} modules",
    )

    stage_start = _log_stage_start(
        f"Applying ModelOpt NVFP4 modules to {len(module_names)} target modules..."
    )
    apply_modelopt_nvfp4_to_meta_model(
        megatron_model[0],
        module_names=module_names,
    )
    _log_stage_done(
        "Applied ModelOpt NVFP4 modules",
        stage_start,
        extra=f"{len(module_names)} modules",
    )

    pg_collection = get_pg_collection(megatron_model)
    stage_start = _log_stage_start("Building sharded state dict template...")
    model_template = megatron_model[0].sharded_state_dict(
        metadata={"dp_cp_group": pg_collection.dp_cp}
    )
    _log_stage_done("Built sharded state dict template", stage_start)

    checkpoint_parent = Path(megatron_path).resolve().parent
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        checkpoint_parent.mkdir(parents=True, exist_ok=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    spill_parent = Path(
        os.environ.get("MEGATRON_BRIDGE_DIRECT_SPILL_DIR", str(checkpoint_parent))
    ).resolve()
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        spill_parent.mkdir(parents=True, exist_ok=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Per-rank spill directory: each rank writes its own intermediate tensors
    # under a unique tempdir. TemporaryDirectory cleans up on exit.
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    spill_prefix = (
        f".{Path(megatron_path).name}.nvfp4_spill_rank{rank:04d}_"
    )
    with tempfile.TemporaryDirectory(
        prefix=spill_prefix,
        dir=spill_parent,
    ) as spill_dir:
        spill_manager = TensorSpillManager(spill_dir)
        print_rank_0(
            f"Using spill directory for direct tensors: {spill_parent} "
            f"(per-rank subdirs prefixed with {spill_prefix!r})"
        )

        stage_start = _log_stage_start("Preparing direct NVFP4 model state dict...")
        model_state = build_nvfp4_direct_model_state_dict(
            auto_bridge._model_bridge,
            auto_bridge.hf_pretrained,
            megatron_model,
            model_template,
            conversion_tasks=conversion_tasks,
            spill_manager=spill_manager,
        )
        _log_stage_done("Prepared direct NVFP4 model state dict", stage_start)

        _save_direct_checkpoint(
            provider,
            megatron_path,
            model_state,
            model_list=megatron_model,
            pg_collection=pg_collection,
            hf_tokenizer_path=hf_model_path,
            hf_tokenizer_kwargs=tokenizer_kwargs,
        )

    print_rank_0(
        f"Done. Direct NVFP4 Megatron checkpoint saved to: {megatron_path}"
    )
    return 0


def main():
    args = parse_args()
    convert_nvfp4_multigpu(
        hf_model_path=args.hf_model_path,
        megatron_path=args.megatron_path,
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        etp=args.etp,
        trust_remote_code=args.trust_remote_code,
        debug_layer_range=args.debug_layer_range,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
