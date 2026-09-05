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

"""Common helpers for low-precision direct checkpoint conversion."""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory


if TYPE_CHECKING:
    from megatron.bridge.models.conversion.auto_bridge import AutoBridge
    from megatron.bridge.models.gpt_provider import GPTModelProvider


__all__ = [
    "add_tensor_entry",
    "atomic_direct_checkpoint_directory",
    "build_single_rank_meta_provider",
    "patch_meta_init_for_te_modules",
    "prepare_empty_model_state",
    "retain_non_tensor_entries",
    "TensorSpillManager",
]


_DIRECT_CHECKPOINT_ARTIFACT = Path("iter_0000000") / "metadata.json"


def _validate_single_rank_direct_conversion_tasks(
    conversion_tasks: Sequence[Any],
    *,
    format_name: str,
) -> None:
    """Require a complete, local task plan for a single-rank direct conversion."""
    for task_index, task in enumerate(conversion_tasks):
        if task is None:
            raise RuntimeError(
                f"Direct {format_name} conversion task index {task_index} is missing; "
                "single-rank direct conversion requires every task to have a local Megatron destination"
            )

        if getattr(task, "megatron_module", None) is None:
            parameter_name = getattr(task, "global_param_name", None) or getattr(task, "param_name", None)
            parameter_detail = f" for parameter {parameter_name!r}" if parameter_name is not None else ""
            raise RuntimeError(
                f"Direct {format_name} conversion task index {task_index}{parameter_detail} "
                "has no local Megatron destination; single-rank direct conversion requires a complete local task plan"
            )


@contextmanager
def atomic_direct_checkpoint_directory(destination: str | Path) -> Iterator[Path]:
    """Stage a direct checkpoint beside its destination and publish it atomically."""
    destination = Path(destination)
    if destination.is_symlink():
        raise FileExistsError(f"Direct checkpoint destination is a symlink: {destination}")
    if destination.exists():
        raise FileExistsError(f"Direct checkpoint destination already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    mode_probe = staging / ".publish-mode"
    mode_probe.mkdir()
    publish_mode = stat.S_IMODE(mode_probe.stat().st_mode)
    mode_probe.rmdir()
    try:
        yield staging

        expected_artifact = staging / _DIRECT_CHECKPOINT_ARTIFACT
        if not expected_artifact.is_file():
            raise RuntimeError(
                "Direct checkpoint is incomplete: expected "
                f"{_DIRECT_CHECKPOINT_ARTIFACT.as_posix()} before publication"
            )

        if destination.is_symlink() or destination.exists():
            raise FileExistsError(
                f"Direct checkpoint destination appeared before checkpoint publication: {destination}"
            )
        staging.chmod(publish_mode)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


class TensorSpillManager:
    """Spill CPU tensors into file-backed storages to cap direct-save RAM usage."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0
        self._paths: list[Path] = []

    def spill_tensor(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach().cpu().contiguous()
        if tensor.numel() == 0:
            return tensor

        spill_path = self.root_dir / f"{self._counter:08d}_{self._sanitize_key(key)}.bin"
        self._counter += 1
        self._paths.append(spill_path)

        spilled = torch.from_file(
            str(spill_path),
            shared=True,
            size=tensor.numel(),
            dtype=tensor.dtype,
        ).view(tensor.shape)
        spilled.copy_(tensor)
        self._flush_spill_file(spill_path)
        return spilled

    def cleanup(self) -> None:
        for path in self._paths:
            path.unlink(missing_ok=True)
        self._paths.clear()

    @staticmethod
    def _sanitize_key(key: str) -> str:
        sanitized = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in key)
        return sanitized[-160:] or "tensor"

    @staticmethod
    def _flush_spill_file(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
            posix_fadvise = getattr(os, "posix_fadvise", None)
            dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
            if posix_fadvise is not None and dontneed is not None:
                posix_fadvise(fd, 0, 0, dontneed)
        finally:
            os.close(fd)


def retain_non_tensor_entries(node: Any) -> Any:
    """Drop sharded tensors while preserving nested checkpoint metadata."""
    if isinstance(node, dict):
        kept = {}
        for key, value in node.items():
            retained = retain_non_tensor_entries(value)
            if retained is None:
                continue
            if isinstance(retained, (dict, list)) and not retained:
                continue
            kept[key] = retained
        return kept

    if isinstance(node, list):
        kept = []
        for value in node:
            retained = retain_non_tensor_entries(value)
            if retained is None:
                continue
            if isinstance(retained, (dict, list)) and not retained:
                continue
            kept.append(retained)
        return kept

    if isinstance(node, (ShardedTensor, ShardedTensorFactory)):
        return None

    return node


def add_tensor_entry(
    model_state: dict[str, Any],
    key: str,
    tensor: torch.Tensor,
    template_entry: Any | None = None,
    spill_manager: TensorSpillManager | None = None,
) -> None:
    """Insert a CPU, contiguous tensor into a Megatron sharded state dict."""
    if tensor.device.type == "meta":
        raise RuntimeError(f"Cannot save meta tensor for {key}")

    if spill_manager is not None:
        tensor = spill_manager.spill_tensor(key, tensor)
    else:
        tensor = tensor.detach().cpu().contiguous()
    if isinstance(template_entry, ShardedTensorFactory):
        model_state[key] = replace(template_entry, key=key, data=tensor)
        return

    if isinstance(template_entry, ShardedTensor):
        model_state[key] = replace(template_entry, key=key, data=tensor, dtype=tensor.dtype)
        return

    model_state[key] = ShardedTensor.from_rank_offsets(key, tensor)


def build_single_rank_meta_provider(
    hf_model_path: str,
    *,
    trust_remote_code: bool = False,
) -> tuple[AutoBridge, GPTModelProvider]:
    """Build a single-rank provider without loading weights for direct conversion."""
    from megatron.bridge.models.conversion.auto_bridge import AutoBridge

    auto_bridge = AutoBridge.from_hf_pretrained(
        hf_model_path,
        trust_remote_code=trust_remote_code,
    )
    provider = auto_bridge.to_megatron_provider(load_weights=False)
    provider.perform_initialization = False

    for attr, value in (
        ("tensor_model_parallel_size", 1),
        ("pipeline_model_parallel_size", 1),
        ("context_parallel_size", 1),
        ("expert_model_parallel_size", 1),
        ("expert_tensor_parallel_size", 1),
        ("virtual_pipeline_model_parallel_size", None),
    ):
        if hasattr(provider, attr):
            setattr(provider, attr, value)

    # Conversion meta models never route tokens, but the Kimi / DeepSeek-v4
    # bridges default to the flex dispatcher, whose _HybridEPManager imports
    # the optional HybridEP package at layer-build time. Downgrade to the
    # plain alltoall dispatcher so direct conversion does not depend on it.
    if getattr(provider, "moe_token_dispatcher_type", None) == "flex":
        provider.moe_token_dispatcher_type = "alltoall"

    return auto_bridge, provider


def prepare_empty_model_state(model_template: dict[str, Any]) -> dict[str, Any]:
    """Keep only non-tensor checkpoint metadata in an empty direct-save skeleton."""
    return retain_non_tensor_entries(model_template) or {}


def patch_meta_init_for_te_modules() -> None:
    """Skip TE affine init copies when building on the meta device."""
    import megatron.core.extensions.transformer_engine as _te_ext
    import megatron.core.tensor_parallel.layers as _tp_layers

    if getattr(_tp_layers._initialize_affine_weight_cpu, "__name__", "") == "_patched_init_affine":
        return

    _orig_init_affine = _tp_layers._initialize_affine_weight_cpu

    def _patched_init_affine(weight, *args, **kwargs):
        try:
            return _orig_init_affine(weight, *args, **kwargs)
        except NotImplementedError:
            weight_device = getattr(getattr(weight, "device", None), "type", None)
            default_device = torch.device(torch.get_default_device()).type
            if weight_device == "meta" or default_device == "meta":
                return None
            raise

    _tp_layers._initialize_affine_weight_cpu = _patched_init_affine
    _te_ext._initialize_affine_weight_cpu = _patched_init_affine
