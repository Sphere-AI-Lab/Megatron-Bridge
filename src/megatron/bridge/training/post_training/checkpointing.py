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

"""Input/output checkpointing for ModelOpt."""

try:
    from modelopt.torch.opt.plugins import restore_sharded_modelopt_state  # noqa: F401  # modelopt install canary
except ImportError as e:
    raise ImportError('Required `"nvidia-modelopt[torch]"` is not installed!') from e

import os

from megatron.core import dist_checkpointing
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import unwrap_model


def _get_modelopt_checkpoint_path(checkpoint_path: str) -> str:
    """Get the path to use for ModelOpt operations (handles iteration directories)."""
    if not checkpoint_path or not os.path.isdir(checkpoint_path):
        return checkpoint_path

    # Check for iter_* folders
    try:
        iter_folders = [
            f
            for f in os.listdir(checkpoint_path)
            if os.path.isdir(os.path.join(checkpoint_path, f)) and f.startswith("iter_")
        ]
    except (OSError, FileNotFoundError):
        # Directory doesn't exist or can't be accessed
        return checkpoint_path

    if iter_folders:
        # Find the folder with the largest iteration number from state dict
        latest_iter_num = -1
        latest_iter_folder = None

        for folder in iter_folders:
            folder_path = os.path.join(checkpoint_path, folder)
            try:
                state_dict = dist_checkpointing.load_common_state_dict(folder_path)
                if state_dict is not None:
                    iter_num = state_dict.get("iteration", 0)
                    if iter_num > latest_iter_num:
                        latest_iter_num = iter_num
                        latest_iter_folder = folder
            except Exception:
                # Skip checkpoints that fail to load
                continue

        if latest_iter_folder is not None:
            return os.path.join(checkpoint_path, latest_iter_folder)

    return checkpoint_path  # No iteration dirs, use root


def has_modelopt_state(checkpoint_path: str) -> bool:
    """Check if ModelOpt state exists inside the checkpoint path.

    Checks for modelopt_state in iteration directories (iter_*) or root directory.
    NOTE: Ignores distillation state which is deprecated and unused.

    Args:
        checkpoint_path: Path to the checkpoint directory

    Returns:
        True if modelopt_state folder exists and contains nontrivial state, else False.
    """
    modelopt_checkpoint_path = _get_modelopt_checkpoint_path(checkpoint_path)
    modelopt_state_path = os.path.join(modelopt_checkpoint_path, "modelopt_state")
    if not os.path.isdir(modelopt_state_path):
        return False

    # orbit-seam(modelopt): the pinned MCore's dist_checkpointing.save() embeds
    # common data as a ShardedObject("common_state") inside the torch-dist files
    # and writes no common.pt, so a direct torch.load of that file name raised
    # FileNotFoundError for every checkpoint this repo saves. MCore's own reader
    # handles the layout and reads rank-locally (no process group needed).
    modelopt_state = dist_checkpointing.load_common_state_dict(modelopt_state_path)
    modes = modelopt_state["modelopt_state_dict"]
    if len(modes) == 1 and modes[0][0] == "kd_loss":
        # Ignore KD state
        modes.pop()

    return len(modes) > 0


def load_modelopt_state(model: list[MegatronModule], checkpoint_path: str) -> None:
    """Load modelopt_state from a checkpoint.
    Args:
        model: The model to load the modelopt_state into
        checkpoint_path: Path to the checkpoint directory
    """
    # orbit-seam(modelopt): install the grouped-MoE ``.weight`` guards BEFORE
    # the restore below, not after it. The restore replays the checkpoint's
    # saved mode list, and a run checkpoint written after ``mtq.compress``
    # contains ``real_quantize``; replaying that mode in a fresh process walks
    # grouped expert linears (``weight0..weightN``, no ``.weight``) through
    # unpatched ModelOpt and raises AttributeError before the patch at the end
    # of this function would ever have been installed.
    from megatron.bridge.orbit.training.modelopt_packed_restore import (
        _maybe_compress_restored_modelopt_model,
        _patch_modelopt_pack_for_grouped_moe,
    )

    _patch_modelopt_pack_for_grouped_moe()

    modelopt_checkpoint_path = _get_modelopt_checkpoint_path(checkpoint_path)
    unwrapped_model = unwrap_model(model)

    # orbit-seam(modelopt): restore through the orbit reader -- the pinned MCore
    # writes no common.pt, which ModelOpt's own restore reads by literal name.
    from megatron.bridge.orbit.training.modelopt_checkpoint import (
        restore_sharded_modelopt_state_via_common_reader,
    )

    restore_sharded_modelopt_state_via_common_reader(unwrapped_model, modelopt_checkpoint_path)

    # orbit-seam(modelopt): compress packed low-precision checkpoints once
    # ModelOpt state is restored.
    _maybe_compress_restored_modelopt_model(unwrapped_model, modelopt_checkpoint_path)
