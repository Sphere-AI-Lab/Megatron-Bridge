# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""ModelOpt checkpoint save/restore extensions (orbit fork).

Extracted from ``megatron.bridge.training.checkpointing``; the shared module
keeps two marked call-site seams. See ``docs/orbit/UPSTREAM_SEAMS.md``.
"""

import os
from pathlib import Path
from typing import Any, Optional

import torch
from megatron.core import dist_checkpointing
from megatron.core.transformer import MegatronModule
from modelopt.torch.opt.plugins import restore_modelopt_state, save_sharded_modelopt_state

from megatron.bridge.utils.common_utils import print_rank_0


def _save_sharded_modelopt_state_with_async_strategy(
    model: list[torch.nn.Module],
    checkpoint_name: str | Path,
    sharded_strategy: tuple[str, int] | None = None,
    async_strategy: str = "nvrx",
) -> None:
    """Save sharded ModelOpt state while honoring the configured async backend."""

    if async_strategy == "nvrx":
        save_sharded_modelopt_state(model, checkpoint_name, sharded_strategy)
        return

    import copy

    import modelopt
    import modelopt.torch.opt as mto
    import modelopt.torch.utils.distributed as modelopt_dist
    import yaml
    from modelopt.torch.opt.plugins import mcore_dist_checkpointing as modelopt_mcore_dcp

    drop_substrings = getattr(modelopt_mcore_dcp, "DROP_SUBSTRINGS", ())
    remove_per_module_state = modelopt_mcore_dcp.remove_per_module_state

    def _parse_transformer_config(transformer_config: dict) -> dict:
        config = {}
        for k, v in transformer_config.items():
            if any(substring in k for substring in drop_substrings):
                continue
            if isinstance(v, (bool, int, str)):
                config[k] = v
            else:
                config[k] = str(v)
        return config

    if modelopt_dist.is_master() and not os.path.exists(f"{checkpoint_name}/run_config.yaml"):
        run_config_name = f"{checkpoint_name}/modelopt_run_config.yaml"
        config_dict = _parse_transformer_config(model[0].config.__dict__)
        config_dict["nvidia_modelopt_version"] = modelopt.__version__
        with open(run_config_name, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    if not mto.ModeloptStateManager.is_converted(model[0]):
        return
    if len(model) > 1:
        raise ValueError("sharded_modelopt_state does not support virtual pipeline parallel!")

    modelopt_checkpoint_name = f"{checkpoint_name}/modelopt_state"
    if modelopt_dist.is_master():
        os.makedirs(modelopt_checkpoint_name, exist_ok=True)

    modelopt_state = copy.deepcopy(mto.modelopt_state(model[0]))
    remove_per_module_state(modelopt_state)
    dist_checkpointing.save(
        modelopt_state,
        modelopt_checkpoint_name,
        sharded_strategy,
        async_strategy=async_strategy,
    )


def restore_sharded_modelopt_state_via_common_reader(
    model: list[MegatronModule],
    checkpoint_name: str | Path,
    prefix: str = "",
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Restore sharded ModelOpt state, reading common state with MCore's reader.

    Mirrors the pinned ModelOpt ``restore_sharded_modelopt_state`` except for one
    line: the common modelopt state is loaded through
    ``dist_checkpointing.load_common_state_dict()`` instead of a direct
    ``common.pt`` read. The pinned MCore's ``dist_checkpointing.save()`` embeds
    common data as a ``ShardedObject("common_state")`` inside the torch-dist
    files and never writes ``common.pt``, so ModelOpt's own restore raises
    FileNotFoundError on every modelopt_state this repo saves. MCore's reader
    understands that layout (and reads rank-locally: torch-dist load with
    ``no_dist=True``).
    """
    import modelopt.torch.opt as mto

    # Private but pinned: the per-module extra_state loader is the second phase
    # of ModelOpt's own two-phase restore and has no public equivalent.
    from modelopt.torch.opt.plugins.mcore_dist_checkpointing import (
        _load_extra_state_from_sharded_checkpoint,
    )

    if len(model) > 1:
        raise ValueError("sharded_modelopt_state does not support virtual pipeline parallel!")

    modelopt_checkpoint_name = f"{checkpoint_name}/modelopt_state"
    if not os.path.exists(modelopt_checkpoint_name) or mto.ModeloptStateManager.is_converted(model[0]):
        return

    common_modelopt_state = dist_checkpointing.load_common_state_dict(modelopt_checkpoint_name)
    print_rank_0(f"nvidia-modelopt ckpt version: {common_modelopt_state.get('modelopt_version')}")

    model[0] = mto.restore_from_modelopt_state(model[0], common_modelopt_state)
    _load_extra_state_from_sharded_checkpoint(model[0], checkpoint_name, prefix, metadata=metadata)


def _maybe_restore_modelopt_state_for_sharded_load(
    model: list[MegatronModule],
    checkpoint_path: Optional[str],
    common_state_dict: Optional[dict[str, Any]],
) -> bool:
    """Restore ModelOpt state before generating a sharded load schema."""

    if checkpoint_path is None or common_state_dict is None:
        return False

    from megatron.bridge.training.post_training.checkpointing import has_modelopt_state

    if not has_modelopt_state(checkpoint_path):
        return False

    # ModelOpt's restore replays the saved mode list, which contains
    # ``real_quantize`` once a prior run has been through ``mtq.compress``.
    # Replaying that mode walks grouped expert linears (``weight0..weightN``,
    # no ``.weight``) through unpatched ModelOpt code, so the guards must be in
    # place before the replay -- not only in the packed-restore path.
    from megatron.bridge.orbit.training.modelopt_packed_restore import _patch_modelopt_pack_for_grouped_moe

    _patch_modelopt_pack_for_grouped_moe()
    restore_modelopt_state(model, common_state_dict)
    return True
