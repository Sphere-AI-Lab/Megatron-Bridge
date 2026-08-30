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

"""Direct-write HF DeepSeek-style FP4/FP8 -> Megatron checkpoint conversion."""

# ruff: noqa: D101, D103  # operational scripts: helpers here are entrypoint plumbing, not API

from __future__ import annotations

import argparse
import importlib
import importlib.abc
import importlib.metadata
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional


def _patch_missing_transformer_engine_metadata() -> None:
    """Handle local package installs whose dist-info directory lacks METADATA."""
    original_version = importlib.metadata.version

    def _version(distribution_name: str) -> str:
        normalized = distribution_name.replace("_", "-").lower()
        try:
            return original_version(distribution_name)
        except KeyError:
            normalized_underscore = normalized.replace("-", "_")
            for entry in sys.path:
                site_path = Path(entry)
                if not site_path.exists():
                    continue
                for dist_info in site_path.glob("*.dist-info"):
                    stem = dist_info.name.removesuffix(".dist-info")
                    if "-" not in stem:
                        continue
                    name_part, version = stem.rsplit("-", 1)
                    dist_normalized = name_part.replace("_", "-").lower()
                    dist_underscore = dist_normalized.replace("-", "_")
                    if normalized in {dist_normalized, dist_underscore, normalized_underscore} and version:
                        return version
            raise

    importlib.metadata.version = _version


_patch_missing_transformer_engine_metadata()


class _TransformerEngineImportBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object | None, target: object | None = None):
        if fullname == "transformer_engine" or fullname.startswith("transformer_engine."):
            raise ModuleNotFoundError("Incomplete transformer_engine install disabled for direct conversion")
        return None


def _disable_incomplete_transformer_engine() -> None:
    """Force Megatron's optional TE paths off when only an empty namespace package exists."""
    try:
        tensor_mod = importlib.import_module("transformer_engine.pytorch.tensor")
        if hasattr(tensor_mod, "QuantizedTensor"):
            return None
    except Exception:
        pass

    for module_name in list(sys.modules):
        if module_name == "transformer_engine" or module_name.startswith("transformer_engine."):
            sys.modules.pop(module_name, None)

    for finder in sys.meta_path:
        if isinstance(finder, _TransformerEngineImportBlocker):
            return finder

    blocker = _TransformerEngineImportBlocker()
    sys.meta_path.insert(0, blocker)
    return blocker


_TE_IMPORT_BLOCKER = _disable_incomplete_transformer_engine()

from megatron.core.optimizer import OptimizerConfig


def _install_transformer_engine_stub() -> None:
    """Install a minimal inert TE module for optional Bridge PEFT imports."""
    if _TE_IMPORT_BLOCKER is not None and _TE_IMPORT_BLOCKER in sys.meta_path:
        sys.meta_path.remove(_TE_IMPORT_BLOCKER)

    if "transformer_engine.pytorch" in sys.modules:
        return

    from contextlib import nullcontext

    import torch.nn as nn

    te_mod = types.ModuleType("transformer_engine")
    pytorch_mod = types.ModuleType("transformer_engine.pytorch")
    ops_mod = types.ModuleType("transformer_engine.pytorch.ops")
    fp8_mod = types.ModuleType("transformer_engine.pytorch.fp8")
    tensor_mod = types.ModuleType("transformer_engine.pytorch.tensor")
    tensor_utils_mod = types.ModuleType("transformer_engine.pytorch.tensor.utils")
    mxfp8_mod = types.ModuleType("transformer_engine.pytorch.tensor.mxfp8_tensor")
    common_mod = types.ModuleType("transformer_engine.common")
    recipe_mod = types.ModuleType("transformer_engine.common.recipe")

    class _StubModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, *args, **kwargs):
            raise RuntimeError("Transformer Engine stub cannot execute forward passes.")

    class _QuantizedTensor:
        pass

    class _FP8GlobalStateManager:
        pass

    class _Format:
        E4M3 = "E4M3"
        HYBRID = "HYBRID"

    class _Recipe:
        def __init__(self, *args, **kwargs):
            pass

    pytorch_mod.Linear = _StubModule
    pytorch_mod.LayerNormLinear = _StubModule
    pytorch_mod.DotProductAttention = _StubModule
    pytorch_mod.GroupedLinear = _StubModule
    pytorch_mod.LayerNorm = _StubModule
    pytorch_mod.RMSNorm = _StubModule
    pytorch_mod.fp8_autocast = lambda *args, **kwargs: nullcontext()
    pytorch_mod.ops = ops_mod

    for name in (
        "Sequential",
        "LayerNorm",
        "RMSNorm",
        "Quantize",
        "MakeExtraOutput",
        "Linear",
        "Dropout",
        "ConstantScale",
        "AddExtraInput",
        "BasicLinear",
        "Bias",
        "ReduceScatter",
        "AllReduce",
        "SwiGLU",
        "GEGLU",
        "ReGLU",
        "GELU",
        "ReLU",
        "SiLU",
        "FusibleOperation",
    ):
        setattr(ops_mod, name, _StubModule)

    fp8_mod.FP8GlobalStateManager = _FP8GlobalStateManager
    fp8_mod.fp8_autocast = lambda *args, **kwargs: nullcontext()
    tensor_mod.QuantizedTensor = _QuantizedTensor
    mxfp8_mod.MXFP8Tensor = _QuantizedTensor
    tensor_utils_mod.post_all_gather_processing = None

    recipe_mod.Format = _Format
    recipe_mod.DelayedScaling = _Recipe
    recipe_mod.Float8CurrentScaling = _Recipe
    recipe_mod.Float8BlockScaling = _Recipe
    recipe_mod.MXFP8BlockScaling = _Recipe
    recipe_mod.NVFP4BlockScaling = _Recipe
    recipe_mod.CustomRecipe = _Recipe
    common_mod.recipe = recipe_mod

    te_mod.pytorch = pytorch_mod
    te_mod.common = common_mod

    sys.modules["transformer_engine"] = te_mod
    sys.modules["transformer_engine.pytorch"] = pytorch_mod
    sys.modules["transformer_engine.pytorch.ops"] = ops_mod
    sys.modules["transformer_engine.pytorch.fp8"] = fp8_mod
    sys.modules["transformer_engine.pytorch.tensor"] = tensor_mod
    sys.modules["transformer_engine.pytorch.tensor.utils"] = tensor_utils_mod
    sys.modules["transformer_engine.pytorch.tensor.mxfp8_tensor"] = mxfp8_mod
    sys.modules["transformer_engine.common"] = common_mod
    sys.modules["transformer_engine.common.recipe"] = recipe_mod


_install_transformer_engine_stub()

from megatron.bridge.orbit.low_precision.common import (
    build_single_rank_meta_provider,
    patch_meta_init_for_te_modules,
)
from megatron.bridge.orbit.low_precision.fp4 import build_fp4_direct_model_state_dict
from megatron.bridge.training.checkpointing import (
    get_checkpoint_name,
    save_checkpoint,
    save_tokenizer_assets,
)
from megatron.bridge.training.config import CheckpointConfig, ConfigContainer, LoggerConfig
from megatron.bridge.training.model_load_save import temporary_distributed_context
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.training.utils.pg_utils import get_pg_collection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct-write HF DeepSeek-style FP4/FP8 -> Megatron checkpoint conversion",
    )
    parser.add_argument("--hf-model-path", required=True, help="Path to HF DeepSeek-style FP4/FP8 model directory")
    parser.add_argument("--megatron-path", required=True, help="Output Megatron checkpoint directory")
    return parser.parse_args()


@contextmanager
def keep_meta_model_unmaterialized():
    """Keep direct-conversion template models on meta instead of CUDA."""

    import megatron.bridge.models.common.unimodal as unimodal_common
    import megatron.bridge.models.model_provider as model_provider

    original_unimodal_to_empty = unimodal_common.to_empty_if_meta_device
    original_provider_to_empty = getattr(model_provider, "to_empty_if_meta_device", None)

    def _identity_to_empty(module, *, device, recurse=True):
        return module

    unimodal_common.to_empty_if_meta_device = _identity_to_empty
    if original_provider_to_empty is not None:
        model_provider.to_empty_if_meta_device = _identity_to_empty

    try:
        yield
    finally:
        unimodal_common.to_empty_if_meta_device = original_unimodal_to_empty
        if original_provider_to_empty is not None:
            model_provider.to_empty_if_meta_device = original_provider_to_empty


def _save_direct_checkpoint(
    provider: Any,
    path: str,
    model_state: dict[str, Any],
    *,
    pg_collection: Any,
    hf_tokenizer_path: Optional[str],
    hf_tokenizer_kwargs: Optional[dict[str, Any]],
) -> None:
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
            storage_writers_per_rank=16,
        ),
        dist=None,
    )

    prebuilt_state_dict = {
        "checkpoint_version": 3.0,
        "iteration": 0,
        "model": model_state,
    }

    t0 = time.monotonic()
    print("Saving checkpoint...", flush=True)
    from megatron.core.dist_checkpointing.strategies import torch as torch_dist_strategy

    original_have_nvrx = torch_dist_strategy.HAVE_NVRX
    torch_dist_strategy.HAVE_NVRX = False
    try:
        save_checkpoint(
            state=state,
            model=[],
            optimizer=None,
            opt_param_scheduler=None,
            num_floating_point_operations_so_far=0,
            prebuilt_state_dict=prebuilt_state_dict,
            pg_collection=pg_collection,
        )
    finally:
        torch_dist_strategy.HAVE_NVRX = original_have_nvrx
    print(f"Checkpoint saved in {time.strftime('%H:%M:%S', time.gmtime(time.monotonic() - t0))}", flush=True)

    if tokenizer_config is not None:
        tokenizer = build_tokenizer(tokenizer_config)
        checkpoint_name = get_checkpoint_name(str(path), 0, release=False)
        save_tokenizer_assets(tokenizer, tokenizer_config, checkpoint_name)


def main() -> int:
    args = parse_args()

    print(f"Converting FP4/FP8 checkpoint directly: {args.hf_model_path} -> {args.megatron_path}")
    print("Using single-process meta-model conversion (TP=PP=EP=1 checkpoint write)")

    auto_bridge, provider = build_single_rank_meta_provider(
        args.hf_model_path,
        trust_remote_code=True,
    )
    bridge = auto_bridge._model_bridge

    def _direct_local_block_spec(config: Any, vp_stage: int | None = None):
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
        from megatron.core.transformer.torch_norm import WrappedTorchNorm

        block_spec = get_gpt_decoder_block_spec(
            config,
            use_transformer_engine=False,
            normalization=config.normalization,
            vp_stage=vp_stage,
        )
        if config.normalization == "RMSNorm":
            block_spec.layer_norm = WrappedTorchNorm
        return block_spec

    try:
        provider.transformer_layer_spec = _direct_local_block_spec
        provider.use_transformer_engine_full_layer_spec = False
        provider.restore_modelopt_state = False
        print("Using local GPT block spec for direct meta-model construction", flush=True)
    except Exception as exc:
        print(f"Could not force local GPT layer spec; continuing with provider default: {exc}", flush=True)

    if hasattr(provider, "finalize"):
        provider.finalize()

    trust_remote_code = getattr(auto_bridge.hf_pretrained, "trust_remote_code", False)
    tokenizer_kwargs = {"trust_remote_code": True} if trust_remote_code else None

    patch_meta_init_for_te_modules()

    with temporary_distributed_context(backend="gloo"):
        with keep_meta_model_unmaterialized():
            meta_model = provider.provide_distributed_model(
                wrap_with_ddp=False,
                use_cpu_initialization=True,
                init_model_with_meta_device=True,
                mixed_precision_wrapper=None,
            )

        pg_collection = get_pg_collection(meta_model)
        model_template = meta_model[0].sharded_state_dict(metadata={"dp_cp_group": pg_collection.dp_cp})
        model_state = build_fp4_direct_model_state_dict(
            bridge,
            auto_bridge.hf_pretrained,
            meta_model,
            model_template,
        )

        del meta_model
        del model_template

        _save_direct_checkpoint(
            provider,
            args.megatron_path,
            model_state,
            pg_collection=pg_collection,
            hf_tokenizer_path=args.hf_model_path,
            hf_tokenizer_kwargs=tokenizer_kwargs,
        )

    print(f"Done. Direct FP4/FP8 Megatron checkpoint saved to: {args.megatron_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
