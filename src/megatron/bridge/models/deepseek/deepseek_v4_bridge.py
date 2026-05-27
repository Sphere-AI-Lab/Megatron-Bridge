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

"""Megatron Bridge support for native DeepSeek V4 checkpoints."""

import math
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)

try:
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_experimental_attention_variant_transformer_block_cls as _mcore_get_block_cls,
    )
except ImportError:
    _mcore_get_block_cls = None

from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    ColumnParallelMapping,
    QuantScaleMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.bridge.peft.oft import OFT
from megatron.bridge.peft.oft_layers import OFTRotationModule
from megatron.bridge.peft.utils import is_expert_linear, wildcard_match


_DSV4_MOE_OFT_PROJECTIONS = ("w1", "w2", "w3")


def _get_experimental_attention_variant_transformer_block_cls(provider):
    if _mcore_get_block_cls is not None:
        return _mcore_get_block_cls(provider)

    if getattr(provider, "experimental_attention_variant", None) != "dsv4":
        return None

    try:
        from megatron.core.transformer.experimental_attention_variant.deepseek_v4 import (
            DeepSeekV4TransformerBlock,
        )
    except ImportError as exc:
        raise RuntimeError(
            "DeepSeek V4 requires a Megatron-LM submodule with "
            "DeepSeekV4TransformerBlock support. Update 3rdparty/Megatron-LM "
            "before constructing DeepSeek V4 models."
        ) from exc

    return DeepSeekV4TransformerBlock


def _compose_module_name(name: Optional[str] = None, prefix: Optional[str] = None) -> str:
    if prefix and name:
        return f"{prefix}.{name}"
    if prefix:
        return prefix
    return name or ""


def _dsv4_native_expert_projection(full_name: str) -> Optional[str]:
    if ".shared_experts." in full_name:
        return None
    if ".experts." not in full_name:
        return None
    leaf = full_name.rsplit(".", 1)[-1]
    if leaf in _DSV4_MOE_OFT_PROJECTIONS:
        return leaf
    return None


def _dsv4_moe_oft_target_projections(target_modules: list[str], moe_name: str) -> list[str]:
    projections = []
    for proj in _DSV4_MOE_OFT_PROJECTIONS:
        sample = f"{moe_name}.experts.0.{proj}" if moe_name else f"experts.0.{proj}"
        for pattern in target_modules or []:
            if pattern == proj or wildcard_match(pattern, sample):
                projections.append(proj)
                break
            if ".experts." not in pattern or not pattern.endswith(f".{proj}"):
                continue
            parent_pattern = pattern.split(".experts.", 1)[0]
            if wildcard_match(parent_pattern, moe_name):
                projections.append(proj)
                break
    return projections


class DSV4OFTLinear(nn.Module):
    """OFT wrapper that defers to ``DSV4Linear``'s native quantized forward.

    Bridge's stock ``OFTLinear._forward_fp8`` dequantizes via a Bridge-style
    ``weight_scale_inv`` buffer (block-128 scales). DSV4Linear stores its
    block scales on a separate ``self.scale`` Parameter and dispatches via
    ``_quantized_linear`` based on ``weight.dtype`` (fp4 / fp8 / fp32 /
    bf16). Routing through Bridge's fp8 path would mis-dequant DSV4 weights.

    This wrapper sidesteps the issue by computing
    ``y = linear(adapter(x))`` and letting DSV4Linear handle quant
    natively. Forward signature matches DSV4Linear (single Tensor in,
    single Tensor out), unlike the stock OFTLinear which returns
    ``(out, bias)`` tuples for TE compat.

    Attribute proxy: V4 attention's ``forward`` reaches into
    ``self.wo_a.weight`` directly for a manual einsum reshape, bypassing
    ``self.wo_a(x)``. ``__getattr__`` forwards unknown attribute access to
    ``self.to_wrap`` so that ``DSV4OFTLinear.weight`` /
    ``DSV4OFTLinear.scale`` continue to resolve. Under identity-OFT this is
    correctness-safe (rotation = I, so ``R · W == W``). Under non-identity
    training the einsum bypass would silently skip ``wo_a``'s rotation
    unless the attention forward calls ``self.wo_a.apply_input_rotation``
    (or ``self.wo_a(x)``) before reading ``wo_a.weight``.
    """

    def __init__(self, to_wrap: nn.Module, adapter: OFTRotationModule):
        super().__init__()
        self.to_wrap = to_wrap
        self.adapter = adapter
        self._adapter_enabled = True

    @property
    def linear(self) -> nn.Module:
        return self.to_wrap

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.to_wrap, name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._adapter_enabled:
            x = self.adapter(x.contiguous())
        return self.to_wrap(x)

    def apply_input_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply only the OFT rotation to ``x``, returning the rotated
        input *without* running the base linear.

        Used by V4 attention's einsum-bypass code path
        (deepseek_v4.py:1048) for ``wo_a``. The default ``DSV4Linear``
        impl is identity; this OFT-aware override applies the adapter
        so ``out = einsum(rotated_out, wo_a.weight)`` is mathematically
        equivalent to ``out = wo_a(rotated_out)`` with the rotation
        applied to the *input* (consistent with the input-rotation
        pattern in the other 4 OFT-wrapped sublayers).

        wo_a is a *grouped* linear: ``wo_a.weight`` views as
        ``(n_local_groups, o_lora_rank, head_local_dim_per_group)`` and
        the einsum at deepseek_v4.py:1058 contracts only the
        ``head_local_dim_per_group`` axis. Accordingly, ``wo_a.in_features``
        is the per-group dim (e.g. 4096 for the V4 debug ckpt) — *not*
        the full flattened ``n_local_groups * head_local_dim_per_group``
        (e.g. 32768). The adapter was constructed with the per-group dim,
        so the rotation must be applied *per group*: reshape the
        flattened tensor into groups, rotate each group with the same
        block-diagonal R, then flatten back. Pre-fix, the kernel saw
        ``x.shape=(_, 32768)`` paired with ``R.shape=(128, 32, 32)`` (=
        4096 features rotated), the remaining 28672 features were left
        in the uninitialised ``torch.empty_like`` buffer — step 1 worked
        only because that buffer was zero on first allocation, and step 2
        NaN'd as soon as the allocator returned dirty memory.
        """
        if not self._adapter_enabled:
            return x
        adapter_in = self.adapter.in_features
        last = x.shape[-1]
        if last == adapter_in:
            return self.adapter(x.contiguous())
        if last % adapter_in != 0:
            raise ValueError(
                f"DSV4OFTLinear.apply_input_rotation: last-dim {last} is "
                f"not a multiple of adapter.in_features {adapter_in}"
            )
        n_groups = last // adapter_in
        x_grouped = x.reshape(*x.shape[:-1], n_groups, adapter_in).contiguous()
        x_rot = self.adapter(x_grouped)
        return x_rot.reshape(*x.shape[:-1], last)


@dataclass
class DSV4OFT(OFT):
    """OFT specialization for DSV4 native-quant linears.

    Attention ``DSV4Linear`` modules (and their TP-aware ColumnParallel /
    RowParallel subclasses) are wrapped with ``DSV4OFTLinear`` instead of
    the stock ``OFTLinear``. Native routed-MoE expert targets are handled
    at the parent ``DeepSeekV4MoE`` module by registering grouped
    ``w*_oft_r`` tensors; individual ``experts.*.w*`` linears are not
    wrapped.

    The previous ``__call__`` override skipped Bridge's
    ``normalize_disabled_bias_placeholders`` walk because that pass
    nuked V4 router-gate biases (``DeepSeekV4Gate.bias`` at layers 3-5,
    shape ``(256,)``, magnitudes ~9-27) — Bridge's ``module_bias_enabled``
    catch-all branch judged any module with a ``bias`` attr as "linear-
    like" and applied ``config.add_bias_linear`` (False for V4). That was
    fixed upstream by restricting the catch-all to ``nn.Linear`` /
    TE Linear types; this dataclass now inherits the standard ``OFT``
    ``__call__`` (the walk is a no-op for V4 under the upstream fix).
    """
    adapter_dtype: Optional[torch.dtype] = None

    def _transform_grouped_moe_oft(
        self, module: nn.Module, projections: list[str], full_name: str
    ) -> nn.Module:
        if self.r > 0:
            raise ValueError(
                "Grouped DSV4 MoE OFT uses block_size; r-based sizing is not supported."
            )
        if self.block_size <= 0:
            raise ValueError("Grouped DSV4 MoE OFT requires a positive block_size.")
        if self.coft:
            raise ValueError("Grouped DSV4 MoE OFT does not support coft=True yet.")
        if self.block_share:
            raise ValueError("Grouped DSV4 MoE OFT does not support block_share=True.")
        if self.module_dropout:
            raise ValueError("Grouped DSV4 MoE OFT does not support module_dropout.")

        ensure = getattr(module, "ensure_dsv4_expert_oft_r")
        dtype = self.adapter_dtype or torch.get_default_dtype()
        for proj in projections:
            ensure(proj, block_size=self.block_size, dtype=dtype, sample=None)
            oft_r = getattr(module, f"{proj}_oft_r", None)
            if oft_r is None:
                raise RuntimeError(f"DSV4 MoE did not register {proj}_oft_r")
            oft_r.requires_grad_(True)
        prefixes = getattr(self, "_dsv4_grouped_moe_oft_prefixes", None)
        if prefixes is None:
            prefixes = set()
            setattr(self, "_dsv4_grouped_moe_oft_prefixes", prefixes)
        prefixes.add(full_name)
        return module

    def transform(
        self, module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None
    ) -> nn.Module:
        if isinstance(module, DSV4OFTLinear):
            return module

        full_name = _compose_module_name(name, prefix)
        projections = _dsv4_moe_oft_target_projections(self.target_modules, full_name)
        if projections and callable(getattr(module, "ensure_dsv4_expert_oft_r", None)):
            return self._transform_grouped_moe_oft(module, projections, full_name)

        if (
            (proj := _dsv4_native_expert_projection(full_name)) is not None
            and proj
            in _dsv4_moe_oft_target_projections(
                self.target_modules, full_name.rsplit(".experts.", 1)[0]
            )
            and full_name.rsplit(".experts.", 1)[0]
            in getattr(self, "_dsv4_grouped_moe_oft_prefixes", set())
        ):
            return module

        from megatron.core.transformer.experimental_attention_variant.dsv4_linear import (
            DSV4Linear,
            DSV4RowParallelLinear,
        )

        if not isinstance(module, DSV4Linear):
            # Non-DSV4 linear types (norms, output_layer, etc.) should not
            # be wrapped — the explicit attention-only target glob already
            # excludes them. Defer to the stock OFT.transform so any other
            # PEFT use of this dataclass keeps working.
            return super().transform(module, name, prefix)

        if (ans := self.match(module, name, prefix)) is None:
            return module
        _, full_name = ans
        is_expert = is_expert_linear(full_name)
        input_is_parallel = isinstance(module, DSV4RowParallelLinear)

        adapter = OFTRotationModule(
            in_features=module.in_features,
            r=self.r,
            block_size=self.block_size,
            coft=self.coft,
            eps=self.eps,
            block_share=self.block_share,
            module_dropout=self.module_dropout,
            model_parallel_config=None,
            input_is_parallel=input_is_parallel,
            is_expert=is_expert,
            dtype=self.adapter_dtype,
        )
        return DSV4OFTLinear(module, adapter)


def _swap_decoder_to_dsv4(provider, megatron_models):
    """Swap each model's vanilla ``TransformerBlock`` decoder for the V4 block.

    Upstream ``GPTModel.__init__`` always builds a vanilla
    ``TransformerBlock`` for ``self.decoder`` and does not yet honour
    ``get_experimental_attention_variant_transformer_block_cls``.  Mirroring
    the dispatch hook here keeps the V4-specific change on the Bridge side
    instead of patching upstream MCore.

    The wrapper-owned ``embedding`` / ``output_layer`` are untouched.
    """
    cls = _get_experimental_attention_variant_transformer_block_cls(provider)
    if cls is None:
        return megatron_models

    models = (
        megatron_models if isinstance(megatron_models, list) else [megatron_models]
    )
    for m in models:
        if not hasattr(m, "decoder") or m.decoder is None:
            continue
        if isinstance(m.decoder, cls):
            continue
        old_decoder = m.decoder
        try:
            device = next(old_decoder.parameters()).device
        except StopIteration:
            device = None
        # Free the vanilla TransformerBlock's parameters BEFORE constructing the
        # V4 block. With ``num_moe_experts=256`` the vanilla decoder allocates
        # ~80 GiB of bf16 expert weights that are immediately unreachable; the
        # V4 block then allocates its own native-quant (FP4/FP8) weights on
        # top. On GPUs without spare headroom (e.g. shared between processes)
        # we OOM during V4 construction. Detach + free first so the V4 alloc
        # has the same memory budget as a fresh build.
        m.decoder = None
        del old_decoder
        import gc

        gc.collect()
        try:
            import torch as _torch

            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass

        new_decoder = cls(config=provider, pg_collection=provider._pg_collection)
        if device is not None:
            new_decoder = new_decoder.to(device)
        m.decoder = new_decoder

        # Megatron's standard GPTModel.forward does not thread ``input_ids``
        # to the decoder, but DeepSeekV4TransformerBlock.forward needs it
        # (per-layer hash-gate path inside DeepSeekV4Gate). Wrap the wrapper's
        # forward so we stash input_ids on the decoder before delegating to
        # the original forward, and clear it afterwards.
        if not getattr(m, "_dsv4_forward_wrapped", False):
            _orig_forward = m.forward

            def _forward_with_input_ids_cache(input_ids, *args, _orig=_orig_forward, _decoder_ref=new_decoder, **kwargs):
                _decoder_ref._input_ids_cache = input_ids
                try:
                    return _orig(input_ids, *args, **kwargs)
                finally:
                    _decoder_ref._input_ids_cache = None

            m.forward = _forward_with_input_ids_cache
            m._dsv4_forward_wrapped = True
    return megatron_models


def _upcast_output_layer_to_fp32(provider, megatron_models):
    """Upcast each model's ``output_layer`` weight to fp32 + cast input to fp32 on forward.

    Mirrors the official ``ParallelHead`` (inference/model.py:704-718): the
    lm_head is *stored* in bf16 on disk but the *parameter* and the matmul are
    fp32 for stable logit computation. Megatron's ``ColumnParallelLinear``
    has no per-instance dtype override and would otherwise hold the head in
    ``params_dtype`` (bf16), giving ~6e-2 max abs logit drift vs the official
    fp32 baseline. We re-allocate the weight as fp32 and register a
    forward-pre-hook that casts inputs to fp32 — one-time, no impact on the
    decoder's bf16 path.
    """
    import torch

    models = megatron_models if isinstance(megatron_models, list) else [megatron_models]
    for m in models:
        ol = getattr(m, "output_layer", None)
        if ol is None or getattr(ol, "weight", None) is None:
            continue
        if ol.weight.dtype == torch.float32:
            continue
        # Replace the weight with an fp32 copy on the same device.
        ol.weight = torch.nn.Parameter(ol.weight.detach().to(torch.float32))

        # Forward-pre-hook: cast inputs to fp32 to match the new weight dtype.
        # ``ColumnParallelLinear.forward(input_, weight=None, runtime_gather_output=...)``;
        # only the first positional ``input_`` needs a cast.
        def _cast_input_to_fp32(_mod, args, kwargs):
            if args:
                x = args[0]
                if isinstance(x, torch.Tensor) and x.dtype != torch.float32:
                    args = (x.float(),) + args[1:]
            elif "input_" in kwargs:
                x = kwargs["input_"]
                if isinstance(x, torch.Tensor) and x.dtype != torch.float32:
                    kwargs["input_"] = x.float()
            return args, kwargs

        ol.register_forward_pre_hook(_cast_input_to_fp32, with_kwargs=True)
    return megatron_models


def _maybe_wrap_dsv4_lm_head_batch_invariant(megatron_models):
    """Apply MEGATRON_DSV4_CANONICAL_LM_HEAD env-gated wrap to each model.

    No-op unless the env var is set to "1". Default off because the BI
    matmul is slower than cuBLAS.
    """
    from megatron.core.transformer.experimental_attention_variant.deepseek_v4 import (
        maybe_wrap_dsv4_lm_head_batch_invariant,
    )

    if isinstance(megatron_models, list):
        for m in megatron_models:
            maybe_wrap_dsv4_lm_head_batch_invariant(m)
    else:
        maybe_wrap_dsv4_lm_head_batch_invariant(megatron_models)
    return megatron_models


def _dsv4_process_group_size(group) -> int:
    if group is None:
        return 1

    size_attr = getattr(group, "size", None)
    if callable(size_attr):
        try:
            return max(1, int(size_attr()))
        except Exception:
            pass
    elif size_attr is not None:
        try:
            return max(1, int(size_attr))
        except Exception:
            pass

    try:
        from megatron.core import utils as mcore_utils

        return max(1, int(mcore_utils.get_pg_size(group)))
    except Exception:
        return 1


def _dsv4_provider_ep_size(provider) -> int:
    provider_ep_size = max(1, int(getattr(provider, "expert_model_parallel_size", 1) or 1))
    pg_collection = getattr(provider, "_pg_collection", None)
    actual_ep_size = _dsv4_process_group_size(getattr(pg_collection, "ep", None))
    return max(provider_ep_size, actual_ep_size)


def _install_dsv4_decoder_swap(provider) -> None:
    """Install the decoder-swap and fp32 output-layer upcast on a V4 provider.

    1. Registers a ``pre_wrap_hook`` so both transforms fire through the
       production ``provide_distributed_model`` path (used by AutoBridge).
    2. Wraps ``provider.provide`` so direct callers (unit tests, inspection
       tools) also get them. Both paths are idempotent so a double-fire
       is a no-op.
    """

    def _pre_wrap_hook(megatron_models, *, _p=provider):
        megatron_models = _swap_decoder_to_dsv4(_p, megatron_models)
        megatron_models = _upcast_output_layer_to_fp32(_p, megatron_models)
        megatron_models = _maybe_wrap_dsv4_lm_head_batch_invariant(megatron_models)
        return megatron_models

    provider.register_pre_wrap_hook(_pre_wrap_hook)

    original_provide = provider.provide

    def _provide_with_dsv4_decoder(*args, **kwargs):
        # ── Provider-shrink trick: the vanilla MCoreGPTModel built by
        # original_provide allocates the full bf16 TE-MoE (num_layers ×
        # num_moe_experts × 3 × hidden × moe_inter) on GPU before the V4
        # decoder swap replaces it. For DeepSeek-V4-Pro that's ~154 GiB per
        # EP=8 shard and OOMs alongside SGLang's ~23 GiB rollout residual.
        # The vanilla decoder is thrown away by _swap_decoder_to_dsv4 anyway,
        # so build a 1-layer / toy-expert version: ~MiB total, same code path,
        # then restore the real metadata before the swap allocates the actual
        # EP-sharded FP4 DSV4 weights (~12 GiB/shard).
        #
        # num_moe_experts must satisfy:
        #   * divisible by ep_size (moe_layer.py asserts ep_size | n_experts)
        #   * divisible by moe_router_num_groups (transformer_config asserts)
        #   * >= moe_router_topk (each token needs topk candidates)
        # MoELayer asserts against the actual pg_collection.ep size, which can
        # be newer than provider.expert_model_parallel_size during conversion.
        ep_size = _dsv4_provider_ep_size(provider)
        num_groups = max(1, getattr(provider, "moe_router_num_groups", 1) or 1)
        topk = max(1, getattr(provider, "moe_router_topk", 1) or 1)
        step = math.lcm(ep_size, num_groups)
        toy_num_experts = step
        while toy_num_experts < max(topk, ep_size):
            toy_num_experts += step

        saved_num_layers = provider.num_layers
        saved_num_moe_experts = provider.num_moe_experts
        saved_moe_layer_freq = provider.moe_layer_freq

        provider.num_layers = 1
        provider.num_moe_experts = toy_num_experts
        provider.moe_layer_freq = [1]

        try:
            model = original_provide(*args, **kwargs)
        finally:
            # Restore BEFORE the swap, so DeepSeekV4TransformerBlock is built
            # with the real num_layers / num_moe_experts.
            provider.num_layers = saved_num_layers
            provider.num_moe_experts = saved_num_moe_experts
            provider.moe_layer_freq = saved_moe_layer_freq

        _swap_decoder_to_dsv4(provider, model)
        _upcast_output_layer_to_fp32(provider, model)
        _maybe_wrap_dsv4_lm_head_batch_invariant(model)
        return model

    provider.provide = _provide_with_dsv4_decoder


@MegatronModelBridge.register_bridge(
    source="DeepseekV4ForCausalLM",
    target=GPTModel,
    provider=MLAModelProvider,
    model_type="deepseek_v4",
)
class DeepSeekV4Bridge(DeepSeekV3Bridge):
    """Megatron Bridge for native DeepSeek V4 Flash/Pro checkpoints."""

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MLAModelProvider:
        provider = MegatronModelBridge.provider_bridge(self, hf_pretrained)
        hf_config = hf_pretrained.config

        head_dim = getattr(hf_config, "head_dim", 512)
        num_layers = hf_config.num_hidden_layers

        provider.transformer_layer_spec = partial(
            get_transformer_block_with_experimental_attention_variant_spec
        )
        provider.experimental_attention_variant = "dsv4"
        provider.multi_latent_attention = True
        provider.qk_layernorm = True
        provider.apply_rope_fusion = False

        provider.kv_lora_rank = head_dim
        provider.qk_head_dim = head_dim
        provider.v_head_dim = head_dim
        provider.qk_pos_emb_head_dim = hf_config.qk_rope_head_dim

        # DeepSeek V4 Flash has no dense replacement prefix; every decoder layer is MoE.
        provider.ffn_hidden_size = hf_config.moe_intermediate_size
        provider.moe_layer_freq = [1] * num_layers
        provider.moe_shared_expert_intermediate_size = (
            hf_config.moe_intermediate_size * hf_config.n_shared_experts
        )
        provider.moe_router_pre_softmax = True
        provider.moe_router_load_balancing_type = "seq_aux_loss"
        provider.moe_router_score_function = hf_config.scoring_func
        # V4 uses "sqrtsoftplus" scoring; MCore validates that expert_bias is only
        # compatible with sigmoid scoring. V4's DeepSeekV4Gate hardcodes bias-free
        # routing for hash layers and a static bias for score layers — neither
        # uses MCore's expert_bias path.
        provider.moe_router_enable_expert_bias = False
        provider.moe_router_dtype = "fp32"
        provider.moe_router_topk_scaling_factor = hf_config.routed_scaling_factor
        provider.moe_aux_loss_coeff = 0.0
        # V4's DSV4Linear / DeepSeekV4Expert both construct their linears with
        # bias=False (DeepSeekV4Expert: deepseek_v4.py:~1058). Override the
        # MLATransformerConfig default (True) to match.
        provider.add_bias_linear = False

        provider.dsa_indexer_n_heads = getattr(hf_config, "index_n_heads", 64)
        provider.dsa_indexer_head_dim = getattr(hf_config, "index_head_dim", 128)
        provider.dsa_indexer_topk = getattr(hf_config, "index_topk", 512)

        provider.dsv4_o_groups = getattr(hf_config, "o_groups", 8)
        provider.dsv4_o_lora_rank = getattr(hf_config, "o_lora_rank", 1024)
        provider.dsv4_window_size = getattr(hf_config, "sliding_window", 128)
        provider.dsv4_compress_ratios = getattr(hf_config, "compress_ratios", None)
        provider.dsv4_compress_rope_theta = getattr(hf_config, "compress_rope_theta", 160000)
        provider.dsv4_hc_mult = getattr(hf_config, "hc_mult", 4)
        provider.dsv4_hc_sinkhorn_iters = getattr(hf_config, "hc_sinkhorn_iters", 20)
        provider.dsv4_hc_eps = getattr(hf_config, "hc_eps", 1e-6)
        provider.dsv4_swiglu_limit = getattr(hf_config, "swiglu_limit", 0.0)
        provider.dsv4_n_hash_layers = getattr(hf_config, "num_hash_layers", 0)
        # Runtime-only knob (not in HF config); defaults to the bit-exact
        # ``naive`` path. Override via ``--dsv4-moe-dispatcher`` after
        # ``to_megatron_provider``. ``deepep`` requires ``moe_permute_fusion``.
        provider.dsv4_moe_dispatcher = getattr(provider, "dsv4_moe_dispatcher", "naive")

        if provider.dsv4_swiglu_limit and provider.dsv4_swiglu_limit > 0:
            provider.bias_activation_fusion = False
            provider.activation_func_clamp_value = provider.dsv4_swiglu_limit

        provider.mtp_num_layers = getattr(hf_config, "num_nextn_predict_layers", 0) or None

        _install_dsv4_decoder_swap(provider)
        return provider

    @classmethod
    def megatron_to_hf_config(cls, provider: MLAModelProvider) -> dict:
        hf_cfg = super().megatron_to_hf_config(provider)
        hf_cfg["head_dim"] = getattr(provider, "kv_lora_rank", None)
        hf_cfg["qk_rope_head_dim"] = getattr(provider, "qk_pos_emb_head_dim", None)
        hf_cfg["o_groups"] = getattr(provider, "dsv4_o_groups", None)
        hf_cfg["o_lora_rank"] = getattr(provider, "dsv4_o_lora_rank", None)
        hf_cfg["sliding_window"] = getattr(provider, "dsv4_window_size", None)
        hf_cfg["compress_ratios"] = getattr(provider, "dsv4_compress_ratios", None)
        hf_cfg["compress_rope_theta"] = getattr(provider, "dsv4_compress_rope_theta", None)
        hf_cfg["hc_mult"] = getattr(provider, "dsv4_hc_mult", None)
        hf_cfg["hc_sinkhorn_iters"] = getattr(provider, "dsv4_hc_sinkhorn_iters", None)
        hf_cfg["hc_eps"] = getattr(provider, "dsv4_hc_eps", None)
        hf_cfg["swiglu_limit"] = getattr(provider, "dsv4_swiglu_limit", None)
        hf_cfg["num_hash_layers"] = getattr(provider, "dsv4_n_hash_layers", None)
        hf_cfg["index_n_heads"] = getattr(provider, "dsa_indexer_n_heads", None)
        hf_cfg["index_head_dim"] = getattr(provider, "dsa_indexer_head_dim", None)
        hf_cfg["index_topk"] = getattr(provider, "dsa_indexer_topk", None)
        hf_cfg["num_nextn_predict_layers"] = hf_cfg.get("num_nextn_predict_layers") or 0
        return hf_cfg

    def mapping_registry(self) -> MegatronMappingRegistry:
        return MegatronMappingRegistry(*_make_v4_mapping_list())


def _make_v4_mapping_list():
    """Construct V4 native-checkpoint <-> Megatron mappings.

    Notable departures from V3:
      * Embeddings/head are keyed on ``embed.weight`` / ``head.weight`` (V4)
        instead of ``model.embed_tokens.weight`` / ``lm_head.weight`` (V3).
      * Routed experts are stored as a per-expert ``ModuleList`` with
        independent ``w1``/``w2``/``w3`` tensors, so we map each one directly
        instead of fusing through :class:`GatedMLPMapping`.
      * FP4/FP8 quant siblings (``.scale``) are loaded via
        :class:`QuantScaleMapping` to preserve the non-IEEE FP8 E8M0 dtype.
      * The MoE router lives at ``mlp.gate.{weight,bias,tid2eid}`` (not
        ``mlp.router.*``); ``tid2eid`` is an int32 hash table.
    """
    return [
        # Embedding and head (owned by the GPTModel wrapper, not the V4 block).
        ColumnParallelMapping("embedding.word_embeddings.weight", "embed.weight"),
        ColumnParallelMapping("output_layer.weight", "head.weight"),
        # Final norm + decoder-level mHC head.
        ReplicatedMapping("decoder.final_layernorm.weight", "norm.weight"),
        ReplicatedMapping("decoder.hc_head_params.hc_head_fn", "hc_head_fn"),
        ReplicatedMapping("decoder.hc_head_params.hc_head_base", "hc_head_base"),
        ReplicatedMapping("decoder.hc_head_params.hc_head_scale", "hc_head_scale"),
        # Layer-level norms.
        ReplicatedMapping(
            "decoder.layers.*.input_layernorm.weight", "layers.*.attn_norm.weight"
        ),
        ReplicatedMapping(
            "decoder.layers.*.pre_mlp_layernorm.weight", "layers.*.ffn_norm.weight"
        ),
        # Per-layer mHC params.
        ReplicatedMapping("decoder.layers.*.hc_attn_fn", "layers.*.hc_attn_fn"),
        ReplicatedMapping("decoder.layers.*.hc_attn_base", "layers.*.hc_attn_base"),
        ReplicatedMapping("decoder.layers.*.hc_attn_scale", "layers.*.hc_attn_scale"),
        ReplicatedMapping("decoder.layers.*.hc_ffn_fn", "layers.*.hc_ffn_fn"),
        ReplicatedMapping("decoder.layers.*.hc_ffn_base", "layers.*.hc_ffn_base"),
        ReplicatedMapping("decoder.layers.*.hc_ffn_scale", "layers.*.hc_ffn_scale"),
        # Attention linears (FP8): weight + scale. wo_a is bf16 → no scale.
        # ``wq_a`` and ``wkv`` are bare ``DSV4Linear`` (not TP-aware) because
        # their outputs feed an RMSNorm (q_norm, kv_norm) that requires the
        # full last-dim row to compute the norm correctly. So they stay
        # replicated across TP ranks; the per-layer cost is small (4 MB +
        # 2 MB at debug, ~11 MB + ~5 MB at V4-Pro, totalling under 1 GB
        # across all layers in V4-Pro). ``wq_b`` IS TP-aware
        # (``DSV4ColumnParallelLinear``) and stays ColumnParallel here.
        ReplicatedMapping(
            "decoder.layers.*.self_attention.wq_a.weight", "layers.*.attn.wq_a.weight"
        ),
        QuantScaleMapping(
            "decoder.layers.*.self_attention.wq_a.scale",
            "layers.*.attn.wq_a.scale",
            parent_kind="replicated",
        ),
        ReplicatedMapping(
            "decoder.layers.*.self_attention.q_norm.weight", "layers.*.attn.q_norm.weight"
        ),
        ColumnParallelMapping(
            "decoder.layers.*.self_attention.wq_b.weight", "layers.*.attn.wq_b.weight"
        ),
        QuantScaleMapping(
            "decoder.layers.*.self_attention.wq_b.scale",
            "layers.*.attn.wq_b.scale",
            parent_kind="column",
        ),
        ReplicatedMapping(
            "decoder.layers.*.self_attention.wkv.weight", "layers.*.attn.wkv.weight"
        ),
        QuantScaleMapping(
            "decoder.layers.*.self_attention.wkv.scale",
            "layers.*.attn.wkv.scale",
            parent_kind="replicated",
        ),
        ReplicatedMapping(
            "decoder.layers.*.self_attention.kv_norm.weight",
            "layers.*.attn.kv_norm.weight",
        ),
        ColumnParallelMapping(
            "decoder.layers.*.self_attention.wo_a.weight", "layers.*.attn.wo_a.weight"
        ),  # bf16, no scale
        RowParallelMapping(
            "decoder.layers.*.self_attention.wo_b.weight", "layers.*.attn.wo_b.weight"
        ),
        QuantScaleMapping(
            "decoder.layers.*.self_attention.wo_b.scale",
            "layers.*.attn.wo_b.scale",
            parent_kind="row",
        ),
        ColumnParallelMapping(
            # attn_sink is per-attention-head ([n_local_heads]); shard like the
            # head dim of the column-parallel q/k projections so each rank holds
            # the sinks for its own heads. At TP=1 this degenerates to a direct
            # copy (bit-exact with the previous ReplicatedMapping).
            "decoder.layers.*.self_attention.attn_sink", "layers.*.attn.attn_sink"
        ),
        # Compressor (fp32 internal projections — replicated, no scale).
        ReplicatedMapping(
            "decoder.layers.*.self_attention.compressor.ape",
            "layers.*.attn.compressor.ape",
        ),
        ReplicatedMapping(
            "decoder.layers.*.self_attention.compressor.wkv.weight",
            "layers.*.attn.compressor.wkv.weight",
        ),
        ReplicatedMapping(
            "decoder.layers.*.self_attention.compressor.wgate.weight",
            "layers.*.attn.compressor.wgate.weight",
        ),
        ReplicatedMapping(
            "decoder.layers.*.self_attention.compressor.norm.weight",
            "layers.*.attn.compressor.norm.weight",
        ),
        # Indexer (FP8 wq_b + bf16 weights_proj + replicated compressor).
        # ``indexer.linear_wq_b`` is bare ``DSV4Linear`` (not TP-aware); its
        # output is consumed by per-head index logic that needs the full
        # row, so we keep it replicated. Per-layer cost ~8 MB at debug,
        # ~12 MB at V4-Pro.
        ReplicatedMapping(
            "decoder.layers.*.self_attention.indexer.linear_wq_b.weight",
            "layers.*.attn.indexer.wq_b.weight",
        ),
        QuantScaleMapping(
            "decoder.layers.*.self_attention.indexer.linear_wq_b.scale",
            "layers.*.attn.indexer.wq_b.scale",
            parent_kind="replicated",
        ),
        ReplicatedMapping(
            "decoder.layers.*.self_attention.indexer.linear_weights_proj.weight",
            "layers.*.attn.indexer.weights_proj.weight",
        ),
        ReplicatedMapping(
            "decoder.layers.*.self_attention.indexer.compressor.ape",
            "layers.*.attn.indexer.compressor.ape",
        ),
        ReplicatedMapping(
            "decoder.layers.*.self_attention.indexer.compressor.wkv.weight",
            "layers.*.attn.indexer.compressor.wkv.weight",
        ),
        ReplicatedMapping(
            "decoder.layers.*.self_attention.indexer.compressor.wgate.weight",
            "layers.*.attn.indexer.compressor.wgate.weight",
        ),
        ReplicatedMapping(
            "decoder.layers.*.self_attention.indexer.compressor.norm.weight",
            "layers.*.attn.indexer.compressor.norm.weight",
        ),
        # MoE gate (V4 native names: gate, not router).
        ReplicatedMapping("decoder.layers.*.mlp.gate.weight", "layers.*.ffn.gate.weight"),
        ReplicatedMapping("decoder.layers.*.mlp.gate.bias", "layers.*.ffn.gate.bias"),
        ReplicatedMapping(
            "decoder.layers.*.mlp.gate.tid2eid", "layers.*.ffn.gate.tid2eid"
        ),
        # Per-expert MoE (FP4) — w1/w3 column-parallel, w2 row-parallel.
        ColumnParallelMapping(
            "decoder.layers.*.mlp.experts.*.w1.weight",
            "layers.*.ffn.experts.*.w1.weight",
        ),
        QuantScaleMapping(
            "decoder.layers.*.mlp.experts.*.w1.scale",
            "layers.*.ffn.experts.*.w1.scale",
            parent_kind="column",
        ),
        RowParallelMapping(
            "decoder.layers.*.mlp.experts.*.w2.weight",
            "layers.*.ffn.experts.*.w2.weight",
        ),
        QuantScaleMapping(
            "decoder.layers.*.mlp.experts.*.w2.scale",
            "layers.*.ffn.experts.*.w2.scale",
            parent_kind="row",
        ),
        ColumnParallelMapping(
            "decoder.layers.*.mlp.experts.*.w3.weight",
            "layers.*.ffn.experts.*.w3.weight",
        ),
        QuantScaleMapping(
            "decoder.layers.*.mlp.experts.*.w3.scale",
            "layers.*.ffn.experts.*.w3.scale",
            parent_kind="column",
        ),
        # Shared expert (FP8).
        ColumnParallelMapping(
            "decoder.layers.*.mlp.shared_experts.w1.weight",
            "layers.*.ffn.shared_experts.w1.weight",
        ),
        QuantScaleMapping(
            "decoder.layers.*.mlp.shared_experts.w1.scale",
            "layers.*.ffn.shared_experts.w1.scale",
            parent_kind="column",
        ),
        RowParallelMapping(
            "decoder.layers.*.mlp.shared_experts.w2.weight",
            "layers.*.ffn.shared_experts.w2.weight",
        ),
        QuantScaleMapping(
            "decoder.layers.*.mlp.shared_experts.w2.scale",
            "layers.*.ffn.shared_experts.w2.scale",
            parent_kind="row",
        ),
        ColumnParallelMapping(
            "decoder.layers.*.mlp.shared_experts.w3.weight",
            "layers.*.ffn.shared_experts.w3.weight",
        ),
        QuantScaleMapping(
            "decoder.layers.*.mlp.shared_experts.w3.scale",
            "layers.*.ffn.shared_experts.w3.scale",
            parent_kind="column",
        ),
    ]
