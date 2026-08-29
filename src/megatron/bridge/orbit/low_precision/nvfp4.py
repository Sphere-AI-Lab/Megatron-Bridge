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

"""NVFP4 conversion-general helpers."""

from __future__ import annotations

from copy import deepcopy
import gc
import re
import time
from typing import Any, Iterable, Mapping

import torch

from megatron.core.dist_checkpointing.mapping import ShardedTensorFactory

# INT4 is the structural reference for the NVFP4 dist-ckpt load path; reuse the
# byte-identical helpers (replica-id tag, payload extractor, empty-storage view)
# and the per-layer-key regex so the two paths stay in lockstep.
from megatron.bridge.orbit.low_precision.int4 import (
    _empty_storage_view,
    _EXPLICIT_LAYER_KEY_RE,
    _loaded_tensor_payload,
    _replica_id_with_current_tp_rank,
)

from megatron.bridge.orbit.low_precision.common import (
    TensorSpillManager,
    add_tensor_entry,
    prepare_empty_model_state,
)
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    DirectMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
    RowParallelMapping,
    merge_qkv_biases,
    merge_qkv_weights,
)

__all__ = [
    "NVFP4_GROUP_SIZE",
    "apply_modelopt_nvfp4_to_meta_model",
    "build_fused_nvfp4_weight_entries",
    "build_megatron_nvfp4_weight_entries",
    "build_nvfp4_direct_model_state_dict",
    "collect_nvfp4_target_module_names",
    "dequantize_nvfp4",
    "extract_nvfp4_weight_bundle",
    "hf_param_uses_nvfp4",
    "is_nvfp4_source",
    "is_nvfp4_weight_mapping",
    "populate_nvfp4_quantizer_buffers",
    "quantize_to_nvfp4",
    "register_nvfp4_buffers_after_load_dense",
    "scale_to_amax",
    "transform_sharded_state_dict_for_nvfp4_dense",
]

NVFP4_GROUP_SIZE = 16
NVFP4_AMAX_SCALE = 6.0 * 448.0
_MEGATRON_WEIGHT_KEY_RE = re.compile(r"^(?P<prefix>.+)\.weight(?P<expert_idx>\d+)?$")
_SPLIT_SWIGLU_LAYOUT_FACTORY = "factory"
_SPLIT_SWIGLU_LAYOUT_SPLIT_KEYS = "split_keys"

# Dense-linear pre-load shape transform regexes.
# Match any "<module>.weight" Megatron key.
_DENSE_NVFP4_WEIGHT_RE = re.compile(r"^(?P<module>.+)\.weight$")
# Excludes grouped MoE expert keys from the dense transform (they're handled by
# peft/nvfp4_utils). Distinct from peft/nvfp4_utils._EXPERT_WEIGHT_RE which has
# narrower semantics: this one matches `experts.linear_fc1.weight` (no digit),
# `experts.linear_fc2.weight0`, `experts.linear_fc1.weight5_w`, etc.
_EXPERT_KEY_EXCLUDE_RE = re.compile(r"\.experts\.linear_fc[12]\.weight\d*(_[wv])?$")


# NVFP4 dequant: e2m1 lookup (matches NVIDIA's standard NVFP4 / e2m1 encoding).
# Low nibble = even input index, high nibble = odd input index (per modelopt /
# flashinfer / sglang parity).
_E2M1_DECODE = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def dequantize_nvfp4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_double_scale: torch.Tensor,
    weight_shape,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an NVFP4-packed weight tensor to ``dtype``.

    NVFP4 storage layout:
        weight_packed:        uint8, [out, in/2]   (two e2m1 per byte; low=even k, high=odd k)
        weight_scale:         e4m3,  [out, in/16]  (per-16-element block scale)
        weight_double_scale:  fp32 scalar          (global scale)
        weight_shape:         (out, in)            (logical unpacked shape; tuple or 1-D Tensor)

    Prefers modelopt's NVFP4QTensor.dequantize when importable (it is the source
    of truth for the encoding); falls back to a Python decode using ``_E2M1_DECODE``.
    """
    if isinstance(weight_shape, torch.Tensor):
        weight_shape = tuple(int(x) for x in weight_shape.tolist())
    out_features, in_features = int(weight_shape[0]), int(weight_shape[1])
    half_in = (in_features + 1) // 2
    scale_cols = (in_features + 15) // 16

    target_device = weight_packed.device if device is None else torch.device(device)

    # Trim padded layouts (some loaders pad the packed weight for FP4 kernels).
    w_u8 = weight_packed[:out_features, :half_in]
    ws = weight_scale[:out_features, :scale_cols]
    ws2 = weight_double_scale.reshape(()).to(torch.float32)

    # Try modelopt first.
    try:
        from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

        try:
            qtensor = NVFP4QTensor(
                torch.Size((out_features, in_features)), dtype, w_u8
            )
        except TypeError:
            qtensor = NVFP4QTensor(
                w_u8, metadata={"shape": (out_features, in_features), "dtype": dtype}
            )
        return qtensor.dequantize(
            dtype=dtype,
            scale=ws,
            double_scale=ws2.to(device=ws.device),
            block_sizes={-1: NVFP4_GROUP_SIZE},
        ).to(target_device)
    except Exception:
        pass

    # Python fallback.
    low = (w_u8 & 0x0F).to(torch.int64)
    high = ((w_u8 >> 4) & 0x0F).to(torch.int64)
    lookup = torch.tensor(_E2M1_DECODE, dtype=torch.float32, device=w_u8.device)
    vals_low = lookup[low]   # [out, in/2]
    vals_high = lookup[high]  # [out, in/2]

    vals = torch.empty(out_features, in_features, dtype=torch.float32, device=w_u8.device)
    vals[:, 0::2] = vals_low
    vals[:, 1::2] = vals_high

    ws_f32 = ws.to(torch.float32)
    ws_exp = ws_f32.repeat_interleave(16, dim=-1)[:, :in_features]
    result = vals * ws_exp * ws2
    return result.to(dtype=dtype, device=target_device)


def quantize_to_nvfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2D dense weight to the local NVFP4 buffer layout."""
    if weight.ndim != 2:
        raise ValueError(f"NVFP4 quantize expects a 2D weight, got {tuple(weight.shape)}")

    out_features, in_features = weight.shape
    if in_features % NVFP4_GROUP_SIZE != 0:
        raise ValueError(
            f"in_features must be divisible by {NVFP4_GROUP_SIZE}, got {in_features}"
        )

    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

    scale_2 = (weight.float().abs().amax() / NVFP4_AMAX_SCALE).to(torch.float32)
    try:
        qt, scale, scale_2 = NVFP4QTensor.quantize(
            weight,
            block_size=NVFP4_GROUP_SIZE,
            weights_scaling_factor_2=scale_2,
        )
    except TypeError:
        qt, scale = NVFP4QTensor.quantize(
            weight,
            block_sizes={-1: NVFP4_GROUP_SIZE},
            weights_scaling_factor_2=scale_2,
        )
    packed = qt._quantized_data
    if packed.dtype != torch.uint8:
        raise RuntimeError(f"modelopt returned packed dtype {packed.dtype}, expected uint8")

    shape = torch.tensor([out_features, in_features], dtype=torch.int64, device=weight.device)
    return packed.contiguous(), scale.contiguous(), scale_2.contiguous(), shape


def _hf_weight_has_nvfp4_bundle(weight_key: str, hf_state_dict: Mapping[str, torch.Tensor]) -> bool:
    if not weight_key.endswith(".weight"):
        return False
    return (
        weight_key in hf_state_dict
        and f"{weight_key}_scale" in hf_state_dict
        and f"{weight_key}_scale_2" in hf_state_dict
    )


def hf_param_uses_nvfp4(hf_param: Any, hf_state_dict: Mapping[str, torch.Tensor]) -> bool:
    """Return True if any HF weight key in ``hf_param`` has an NVFP4 bundle."""
    if isinstance(hf_param, str):
        return _hf_weight_has_nvfp4_bundle(hf_param, hf_state_dict)
    if isinstance(hf_param, dict):
        return any(hf_param_uses_nvfp4(value, hf_state_dict) for value in hf_param.values())
    return False


def _selective_nvfp4_quant_cfg(base_quant_cfg: Any, module_names: Iterable[str]) -> Any:
    """Restrict a ModelOpt NVFP4 ``quant_cfg`` to the given modules.

    Handles both ModelOpt schemas:

    - dict (modelopt < 0.44): pattern -> quantizer config. The global
      ``*weight_quantizer`` / ``*input_quantizer`` enables become disables and
      per-module enables are added.
    - list (modelopt >= 0.44): ordered ``{"quantizer_name": ..., "cfg": ...}``
      entries, documented as disable-all first, selective enables second,
      standard exclusions last. The global enable entries are replaced
      in place by per-module enable entries, so both surrounding blocks keep
      their documented position and precedence.
    """
    names = sorted(set(module_names))
    globals_ = ("*weight_quantizer", "*input_quantizer")

    if isinstance(base_quant_cfg, dict):
        cfg = dict(base_quant_cfg)
        enabled_weight = deepcopy(cfg.get("*weight_quantizer", {"enable": True}))
        enabled_input = deepcopy(cfg.get("*input_quantizer", {"enable": True}))
        cfg["*weight_quantizer"] = {"enable": False}
        cfg["*input_quantizer"] = {"enable": False}
        for name in names:
            cfg[f"{name}.weight_quantizer"] = deepcopy(enabled_weight)
            cfg[f"{name}.input_quantizer"] = deepcopy(enabled_input)
        return cfg

    if isinstance(base_quant_cfg, list):

        def _template(quantizer_name: str) -> dict:
            for entry in base_quant_cfg:
                if entry.get("quantizer_name") == quantizer_name and "cfg" in entry:
                    return deepcopy(entry["cfg"])
            return {"enable": True}

        enabled_weight = _template("*weight_quantizer")
        enabled_input = _template("*input_quantizer")

        cfg = []
        spliced = False
        for entry in base_quant_cfg:
            if entry.get("quantizer_name") in globals_:
                if not spliced:
                    for name in names:
                        cfg.append({"quantizer_name": f"{name}.weight_quantizer", "cfg": deepcopy(enabled_weight)})
                        cfg.append({"quantizer_name": f"{name}.input_quantizer", "cfg": deepcopy(enabled_input)})
                    spliced = True
                continue
            cfg.append(deepcopy(entry))
        if not spliced:
            for name in names:
                cfg.append({"quantizer_name": f"{name}.weight_quantizer", "cfg": deepcopy(enabled_weight)})
                cfg.append({"quantizer_name": f"{name}.input_quantizer", "cfg": deepcopy(enabled_input)})
        return cfg

    raise TypeError(
        f"Unsupported ModelOpt quant_cfg type: {type(base_quant_cfg).__name__}; expected dict or list"
    )


def apply_modelopt_nvfp4_to_meta_model(
    module: Any,
    module_names: Iterable[str] | None = None,
    *,
    compress_weights: bool = False,
) -> None:
    """Install NVFP4 quantizer modules on a meta-device Megatron module.

    For direct conversion we only need the ModelOpt wrapper classes, quantizer
    modules, and extra-state callbacks. Real weight packing is optional because
    current ModelOpt TE grouped-linear modules do not expose a persistent
    ``weight`` attribute during ``compress()``, which breaks MoE meta-model
    setup for models like Kimi-K2.5.
    """
    import modelopt.torch.quantization as mtq

    quant_cfg = deepcopy(mtq.NVFP4_DEFAULT_CFG)
    if module_names is not None:
        quant_cfg["quant_cfg"] = _selective_nvfp4_quant_cfg(quant_cfg.get("quant_cfg"), module_names)

    def _noop_forward_loop(_m):
        return None

    mtq.quantize(module, quant_cfg, _noop_forward_loop)
    if compress_weights:
        mtq.compress(module)


def is_nvfp4_source(hf_config: Any) -> bool:
    """Return True when the HF config advertises an NVFP4 quantized source model."""
    quant_cfg = getattr(hf_config, "quantization_config", None)
    if quant_cfg is None:
        return False

    if isinstance(quant_cfg, dict):
        method = str(quant_cfg.get("quant_method", "")).lower()
        algo = str(quant_cfg.get("quant_algo", "")).lower()
        fmt = str(quant_cfg.get("format", "")).lower()
    else:
        method = str(getattr(quant_cfg, "quant_method", "")).lower()
        algo = str(getattr(quant_cfg, "quant_algo", "")).lower()
        fmt = str(getattr(quant_cfg, "format", "")).lower()

    return "nvfp4" in {algo, fmt} or ("modelopt" in method and "nvfp4" in algo)


def extract_nvfp4_weight_bundle(
    weight_key: str,
    hf_state_dict: Mapping[str, torch.Tensor],
    *,
    available_keys: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Collect the NVFP4 tensor bundle associated with one HF weight key."""
    if not weight_key.endswith(".weight"):
        raise ValueError(f"Expected a weight key ending in '.weight', got: {weight_key}")

    module_prefix = weight_key[: -len(".weight")]
    bundle: dict[str, torch.Tensor] = {}
    missing = []
    expected_keys = {
        "weight": weight_key,
        "weight_scale": f"{weight_key}_scale",
        "weight_scale_2": f"{weight_key}_scale_2",
        "input_scale": f"{module_prefix}.input_scale",
    }

    if available_keys is not None:
        missing = [key for key in expected_keys.values() if key not in available_keys]

    if missing:
        raise KeyError(
            f"Missing NVFP4 bundle tensors for {weight_key}: {', '.join(missing)}"
        )

    keys_to_load = list(expected_keys.values())
    if hasattr(hf_state_dict, "source") and hasattr(hf_state_dict.source, "load_tensors"):
        loaded_by_key = hf_state_dict.source.load_tensors(keys_to_load)
    else:
        loaded_by_key = {key: hf_state_dict[key] for key in keys_to_load}

    missing = [key for key in keys_to_load if key not in loaded_by_key]
    if missing:
        raise KeyError(f"Missing NVFP4 bundle tensors for {weight_key}: {', '.join(missing)}")

    for bundle_name, key in expected_keys.items():
        bundle[bundle_name] = loaded_by_key[key]

    return bundle


def _has_nvfp4_weight_bundle_keys(
    weight_key: str,
    available_keys: set[str],
) -> bool:
    if not weight_key.endswith(".weight"):
        return False

    module_prefix = weight_key[: -len(".weight")]
    expected_keys = (
        weight_key,
        f"{weight_key}_scale",
        f"{weight_key}_scale_2",
        f"{module_prefix}.input_scale",
    )
    return all(key in available_keys for key in expected_keys)


def scale_to_amax(scale: torch.Tensor) -> torch.Tensor:
    """Convert ModelOpt NVFP4 scaling factor back to the stored amax value.

    The post-``mtq.compress`` buffer dtype is ``float32`` (verified by
    ``examples/conversion/dump_nvfp4_meta_keys.py --probe-compress``). Casting
    to bfloat16 here would drop ~8 mantissa bits from a scalar that the quant
    kernel multiplies through every 16-element block on dequant.
    """
    return (scale.float() * NVFP4_AMAX_SCALE).to(torch.float32)


def populate_nvfp4_quantizer_buffers(
    megatron_module: Any,
    *,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    input_scale: torch.Tensor,
) -> None:
    """Copy HF NVFP4 bundle scales onto a module's modelopt quantizer buffers.

    After ``mtq.quantize`` + ``mtq.compress`` (verified via ``dump_nvfp4_meta_keys.py
    --probe-roundtrip``), modelopt's ``TensorQuantizer`` expects exactly these
    per-module buffers for NVFP4:

      * ``weight_quantizer._scale``        (``float8_e4m3fn``, ``(out, in/16)``)
      * ``weight_quantizer._double_scale`` (``float32`` scalar)
      * ``weight_quantizer._amax``         (``float32`` scalar)
      * ``input_quantizer._amax``          (``float32`` scalar)

    Registering the same tensors via ``register_buffer`` produces a state_dict
    byte-identical to ``mtq.compress`` on CPU; see ``nvfp4_probe_roundtrip.log``.
    This bypass is necessary because the direct-save meta model cannot run
    ``mtq.compress`` (the kernel requires real tensors / CUDA).

    For **grouped** MoE linears that share a single ``weight_quantizer`` across
    N experts, the caller must stack per-expert tensors along a new leading
    expert dim before passing them in:

      * ``weight_scale`` -> ``(N, out, in/16)``
      * ``weight_scale_2`` -> ``(N,)``
      * ``input_scale``  -> ``(N,)`` or ``()`` if it is genuinely shared

    ``input_quantizer._amax`` is derived from ``input_scale``; if ``input_scale``
    is 1-D with length N and every value is equal, it is collapsed to a scalar
    (the shared-activation case).
    """
    weight_quantizer = getattr(megatron_module, "weight_quantizer", None)
    input_quantizer = getattr(megatron_module, "input_quantizer", None)
    if weight_quantizer is None or input_quantizer is None:
        raise RuntimeError(
            f"Expected module {type(megatron_module).__name__} to own both "
            "weight_quantizer and input_quantizer after mtq.quantize."
        )

    w_scale = weight_scale.contiguous()
    if w_scale.dtype != torch.float8_e4m3fn:
        w_scale = w_scale.to(torch.float8_e4m3fn)
    w_double = weight_scale_2.to(torch.float32).contiguous()
    # _amax = weight_scale_2 * NVFP4_AMAX_SCALE (reconstruct max-abs from the scaling factor).
    w_amax = (w_double * NVFP4_AMAX_SCALE).contiguous()

    in_scale = input_scale.to(torch.float32)
    if in_scale.ndim == 1 and torch.all(in_scale.eq(in_scale[0])):
        in_scale = in_scale[0].reshape(())
    in_amax = (in_scale * NVFP4_AMAX_SCALE).contiguous()

    weight_quantizer.register_buffer("_scale", w_scale)
    weight_quantizer.register_buffer("_double_scale", w_double)
    weight_quantizer.register_buffer("_amax", w_amax)
    input_quantizer.register_buffer("_amax", in_amax)


def _split_megatron_weight_key(megatron_weight_key: str) -> tuple[str, str, str]:
    """Split a Megatron weight key into ``(weight_key, module_prefix, expert_idx)``.

    Supports both regular keys like ``...linear_fc2.weight`` (expert_idx = "")
    and grouped-expert keys like ``...linear_fc1.weight0`` (expert_idx = "0"),
    where the quantizer state lives on the shared module prefix rather than on
    ``weight0`` itself. Returning the expert index lets callers build unique
    per-expert quantizer key names that do not collide across experts of the
    same grouped linear.
    """
    match = _MEGATRON_WEIGHT_KEY_RE.fullmatch(megatron_weight_key)
    if match is None:
        raise ValueError(
            "Expected Megatron weight key ending in '.weight' or '.weight<expert_idx>', "
            f"got: {megatron_weight_key}"
        )
    return megatron_weight_key, match.group("prefix"), match.group("expert_idx") or ""


def build_megatron_nvfp4_weight_entries(
    megatron_weight_key: str,
    bundle: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map one HF NVFP4 weight bundle to Megatron quantizer checkpoint entries.

    Emits the packed FP4 weight plus per-weight quantizer state under the
    same ``<module_prefix>.weight_quantizer.*`` key family modelopt uses
    post-``mtq.compress``. For grouped-experts modules the expert index
    (``0``, ``1``, …) is appended to per-weight keys so the 384 experts in
    a grouped linear do not collide on the shared module prefix. The shared
    ``input_quantizer._amax`` carries no suffix because all experts in a
    grouped linear see the same activation.
    """
    weight_key, weight_prefix, expert_idx = _split_megatron_weight_key(megatron_weight_key)

    required = {"weight", "weight_scale", "weight_scale_2", "input_scale"}
    missing = required.difference(bundle)
    if missing:
        raise KeyError(
            f"Missing NVFP4 bundle fields for {megatron_weight_key}: {', '.join(sorted(missing))}"
        )

    weight_scale_2 = bundle["weight_scale_2"].reshape(()).to(torch.float32)
    input_scale = bundle["input_scale"].reshape(()).to(torch.float32)

    return {
        weight_key: bundle["weight"],
        f"{weight_prefix}.weight_quantizer._scale{expert_idx}": bundle["weight_scale"],
        f"{weight_prefix}.weight_quantizer._double_scale{expert_idx}": weight_scale_2,
        f"{weight_prefix}.weight_quantizer._amax{expert_idx}": scale_to_amax(weight_scale_2),
        f"{weight_prefix}.input_quantizer._amax": scale_to_amax(input_scale),
    }


def _merge_qkv_like(
    config: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Merge Q/K/V tensors while preserving a compressed trailing feature dimension."""
    head_num = config.num_attention_heads
    num_query_groups = config.num_query_groups
    heads_per_group = head_num // num_query_groups
    head_size = config.kv_channels or (config.hidden_size // head_num)
    is_bias = q.ndim == 1
    q_head_size = head_size * 2 if getattr(config, "attention_output_gate", False) else head_size

    if is_bias:
        q_reshaped = q.view(head_num, q_head_size)
        k_reshaped = k.view(num_query_groups, head_size)
        v_reshaped = v.view(num_query_groups, head_size)
        feature_dim = None
    else:
        feature_dim = q.shape[-1]
        if k.shape[-1] != feature_dim or v.shape[-1] != feature_dim:
            raise ValueError(
                "Packed QKV tensors must share the same trailing feature dimension: "
                f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
            )
        q_reshaped = q.view(head_num, q_head_size, feature_dim)
        k_reshaped = k.view(num_query_groups, head_size, feature_dim)
        v_reshaped = v.view(num_query_groups, head_size, feature_dim)

    if getattr(config, "attention_output_gate", False):
        q_reshaped, z_reshaped = torch.chunk(q_reshaped, 2, dim=1)

    qkv_parts = []
    for i in range(num_query_groups):
        q_group = q_reshaped[i * heads_per_group : (i + 1) * heads_per_group]
        k_group = k_reshaped[i : i + 1]
        v_group = v_reshaped[i : i + 1]
        if getattr(config, "attention_output_gate", False):
            z_group = z_reshaped[i * heads_per_group : (i + 1) * heads_per_group]
            qkv_parts.extend([q_group, z_group, k_group, v_group])
        else:
            qkv_parts.extend([q_group, k_group, v_group])

    merged = torch.cat(qkv_parts, dim=0)
    if is_bias:
        return merged.reshape(-1)
    return merged.reshape(-1, feature_dim)


def _merge_fused_nvfp4_component(
    mapping: Any,
    bundles: Mapping[str, Mapping[str, torch.Tensor]],
    field: str,
    megatron_module: Any,
) -> torch.Tensor:
    if set(bundles) == {"q", "k", "v"}:
        config = mapping._get_config(megatron_module)
        return _merge_qkv_like(
            config,
            bundles["q"][field],
            bundles["k"][field],
            bundles["v"][field],
        )
    return mapping.hf_to_megatron(
        {name: bundle[field] for name, bundle in bundles.items()},
        megatron_module,
    )


def is_nvfp4_weight_mapping(
    hf_param: str | Mapping[str, str],
    hf_state_dict: Mapping[str, torch.Tensor],
    *,
    available_keys: set[str] | None = None,
) -> bool:
    """Return whether the mapping can be satisfied from NVFP4 HF tensor bundles."""
    if isinstance(hf_param, str):
        if not hf_param.endswith(".weight"):
            return False
        if available_keys is not None:
            return _has_nvfp4_weight_bundle_keys(hf_param, available_keys)
        try:
            extract_nvfp4_weight_bundle(
                hf_param,
                hf_state_dict,
                available_keys=available_keys,
            )
        except (KeyError, ValueError):
            return False
        return True

    if not hf_param:
        return False

    for source_weight_key in hf_param.values():
        if not source_weight_key.endswith(".weight"):
            return False
        if available_keys is not None:
            if not _has_nvfp4_weight_bundle_keys(source_weight_key, available_keys):
                return False
            continue
        try:
            extract_nvfp4_weight_bundle(
                source_weight_key,
                hf_state_dict,
                available_keys=available_keys,
            )
        except (KeyError, ValueError):
            return False
    return True


def shared_scalar_from_bundles(
    bundles: Mapping[str, Mapping[str, torch.Tensor]],
    field: str,
    *,
    megatron_weight_key: str,
) -> torch.Tensor:
    """Validate that all fused source bundles share the same scalar scale."""
    first_name, first_bundle = next(iter(bundles.items()))
    shared = first_bundle[field].reshape(()).to(torch.float32)

    for name, bundle in bundles.items():
        candidate = bundle[field].reshape(()).to(torch.float32)
        if not torch.allclose(candidate, shared):
            raise ValueError(
                f"NVFP4 fused mapping for {megatron_weight_key} requires a shared global scale for "
                f"{field}, but {first_name} and {name} differ"
            )

    return shared


def normalize_weight_scale_2_for_shared_fused_scale(
    bundles: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    megatron_weight_key: str,
) -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor]:
    """Fold per-bundle ``weight_scale_2`` into ``weight_scale``.

    Some fused Megatron layouts store a single global ``_double_scale`` even
    when the source HF branches carry different ``weight_scale_2`` scalars. In
    that case we preserve the effective dequant scale exactly by choosing the
    largest branch scalar as the shared global value and scaling each branch's
    blockwise ``weight_scale`` by ``branch_scale_2 / shared_scale_2``.
    """
    candidates = {
        name: bundle["weight_scale_2"].reshape(()).to(torch.float32)
        for name, bundle in bundles.items()
    }
    shared = torch.stack(tuple(candidates.values())).amax()
    if shared.item() <= 0.0:
        raise ValueError(
            f"NVFP4 fused mapping for {megatron_weight_key} requires a positive shared global "
            f"scale for weight_scale_2, got {shared.item()}"
        )

    normalized: dict[str, dict[str, torch.Tensor]] = {}
    for name, bundle in bundles.items():
        ratio = candidates[name] / shared
        normalized_bundle = dict(bundle)
        normalized_bundle["weight_scale"] = bundle["weight_scale"].float() * ratio
        normalized[name] = normalized_bundle

    return normalized, shared


def build_fused_nvfp4_weight_entries(
    mapping: Any,
    megatron_weight_key: str,
    bundles: Mapping[str, Mapping[str, torch.Tensor]],
    megatron_module: Any,
    *,
    split_swiglu_weight: bool = False,
    split_swiglu_layout: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build Megatron NVFP4 entries for fused mappings like QKV or gate/up.

    Returns ``(weight_entries, quantizer_state)`` where:
      * ``weight_entries`` is the dict of packed-weight tensors + orphan
        ``.weight_quantizer._scale*`` / ``._double_scale*`` / ``._amax*`` keys
        to write into the sharded checkpoint.
      * ``quantizer_state`` carries the merged ``weight_scale`` / ``weight_scale_2`` /
        ``input_scale`` tensors. Currently unused by the direct-save path but
        returned so a future load-path-aware route (e.g. the ModelOpt
        ``register_buffer`` approach once its save contract is understood) can
        consume them without rebuilding merging logic.
    """
    if split_swiglu_layout is None and split_swiglu_weight:
        split_swiglu_layout = _SPLIT_SWIGLU_LAYOUT_SPLIT_KEYS

    if split_swiglu_layout is not None:
        if set(bundles) != {"gate", "up"}:
            raise ValueError(
                f"Split SwiGLU NVFP4 layout for {megatron_weight_key} requires gate/up bundles, "
                f"got: {sorted(bundles)}"
            )

        weight_key, weight_prefix, expert_idx = _split_megatron_weight_key(megatron_weight_key)
        if weight_key.endswith(".weight"):
            weight_w_key = f"{weight_prefix}.weight_w"
            weight_v_key = f"{weight_prefix}.weight_v"
        else:
            weight_w_key = f"{weight_key}_w"
            weight_v_key = f"{weight_key}_v"
        try:
            weight_scale_bundles = bundles
            weight_scale_2 = shared_scalar_from_bundles(
                bundles,
                "weight_scale_2",
                megatron_weight_key=megatron_weight_key,
            )
        except ValueError:
            weight_scale_bundles, weight_scale_2 = normalize_weight_scale_2_for_shared_fused_scale(
                bundles,
                megatron_weight_key=megatron_weight_key,
            )

        merged_weight_scale = _merge_fused_nvfp4_component(
            mapping,
            weight_scale_bundles,
            "weight_scale",
            megatron_module,
        )
        input_scale = shared_scalar_from_bundles(
            bundles,
            "input_scale",
            megatron_weight_key=megatron_weight_key,
        )
        quantizer_state = {
            "weight_scale": merged_weight_scale,
            "weight_scale_2": weight_scale_2.reshape(()).to(torch.float32),
            "input_scale": input_scale.reshape(()).to(torch.float32),
        }
        if split_swiglu_layout == _SPLIT_SWIGLU_LAYOUT_FACTORY:
            merged_weight = _merge_fused_nvfp4_component(
                mapping,
                bundles,
                "weight",
                megatron_module,
            )
            entries = build_megatron_nvfp4_weight_entries(
                megatron_weight_key,
                {
                    "weight": merged_weight,
                    "weight_scale": merged_weight_scale,
                    "weight_scale_2": weight_scale_2,
                    "input_scale": input_scale,
                },
            )
            return entries, quantizer_state

        if split_swiglu_layout != _SPLIT_SWIGLU_LAYOUT_SPLIT_KEYS:
            raise ValueError(
                f"Unsupported split SwiGLU NVFP4 layout for {megatron_weight_key}: "
                f"{split_swiglu_layout}"
            )

        entries = {
            weight_w_key: bundles["gate"]["weight"],
            weight_v_key: bundles["up"]["weight"],
            f"{weight_prefix}.weight_quantizer._scale{expert_idx}": merged_weight_scale,
            f"{weight_prefix}.weight_quantizer._double_scale{expert_idx}": weight_scale_2,
            f"{weight_prefix}.weight_quantizer._amax{expert_idx}": scale_to_amax(weight_scale_2),
            f"{weight_prefix}.input_quantizer._amax": scale_to_amax(input_scale),
        }
        return entries, quantizer_state

    merged_weight_scale = _merge_fused_nvfp4_component(
        mapping,
        bundles,
        "weight_scale",
        megatron_module,
    )
    merged_weight = _merge_fused_nvfp4_component(
        mapping,
        bundles,
        "weight",
        megatron_module,
    )
    weight_scale_2 = shared_scalar_from_bundles(
        bundles,
        "weight_scale_2",
        megatron_weight_key=megatron_weight_key,
    )
    input_scale = shared_scalar_from_bundles(
        bundles,
        "input_scale",
        megatron_weight_key=megatron_weight_key,
    )
    entries = build_megatron_nvfp4_weight_entries(
        megatron_weight_key,
        {
            "weight": merged_weight,
            "weight_scale": merged_weight_scale,
            "weight_scale_2": weight_scale_2,
            "input_scale": input_scale,
        },
    )
    quantizer_state = {
        "weight_scale": merged_weight_scale,
        "weight_scale_2": weight_scale_2.reshape(()).to(torch.float32),
        "input_scale": input_scale.reshape(()).to(torch.float32),
    }
    return entries, quantizer_state


def _uses_split_swiglu_weight_layout(
    megatron_weight_key: str,
    model_template: dict[str, Any],
) -> str | None:
    if ".linear_fc1.weight" not in megatron_weight_key:
        return None

    if isinstance(model_template.get(megatron_weight_key), ShardedTensorFactory):
        return _SPLIT_SWIGLU_LAYOUT_FACTORY

    if (
        f"{megatron_weight_key}_w" in model_template
        and f"{megatron_weight_key}_v" in model_template
    ):
        return _SPLIT_SWIGLU_LAYOUT_SPLIT_KEYS

    return None


def convert_hf_weight_for_direct_save(task: Any, hf_weights: Any) -> torch.Tensor:
    """HF -> Megatron conversion for the TP=1 direct-save path.

    The direct converter builds a meta-device Megatron model. Some regular
    mapping classes move tensors onto the destination module's device during
    ``hf_to_megatron(...)`` which would turn real CPU tensors into meta tensors.
    For single-rank direct save we can bypass those device moves and apply only
    the structural transforms that matter for checkpoint layout.
    """
    mapping = task.mapping

    if getattr(mapping, "tp_size", 1) != 1:
        raise ValueError(
            "Direct NVFP4 converter currently supports only single-rank TP=1 checkpoint writes."
        )

    if isinstance(mapping, AutoMapping):
        converted = hf_weights
        if mapping.permute_dims is not None:
            converted = torch.permute(converted, mapping.permute_dims).contiguous()
        return converted

    if isinstance(
        mapping,
        (DirectMapping, ColumnParallelMapping, RowParallelMapping, ReplicatedMapping),
    ):
        return hf_weights

    if isinstance(mapping, QKVMapping):
        config = mapping._get_config(task.megatron_module)
        if hf_weights["q"].ndim == 1:
            return merge_qkv_biases(
                config,
                hf_weights["q"],
                hf_weights["k"],
                hf_weights["v"],
            )
        return merge_qkv_weights(
            config,
            hf_weights["q"],
            hf_weights["k"],
            hf_weights["v"],
        )

    if isinstance(mapping, GatedMLPMapping):
        return torch.cat([hf_weights["gate"], hf_weights["up"]], dim=0)

    converted = mapping.hf_to_megatron(hf_weights, task.megatron_module)
    if converted is None:
        raise RuntimeError(f"Conversion returned None for {task.param_name}")
    if converted.device.type == "meta":
        raise RuntimeError(
            f"Conversion unexpectedly produced a meta tensor for {task.param_name}. "
            "Add an explicit direct-save handler for this mapping type."
        )
    return converted


def collect_nvfp4_target_module_names(
    conversion_tasks: list[Any],
    hf_state_dict: Mapping[str, torch.Tensor],
    *,
    show_progress: bool = False,
) -> set[str]:
    """Collect Megatron module names whose source HF weights are stored as NVFP4 bundles."""
    module_names = set()
    available_keys = set(hf_state_dict.keys())
    total_tasks = len(conversion_tasks)
    log_interval = max(1, total_tasks // 100) if show_progress else None
    t_start = time.monotonic() if show_progress else None

    for i, task in enumerate(conversion_tasks):
        if task is None or task.megatron_module is None:
            pass
        elif is_nvfp4_weight_mapping(
            task.mapping.hf_param,
            hf_state_dict,
            available_keys=available_keys,
        ):
            module_names.add(task.param_name.rsplit(".", 1)[0])

        if show_progress and (((i + 1) % log_interval == 0) or ((i + 1) == total_tasks)):
            elapsed = time.monotonic() - t_start
            eta = elapsed / (i + 1) * (total_tasks - i - 1) if i + 1 < total_tasks else 0.0
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
            print(
                f"  [{i+1}/{total_tasks}] found {len(module_names)} NVFP4 target modules | "
                f"elapsed {elapsed_str} ETA {eta_str}",
                flush=True,
            )
    return module_names


def build_nvfp4_direct_model_state_dict(
    bridge: Any,
    hf_pretrained: Any,
    meta_model: list[Any],
    model_template: dict[str, Any],
    conversion_tasks: list[Any] | None = None,
    spill_manager: TensorSpillManager | None = None,
) -> dict[str, Any]:
    """Populate a direct-save model state dict from NVFP4 bridge conversion tasks."""
    if len(meta_model) != 1:
        raise ValueError(
            "Direct NVFP4 converter currently supports a single Megatron model chunk."
        )

    model_state = prepare_empty_model_state(model_template)
    if conversion_tasks is None:
        conversion_tasks = bridge.build_conversion_tasks(hf_pretrained, meta_model)
    hf_state_dict = hf_pretrained.state
    available_keys = set(hf_state_dict.keys())

    num_regular = 0
    num_nvfp4 = 0
    total_tasks = len(conversion_tasks)
    log_interval = max(1, total_tasks // 100)
    gc_interval = max(1, total_tasks // 20)  # force GC ~20 times during conversion
    t_start = time.monotonic()

    for i, task in enumerate(conversion_tasks):
        if task is None or task.megatron_module is None:
            continue

        hf_param = task.mapping.hf_param
        if is_nvfp4_weight_mapping(
            hf_param,
            hf_state_dict,
            available_keys=available_keys,
        ):
            split_swiglu_layout = None
            if isinstance(hf_param, str):
                bundle = extract_nvfp4_weight_bundle(
                    hf_param,
                    hf_state_dict,
                    available_keys=available_keys,
                )
                entries = build_megatron_nvfp4_weight_entries(
                    task.param_name,
                    bundle,
                )
                quantizer_state = {
                    "weight_scale": bundle["weight_scale"],
                    "weight_scale_2": bundle["weight_scale_2"].reshape(()).to(torch.float32),
                    "input_scale": bundle["input_scale"].reshape(()).to(torch.float32),
                }
                del bundle
            else:
                bundles = {
                    name: extract_nvfp4_weight_bundle(
                        source_weight_key,
                        hf_state_dict,
                        available_keys=available_keys,
                    )
                    for name, source_weight_key in hf_param.items()
                }
                split_swiglu_layout = _uses_split_swiglu_weight_layout(
                    task.param_name,
                    model_template,
                )
                entries, quantizer_state = build_fused_nvfp4_weight_entries(
                    task.mapping,
                    task.param_name,
                    bundles,
                    task.megatron_module,
                    split_swiglu_layout=split_swiglu_layout,
                )
                del bundles

            for key, tensor in entries.items():
                # NVFP4 direct-save tensors are already in their packed on-disk
                # layout. Do not reuse the dense meta-model template here because
                # its tensor shapes may still reflect the uncompressed BF16
                # weights. The exception is a factory-backed split-SwiGLU weight
                # entry, where we must preserve the original factory metadata.
                template_entry = None
                if (
                    split_swiglu_layout == _SPLIT_SWIGLU_LAYOUT_FACTORY
                    and key == task.param_name
                ):
                    template_entry = model_template.get(task.param_name)
                add_tensor_entry(
                    model_state,
                    key,
                    tensor,
                    template_entry,
                    spill_manager=spill_manager,
                )
            del entries
            # quantizer_state is returned by the entry builders so the scales
            # flow through the same code path as the weight. It is currently
            # unused because save_sharded_modelopt_state does not serialize
            # register_buffer'd scales on uncompressed meta modules (verified
            # empirically); the scale tensors reach disk via the orphan
            # .weight_quantizer._scale* keys emitted by build_megatron_nvfp4_weight_entries.
            del quantizer_state
            num_nvfp4 += 1
            if (i + 1) % log_interval == 0 or (i + 1) == total_tasks:
                elapsed = time.monotonic() - t_start
                eta = elapsed / (i + 1) * (total_tasks - i - 1) if i + 1 < total_tasks else 0.0
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
                print(
                    f"  [{i+1}/{total_tasks}] {num_regular} regular, {num_nvfp4} NVFP4 | "
                    f"elapsed {elapsed_str} ETA {eta_str} | {task.param_name}",
                    flush=True,
                )
            if (i + 1) % gc_interval == 0:
                gc.collect()
            continue

        hf_weights = bridge.maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)
        converted = convert_hf_weight_for_direct_save(task, hf_weights)
        add_tensor_entry(
            model_state,
            task.param_name,
            converted,
            model_template.get(task.param_name),
            spill_manager=spill_manager,
        )
        del hf_weights, converted
        num_regular += 1

        if (i + 1) % log_interval == 0 or (i + 1) == total_tasks:
            elapsed = time.monotonic() - t_start
            eta = elapsed / (i + 1) * (total_tasks - i - 1) if i + 1 < total_tasks else 0.0
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
            print(
                f"  [{i+1}/{total_tasks}] {num_regular} regular, {num_nvfp4} NVFP4 | "
                f"elapsed {elapsed_str} ETA {eta_str} | {task.param_name}",
                flush=True,
            )

        if (i + 1) % gc_interval == 0:
            gc.collect()

    print(
        f"Prepared direct checkpoint state dict: {num_regular} regular tensors, "
        f"{num_nvfp4} NVFP4 mappings"
    )
    return model_state


def transform_sharded_state_dict_for_nvfp4_dense(
    sharded_state_dict,
    checkpoint_keys,
):
    """Replace dense-linear bf16 weight ShardedTensors with NVFP4 4-tuple entries.

    For every ``<module>.weight`` ShardedTensor whose owning module has a
    ``<module>.weight_quantizer._scale`` key in the checkpoint, swap the bf16
    weight for a uint8 ``[out, in/2]`` ShardedTensor and add three sibling
    ShardedTensors for the quantizer scales:

      * ``weight_quantizer._scale``        (``float8_e4m3fn``, ``(out, in/16)``)
      * ``weight_quantizer._double_scale`` (``float32`` scalar)
      * ``weight_quantizer._amax``         (``float32`` scalar, loaded but
        unused at forward time)

    Grouped MoE expert weight keys (``*.experts.linear_fc{1,2}.weight*``) are
    explicitly skipped — those are handled by ``peft/nvfp4_utils`` which knows
    about the per-expert layout. ``input_quantizer._amax`` is intentionally not
    loaded: the ``input_quantizer`` module is dropped during PEFT bootstrap and
    isn't present on the runtime model.
    """
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    ckpt_keys = {str(k) for k in checkpoint_keys}

    def _split_factory(value: ShardedTensorFactory):
        built_sub = value.build()
        if isinstance(built_sub, list):
            return built_sub[0], built_sub
        sub_sh_ten = next(iter(built_sub.values()))
        return sub_sh_ten, list(built_sub.values())

    def _split_swiglu_checkpoint_keys(module_path: str) -> tuple[str, str] | None:
        weight_w_key = f"{module_path}.weight_w"
        weight_v_key = f"{module_path}.weight_v"
        if weight_w_key in ckpt_keys and weight_v_key in ckpt_keys:
            return weight_w_key, weight_v_key
        return None

    def _packed_weight_sharded_tensor(
        checkpoint_key: str,
        *,
        sh_ten: ShardedTensor,
        local_out: int,
        local_in: int,
        global_out: int,
        global_in: int,
        out_offset: int,
        in_offset: int,
        axis_fragmentations: tuple[int, ...],
    ) -> ShardedTensor:
        return ShardedTensor(
            key=checkpoint_key,
            data=torch.empty((local_out, local_in // 2), dtype=torch.uint8, device="cpu"),
            dtype=torch.uint8,
            local_shape=(local_out, local_in // 2),
            global_shape=(global_out, global_in // 2),
            global_offset=(out_offset, in_offset // 2),
            axis_fragmentations=axis_fragmentations,
            replica_id=sh_ten.replica_id,
            prepend_axis_num=0,
            allow_shape_mismatch=True,
        )

    def _packed_split_sharded_tensor(
        checkpoint_key: str,
        sub_sh_ten: ShardedTensor,
        *,
        split_factor: int,
        same_key_splits: bool,
    ) -> ShardedTensor:
        prepend = sub_sh_ten.prepend_axis_num
        global_shape = list(sub_sh_ten.global_shape[prepend:])
        global_offset = list(sub_sh_ten.global_offset[prepend:])
        axis_fragmentations = list(sub_sh_ten.axis_fragmentations[prepend:])
        local_shape = list(sub_sh_ten.local_shape)
        if len(local_shape) > len(global_shape):
            local_shape = local_shape[-len(global_shape):]

        if same_key_splits:
            if global_shape[0] % split_factor != 0:
                raise ValueError(
                    f"Unexpected NVFP4 split SwiGLU shape for {checkpoint_key}: "
                    f"{global_shape[0]} not divisible by {split_factor}"
                )
            if axis_fragmentations[0] % split_factor != 0:
                raise ValueError(
                    f"Unexpected NVFP4 split SwiGLU fragmentation for {checkpoint_key}: "
                    f"{axis_fragmentations[0]} not divisible by {split_factor}"
                )
            split_global_out = global_shape[0] // split_factor
            global_shape[0] = split_global_out
            global_offset[0] %= split_global_out
            axis_fragmentations[0] //= split_factor

        local_in = local_shape[-1]
        if local_in % 2 != 0:
            raise ValueError(
                f"NVFP4 split SwiGLU requires in-features divisible by 2, "
                f"got {local_in} for {checkpoint_key}"
            )
        local_shape[-1] = local_in // 2
        global_shape[-1] //= 2
        global_offset[-1] //= 2

        return ShardedTensor(
            key=checkpoint_key,
            data=torch.empty(tuple(local_shape), dtype=torch.uint8, device="cpu"),
            dtype=torch.uint8,
            local_shape=tuple(local_shape),
            global_shape=tuple(global_shape),
            global_offset=tuple(global_offset),
            axis_fragmentations=tuple(axis_fragmentations),
            replica_id=sub_sh_ten.replica_id,
            prepend_axis_num=0,
            allow_shape_mismatch=True,
        )

    new_sd = {}
    for key, value in sharded_state_dict.items():
        skey = str(key)

        if _EXPERT_KEY_EXCLUDE_RE.search(skey):
            new_sd[key] = value
            continue

        m = _DENSE_NVFP4_WEIGHT_RE.match(skey)
        if m is None:
            new_sd[key] = value
            continue

        module_path = m.group("module")
        scale_key = f"{module_path}.weight_quantizer._scale"
        if scale_key not in ckpt_keys:
            new_sd[key] = value
            continue

        sub_sh_tens = None
        if isinstance(value, ShardedTensorFactory):
            sh_ten, sub_sh_tens = _split_factory(value)
            if len(sub_sh_tens) < 2:
                new_sd[key] = value
                continue

            fused_local_out = value.data.shape[0]
            split_local_out = sh_ten.local_shape[-2]
            if split_local_out == 0 or fused_local_out % split_local_out != 0:
                raise ValueError(
                    f"Unexpected NVFP4 SwiGLU factory shapes for {skey}: "
                    f"fused local out={fused_local_out}, split local out={split_local_out}"
                )
            split_factor = fused_local_out // split_local_out
            out_axis = sh_ten.prepend_axis_num
            rank_offset = sh_ten.global_offset[out_axis] // split_local_out
            axis_fragmentations_override = list(sh_ten.axis_fragmentations)
            if all(sub.key == value.key for sub in sub_sh_tens[:2]):
                if axis_fragmentations_override[out_axis] % split_factor != 0:
                    raise ValueError(
                        f"Unexpected NVFP4 SwiGLU factory fragmentation for {skey}: "
                        f"{axis_fragmentations_override[out_axis]} not divisible by {split_factor}"
                    )
                axis_fragmentations_override[out_axis] //= split_factor
                global_out_override = sh_ten.global_shape[out_axis]
            else:
                global_out_override = sh_ten.global_shape[out_axis] * split_factor
            local_out_override = fused_local_out
            out_offset_override = rank_offset * fused_local_out
            axis_fragmentations_override = tuple(axis_fragmentations_override)
        elif isinstance(value, ShardedTensor) and len(value.local_shape) >= 2:
            sh_ten = value
            split_factor = 2
            local_out_override = None
            global_out_override = None
            out_offset_override = None
            axis_fragmentations_override = sh_ten.axis_fragmentations
        else:
            new_sd[key] = value
            continue

        prepend = sh_ten.prepend_axis_num
        local_out = local_out_override if local_out_override is not None else sh_ten.local_shape[-2]
        local_in = sh_ten.local_shape[-1]
        global_out = global_out_override if global_out_override is not None else sh_ten.global_shape[-2]
        global_in = sh_ten.global_shape[-1]

        # NVFP4 packs two e2m1 nibbles per byte; the per-group scale spans
        # NVFP4_GROUP_SIZE (=16) input columns. Both must divide the local
        # in-axis cleanly or the dist-ckpt loader cannot stitch shards back
        # together.
        if local_in % 2 != 0:
            raise ValueError(
                f"NVFP4 requires in-features divisible by 2 (pack factor), "
                f"got {local_in} for {skey}"
            )
        if local_in % NVFP4_GROUP_SIZE != 0:
            raise ValueError(
                f"NVFP4 requires in-features divisible by NVFP4_GROUP_SIZE="
                f"{NVFP4_GROUP_SIZE}, got {local_in} for {skey}"
            )

        packed_in = local_in // 2
        global_packed_in = global_in // 2
        num_groups = local_in // NVFP4_GROUP_SIZE
        global_num_groups = global_in // NVFP4_GROUP_SIZE

        # Strip the optional leading "stacked-layer" axis from both the
        # axis_fragmentations tuple and the global-offset tuple — once we
        # unroll the stacked layout below, every emitted ShardedTensor is
        # 2-D (out, in) with prepend_axis_num=0.
        weight_axis_fragmentations = tuple(axis_fragmentations_override[prepend:])
        out_offset_full = (
            out_offset_override
            if out_offset_override is not None
            else (
                sh_ten.global_offset[prepend]
                if len(sh_ten.global_offset) > prepend
                else 0
            )
        )
        in_offset_full = (
            sh_ten.global_offset[prepend + 1]
            if len(sh_ten.global_offset) > prepend + 1
            else 0
        )

        # Megatron may emit linear weights in a stacked-layer layout
        # (key path has a literal `layers.` segment and prepend_axis_num=1
        # so global_shape=(num_layers, out, in)). The on-disk checkpoint
        # carries one quantized entry per layer, so unroll the stacked
        # template into per-layer keys with prepend_axis_num=0.
        has_explicit_layer_idx = _EXPLICIT_LAYER_KEY_RE.search(skey) is not None
        if prepend > 0 and not has_explicit_layer_idx:
            num_local_layers = sh_ten.local_shape[0]
            layer_start = sh_ten.global_offset[prepend - 1]
            layer_indices = [layer_start + i for i in range(num_local_layers)]
        else:
            layer_indices = [None]

        for global_layer_idx in layer_indices:
            if global_layer_idx is None:
                ckpt_key = skey
            else:
                ckpt_key = re.sub(
                    r"(^|\.)layers\.",
                    rf"\1layers.{global_layer_idx}.",
                    skey,
                    count=1,
                )
            ckpt_module_path = _DENSE_NVFP4_WEIGHT_RE.match(ckpt_key).group("module")
            ckpt_scale_key = f"{ckpt_module_path}.weight_quantizer._scale"
            ckpt_ds_key = f"{ckpt_module_path}.weight_quantizer._double_scale"
            ckpt_amax_key = f"{ckpt_module_path}.weight_quantizer._amax"
            split_weight_keys = _split_swiglu_checkpoint_keys(ckpt_module_path)

            if split_weight_keys is not None:
                if sub_sh_tens is not None:
                    same_key_splits = all(sub.key == value.key for sub in sub_sh_tens[:2])
                    for split_key, sub_sh_ten in zip(split_weight_keys, sub_sh_tens[:2]):
                        new_sd[split_key] = _packed_split_sharded_tensor(
                            split_key,
                            sub_sh_ten,
                            split_factor=split_factor,
                            same_key_splits=same_key_splits,
                        )
                else:
                    if local_out % 2 != 0 or global_out % 2 != 0:
                        raise ValueError(
                            f"NVFP4 split SwiGLU requires even out-features, "
                            f"got local={local_out}, global={global_out} for {skey}"
                        )
                    split_axis_fragmentations = weight_axis_fragmentations
                    for split_key in split_weight_keys:
                        new_sd[split_key] = _packed_weight_sharded_tensor(
                            split_key,
                            sh_ten=sh_ten,
                            local_out=local_out // 2,
                            local_in=local_in,
                            global_out=global_out // 2,
                            global_in=global_in,
                            out_offset=out_offset_full % (global_out // 2),
                            in_offset=in_offset_full,
                            axis_fragmentations=split_axis_fragmentations,
                        )
            else:
                # weight -> uint8 [out, in/2]
                packed = ShardedTensor(
                    key=ckpt_key,
                    data=torch.empty((local_out, packed_in), dtype=torch.uint8, device="cpu"),
                    dtype=torch.uint8,
                    local_shape=(local_out, packed_in),
                    global_shape=(global_out, global_packed_in),
                    global_offset=(out_offset_full, in_offset_full // 2),
                    axis_fragmentations=weight_axis_fragmentations,
                    replica_id=sh_ten.replica_id,
                    prepend_axis_num=0,
                    allow_shape_mismatch=True,
                )
                new_sd[ckpt_key] = packed

            # weight_quantizer._scale -> fp8_e4m3fn [out, in/16]
            scale_st = ShardedTensor(
                key=ckpt_scale_key,
                data=torch.empty(
                    (local_out, num_groups), dtype=torch.float8_e4m3fn, device="cpu"
                ),
                dtype=torch.float8_e4m3fn,
                local_shape=(local_out, num_groups),
                global_shape=(global_out, global_num_groups),
                global_offset=(out_offset_full, in_offset_full // NVFP4_GROUP_SIZE),
                axis_fragmentations=weight_axis_fragmentations,
                replica_id=sh_ten.replica_id,
                prepend_axis_num=0,
                allow_shape_mismatch=True,
            )
            new_sd[ckpt_scale_key] = scale_st

            # weight_quantizer._double_scale -> fp32 scalar.
            # Replicated across TP — wrap replica_id with the current TP rank
            # so dist-ckpt sees one primary writer per TP rank rather than
            # every rank claiming to be the same primary.
            ds_st = ShardedTensor(
                key=ckpt_ds_key,
                data=torch.empty((), dtype=torch.float32, device="cpu"),
                dtype=torch.float32,
                local_shape=(),
                global_shape=(),
                global_offset=(),
                axis_fragmentations=(),
                replica_id=_replica_id_with_current_tp_rank(sh_ten.replica_id),
                prepend_axis_num=0,
                allow_shape_mismatch=True,
            )
            new_sd[ckpt_ds_key] = ds_st

            # weight_quantizer._amax -> fp32 scalar (loaded but unused at fwd).
            amax_st = ShardedTensor(
                key=ckpt_amax_key,
                data=torch.empty((), dtype=torch.float32, device="cpu"),
                dtype=torch.float32,
                local_shape=(),
                global_shape=(),
                global_offset=(),
                axis_fragmentations=(),
                replica_id=_replica_id_with_current_tp_rank(sh_ten.replica_id),
                prepend_axis_num=0,
                allow_shape_mismatch=True,
            )
            new_sd[ckpt_amax_key] = amax_st

    return new_sd


# Matches dense per-module weight-quantizer scale entries left in the loaded
# state dict by ``transform_sharded_state_dict_for_nvfp4_dense``. The capturing
# group identifies the owning Megatron module path so we can navigate to it
# from the live model and rebind ``module.weight`` to a real bf16 tensor.
_DENSE_NVFP4_SCALE_RE = re.compile(r"^(?P<module>.+)\.weight_quantizer\._scale$")


def register_nvfp4_buffers_after_load_dense(
    model,
    loaded_state_dict,
) -> int:
    """Dequantize loaded NVFP4 dense weights to bf16 and rebind module.weight.

    For every loaded ``<module>.weight_quantizer._scale`` entry, find the
    matching uint8 packed weight + double_scale, dequantize to bf16, and
    assign the result back onto ``module.weight`` as a fresh
    ``torch.nn.Parameter``. The packed scale and double-scale are kept as
    non-persistent buffers so we can re-dequantize if needed (for QAT-style
    flows); the bf16 weight is what the standard TE forward actually uses.

    Unlike the INT4 dense sibling (which keeps the packed buffer and
    re-dequantizes per forward), this dequantizes once at load time because
    ``linear_qkv``/``linear_proj``/etc. are NOT OFT-wrapped — the live module
    needs a real bf16 ``weight`` Parameter that the unmodified TE linear can
    consume.
    """
    # First pass: index every per-module scale entry; this is the trigger for
    # the rest of the bundle. Doing it in two passes (rather than building the
    # full bundle inline) means we only walk the state dict's full key space
    # once and the second pass touches only the targeted module paths.
    targets: dict[str, dict[str, Any]] = {}
    for key, value in loaded_state_dict.items():
        skey = str(key)
        m = _DENSE_NVFP4_SCALE_RE.match(skey)
        if m is None:
            continue
        module_path = m.group("module")
        targets.setdefault(module_path, {})["scale"] = value

    # Second pass: fill in weight / double_scale / amax for each target.
    for module_path in list(targets):
        weight_key = f"{module_path}.weight"
        weight_w_key = f"{module_path}.weight_w"
        weight_v_key = f"{module_path}.weight_v"
        ds_key = f"{module_path}.weight_quantizer._double_scale"
        am_key = f"{module_path}.weight_quantizer._amax"
        if weight_key in loaded_state_dict:
            targets[module_path]["weight"] = loaded_state_dict[weight_key]
        if weight_w_key in loaded_state_dict:
            targets[module_path]["weight_w"] = loaded_state_dict[weight_w_key]
        if weight_v_key in loaded_state_dict:
            targets[module_path]["weight_v"] = loaded_state_dict[weight_v_key]
        if ds_key in loaded_state_dict:
            targets[module_path]["double_scale"] = loaded_state_dict[ds_key]
        if am_key in loaded_state_dict:
            targets[module_path]["amax"] = loaded_state_dict[am_key]

    registered = 0
    for module_path, parts in targets.items():
        has_weight = "weight" in parts or {"weight_w", "weight_v"}.issubset(parts)
        if not has_weight or "scale" not in parts or "double_scale" not in parts:
            continue

        # Navigate to the module.
        module = model
        for attr in module_path.split("."):
            module = getattr(module, attr)

        if "weight" in parts:
            packed = _loaded_tensor_payload(parts["weight"])
        else:
            packed = torch.cat(
                [
                    _loaded_tensor_payload(parts["weight_w"]),
                    _loaded_tensor_payload(parts["weight_v"]),
                ],
                dim=-2,
            )
        scale = _loaded_tensor_payload(parts["scale"])
        double_scale = _loaded_tensor_payload(parts["double_scale"])

        # Logical (unpacked) shape: derive from the existing bf16 weight
        # Parameter on the live module (which still holds the dense template
        # shape — only its storage may have been emptied/placeholdered).
        w_param = getattr(module, "weight", None)
        if w_param is None:
            continue
        out_features, in_features = w_param.shape[-2], w_param.shape[-1]

        bf16 = dequantize_nvfp4(
            packed,
            scale,
            double_scale,
            (out_features, in_features),
            device=w_param.device if w_param.device.type != "meta" else None,
            dtype=torch.bfloat16,
        )

        with torch.no_grad():
            new_param = torch.nn.Parameter(bf16, requires_grad=w_param.requires_grad)
            # Preserve TP / sharding attributes that downstream code (Megatron
            # parallel_state, distributed optimizer, dist-ckpt save) may
            # inspect on the Parameter (e.g. ``tensor_model_parallel``,
            # ``partition_dim``, ``shared``, ``allreduce``). Skip dunders,
            # ``data``, and ``grad`` which are managed by the Parameter
            # itself.
            for attr_name, attr_value in vars(w_param).items():
                if attr_name.startswith("_") or attr_name in {"data", "grad"}:
                    continue
                setattr(new_param, attr_name, attr_value)
            module.weight = new_param

        # Keep the packed scale + double_scale around as non-persistent
        # buffers for future use (e.g. QAT-style re-quantization). Not
        # strictly required for the bf16 forward path, but cheap to retain
        # and matches the spec ledger. Co-locate with the dequantized weight
        # (which lives on packed.device when w_param is on meta).
        scale_target_device = (
            w_param.device if w_param.device.type != "meta" else packed.device
        )
        module.register_buffer(
            "_nvfp4_weight_scale",
            scale.to(scale_target_device),
            persistent=False,
        )
        module.register_buffer(
            "_nvfp4_weight_double_scale",
            double_scale.to(scale_target_device),
            persistent=False,
        )

        registered += 1

    print(
        f"[NVFP4 dense register] Registered NVFP4 buffers for {registered} dense linears"
    )
    return registered
