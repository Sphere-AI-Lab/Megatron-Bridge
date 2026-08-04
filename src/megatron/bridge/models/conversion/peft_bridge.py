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

from __future__ import annotations

import itertools
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from string import digits
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, TypeVar, Union

import torch
from megatron.core import parallel_state
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_pg_rank, unwrap_model

from megatron.bridge.models.conversion.param_mapping import (
    ColumnParallelMapping,
    ReplicatedMapping,
    RowParallelMapping,
    _split_gdn_grouped_to_separate,
    split_gdn_linear_weights,
    split_qkv_weights,
)
from megatron.bridge.models.conversion.utils import (
    extract_sort_key,
    get_module_and_param_from_name,
    persistent_buffers,
)
from megatron.bridge.peft.canonical_lora import ModuleDict
from megatron.bridge.peft.lora import LoRAMerge
from megatron.bridge.peft.utils import ParallelLinearAdapter, get_adapter_attributes_from_linear, is_expert_linear


if TYPE_CHECKING:
    from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
    from megatron.bridge.models.conversion.model_bridge import HFWeightTuple, MegatronWeightTuple, WeightConversionTask
    from megatron.bridge.peft.base import PEFT


MegatronModel = TypeVar("MegatronModel", bound=MegatronModule)


ADAPTER_NAME_MAP = {
    # Map HF base parameter suffixes (keys) to CanonicalLoRA adapter keys (values)
    ".q_proj.weight": "adapter_q",
    ".k_proj.weight": "adapter_k",
    ".v_proj.weight": "adapter_v",
    ".gate_proj.weight": "adapter_gate",
    ".up_proj.weight": "adapter_up",
}
ADAPTER_KEY_TO_SUFFIX = {value: key for key, value in ADAPTER_NAME_MAP.items()}

# Map Megatron adapter suffixes to HuggingFace LoRA parameter suffixes
MEGATRON_TO_HF_LORA_SUFFIX = {
    ".linear_in.weight": ".lora_A.weight",
    ".linear_out.weight": ".lora_B.weight",
}

# Map Megatron adapter suffixes to HuggingFace OFT parameter suffixes
MEGATRON_TO_HF_OFT_SUFFIX = {
    ".oft_r": ".oft_R",
}

GDN_IN_PROJ_KEYS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
_DSV4_NATIVE_EXPERT_OFT_BASE_PREFIX_RE = re.compile(r"^(?P<head>.*\.mlp\.experts\.)(?P<expert>\d+)(?P<tail>\.w[123])$")
_DSV4_GROUPED_EXPERT_OFT_PARAM_RE = re.compile(r"^(?P<base>.*\.mlp\.(?P<proj>w[123]))_oft_r$")
_DSV4_GROUPED_EXPERT_OFT_BASE_PREFIX_RE = re.compile(r"^(?P<base>.*\.mlp\.(?P<proj>w[123]))$")
_DSV4_GROUPED_EXPERT_OFT_LAYER_RE = re.compile(r"(?:^|\.)decoder\.layers\.(?P<layer>\d+)\.mlp\.(?P<proj>w[123])$")


def _infer_oft_block_size_from_n_elements(n_elements: int) -> int:
    """Infer OFT block size from compact skew-vector length."""
    discriminant = 1 + 8 * n_elements
    root = math.isqrt(discriminant)
    block_size = (1 + root) // 2
    if root * root != discriminant or block_size * (block_size - 1) // 2 != n_elements:
        raise ValueError(f"Cannot infer OFT block size from n_elements={n_elements}")
    return block_size


def _is_dsv4_native_expert_oft_base_prefix(global_base_prefix: str) -> bool:
    """Return whether ``global_base_prefix`` is a native DSV4 per-expert OFT prefix."""

    return _DSV4_NATIVE_EXPERT_OFT_BASE_PREFIX_RE.match(global_base_prefix) is not None


def _globalize_dsv4_native_expert_oft_base_prefix(global_base_prefix: str, num_moe_experts: int) -> str:
    """Map EP-local native DSV4 expert prefixes to global expert IDs for HF export."""

    match = _DSV4_NATIVE_EXPERT_OFT_BASE_PREFIX_RE.match(global_base_prefix)
    if match is None:
        return global_base_prefix

    ep_size = parallel_state.get_expert_model_parallel_world_size()
    if ep_size <= 1:
        return global_base_prefix

    assert num_moe_experts % ep_size == 0, f"num_moe_experts={num_moe_experts} must be divisible by ep_size={ep_size}"
    num_experts_per_rank = num_moe_experts // ep_size
    expert_id = int(match.group("expert"))
    if expert_id >= num_experts_per_rank:
        return global_base_prefix

    global_expert_id = parallel_state.get_expert_model_parallel_rank() * num_experts_per_rank + expert_id
    return f"{match.group('head')}{global_expert_id}{match.group('tail')}"


@dataclass(frozen=True)
class AdapterWeightConversionTask:
    """Task describing an adapter's LoRA weights for conversion or merging."""

    global_base_prefix: str
    adapter_key: Optional[str]
    alpha: int
    dim: int
    linear_in_task: "WeightConversionTask"
    linear_out_task: "WeightConversionTask"


@dataclass(frozen=True)
class AdapterWeight:
    """Materialized adapter weights ready for merge."""

    global_base_prefix: str
    adapter_key: Optional[str]
    alpha: int
    dim: int
    linear_in_weight: "MegatronWeightTuple"
    linear_out_weight: "MegatronWeightTuple"


@dataclass(frozen=True)
class OFTAdapterConversionTask:
    """Task describing an OFT adapter's weight for conversion or merging.

    ``slice_name`` is set for CanonicalOFT split wrappers (one of ``"q"``,
    ``"k"``, ``"v"``, ``"gate"``, ``"up"``) and identifies which sub-adapter's
    ``oft_r`` this task references. ``None`` means a legacy shared-R adapter
    (one R that maps to multiple HF sub-projections at emit time).
    """

    global_base_prefix: str
    local_base_prefix: str
    is_expert: bool
    input_is_parallel: bool
    block_size: int
    r: int
    block_share: bool
    pp_rank: int
    vp_stage: int
    slice_name: Optional[str] = None
    # Set when the slice adapter is a per-expert ``nn.ModuleList`` (grouped split FC1):
    # the task owns one expert's rotation and emits a single per-expert HF name.
    expert_idx: Optional[int] = None


_CANONICAL_OFT_SLICE_TO_HF_LEAF = {
    "q": "q_proj",
    "k": "k_proj",
    "v": "v_proj",
    "gate": "gate_proj",
    "up": "up_proj",
}

_CANONICAL_OFT_SLICE_TO_CHILD_WRAPPER = {
    "q": "._qkv",
    "k": "._qkv",
    "v": "._qkv",
    "gate": "._fc1",
    "up": "._fc1",
}

_CANONICAL_OFT_SLICE_SORT_ORDER = {
    slice_name: index
    for index, slice_name in enumerate(("q", "k", "v", "gate", "up"))
}


def _canonical_oft_export_base_prefix(global_base_prefix: str, slice_name: Optional[str]) -> str:
    if slice_name is None:
        return global_base_prefix
    child_suffix = _CANONICAL_OFT_SLICE_TO_CHILD_WRAPPER[slice_name]
    if global_base_prefix.endswith(child_suffix):
        return global_base_prefix[: -len(child_suffix)]
    return global_base_prefix


def _oft_adapter_info_sort_key(info: tuple) -> tuple:
    """Stable ordering for OFT export tasks that enter distributed collectives.

    CanonicalOFT split siblings share the same base prefix and tensor shape
    (for example grouped-MoE ``adapter_gate`` and ``adapter_up``). Sorting only
    by base prefix leaves their relative order to ``set`` iteration, so
    different ranks can enter all-gathers in different slice order and mix gate
    tensors from one EP rank with up tensors from another. Include the slice in
    the key so every rank executes collectives in the same order.
    """

    slice_name = info[9]
    expert_idx = info[10] if len(info) > 10 else None
    slice_order = _CANONICAL_OFT_SLICE_SORT_ORDER.get(slice_name, -1)
    expert_order = -1 if expert_idx is None else int(expert_idx)
    return (
        extract_sort_key(info[0]),
        slice_order,
        expert_order,
        info[1],
        info[7],
        info[8],
    )


def _select_hf_base_param_name(base_mapping, adapter_key: Optional[str], expected_suffix: str) -> Optional[str]:
    """Return the HF base parameter name associated with this adapter."""

    hf_param = base_mapping.hf_param
    if isinstance(hf_param, str):
        return hf_param if hf_param.endswith(expected_suffix) or expected_suffix == ".weight" else None

    if isinstance(hf_param, dict):
        if adapter_key:
            target_suffix = ADAPTER_KEY_TO_SUFFIX.get(adapter_key)
            if target_suffix:
                for value in hf_param.values():
                    if value.endswith(target_suffix):
                        return value

        # For fused qkv/gate_up case, we just need a placeholder here
        value = next(iter(hf_param.values()))
        return value if value.endswith(expected_suffix) or expected_suffix == ".weight" else None

    return None


class MegatronPeftBridge:
    """Mixin providing adapter-aware utilities for Megatron model bridges."""

    def _get_lora_unwrapped_name(self, megatron_param: str) -> str:
        """Remove `.to_wrap` (LoRA) or `._orig_module` (OFT) from PEFT parameter names."""
        return megatron_param.replace(".to_wrap.", ".").replace("._orig_module.", ".")

    def _is_adapter_param_name(self, param_name: str) -> bool:
        """Return True if the parameter only belongs to a PEFT adapter.

        Matches both the legacy ``.adapter.`` form and the CanonicalOFT split
        forms ``.adapter_q.`` / ``.adapter_k.`` / ``.adapter_v.`` /
        ``.adapter_gate.`` / ``.adapter_up.``.
        """
        if ".adapter." in param_name:
            return True
        for slice_name in _CANONICAL_OFT_SLICE_TO_HF_LEAF:
            if f".adapter_{slice_name}." in param_name:
                return True
        return False

    @staticmethod
    def _parse_canonical_oft_slice(param_name: str) -> Optional[str]:
        """If ``param_name`` is a CanonicalOFT split-adapter param, return its
        slice name (``"q"`` / ``"k"`` / ``"v"`` / ``"gate"`` / ``"up"``).
        Otherwise return ``None``."""
        for slice_name in _CANONICAL_OFT_SLICE_TO_HF_LEAF:
            if f".adapter_{slice_name}." in param_name:
                return slice_name
        return None

    def _get_adapter_wrap_module(
        self,
        local_base_prefix: str,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        vp_stage: int,
    ) -> tuple[Optional[torch.nn.Module], Optional[torch.nn.Module]]:
        """Locate the adapter wrapper and its underlying module."""

        lora_module, _ = get_module_and_param_from_name(megatron_model, local_base_prefix, vp_stage)
        adapter = getattr(lora_module, "adapter", None)
        if adapter is None:
            lora_module, _ = get_module_and_param_from_name(megatron_model, local_base_prefix + ".to_wrap", vp_stage)
        return getattr(lora_module, "adapter", None), getattr(lora_module, "to_wrap", None)

    def _resolve_hf_adapter_param_name(
        self,
        mapping_registry: "MegatronMappingRegistry",
        global_base_prefix: str,
        megatron_adapter_suffix: str,
        base_suffix: str,
        adapter_key: Optional[str],
    ) -> Optional[str]:
        """
        Resolve the HuggingFace adapter parameter name by translating the base Megatron name.

        Note:
            LoRA adapters never register bias tensors for `linear_in` / `linear_out`, so callers
            only pass weight suffixes here. The bias fallback below is solely for robustness in
            case a future adapter type introduces biased projections.
        """

        hf_suffix = MEGATRON_TO_HF_LORA_SUFFIX.get(megatron_adapter_suffix)
        assert hf_suffix is not None, (
            f"Unsupported adapter suffix '{megatron_adapter_suffix}'. Update MEGATRON_TO_HF_LORA_SUFFIX."
        )

        base_mapping = mapping_registry.megatron_to_hf_lookup(f"{global_base_prefix}{base_suffix}")
        assert base_mapping is not None, (
            f"Expected mapping for adapter base '{global_base_prefix}{base_suffix}' but none found"
        )

        # Strip expert layers numbering
        base_suffix = base_suffix.rstrip(digits)
        hf_base_name = _select_hf_base_param_name(base_mapping, adapter_key, base_suffix)
        if hf_base_name is None:
            return None

        if hf_base_name.endswith(base_suffix):
            return hf_base_name[: -len(base_suffix)] + hf_suffix

        # Some HF base names (e.g., Qwen3.5 MoE expert gate_up_proj / down_proj)
        # don't include a trailing ".weight". Allow LoRA suffix to be appended directly.
        if base_suffix == ".weight":
            return hf_base_name + hf_suffix

    def _get_base_hf_param_names_for_adapter(
        self,
        mapping_registry: "MegatronMappingRegistry",
        global_base_prefix: str,
        adapter_key: Optional[str],
        base_suffix: str,
    ) -> List[str]:
        """Return all HF base parameter names associated with this adapter."""

        base_mapping = mapping_registry.megatron_to_hf_lookup(f"{global_base_prefix}{base_suffix}")
        if base_mapping is None:
            return []

        hf_param = base_mapping.hf_param
        if isinstance(hf_param, str):
            return [hf_param]

        values = list(hf_param.values())
        if adapter_key:
            adapter_suffix = ADAPTER_KEY_TO_SUFFIX.get(adapter_key)
            if adapter_suffix:
                filtered = [value for value in values if value.endswith(adapter_suffix)]
                if filtered:
                    return filtered
        return values

    def _make_lora_param_name(self, base_name: str, megatron_adapter_suffix: str) -> Optional[str]:
        """Translate a base HF weight name into its LoRA-specific counterpart."""

        hf_suffix = MEGATRON_TO_HF_LORA_SUFFIX.get(megatron_adapter_suffix)
        if hf_suffix is None:
            return None

        if base_name.endswith(".weight"):
            return base_name[: -len(".weight")] + hf_suffix

        # Some HF base names (e.g., Qwen3.5 MoE expert gate_up_proj) omit ".weight".
        return base_name + hf_suffix

    def _make_oft_param_name(self, base_name: str, megatron_oft_suffix: str = ".oft_r") -> Optional[str]:
        """Translate a base HF weight name into its OFT-specific counterpart.

        Example:
            ``model.layers.0.self_attn.q_proj.weight`` → ``model.layers.0.self_attn.q_proj.oft_R``
        """
        if not base_name.endswith(".weight"):
            return None
        hf_suffix = MEGATRON_TO_HF_OFT_SUFFIX.get(megatron_oft_suffix)
        if hf_suffix is None:
            return None
        return base_name[: -len(".weight")] + hf_suffix

    def _is_fused_qkv(self, hf_weight_names: Iterable[str]) -> bool:
        """Check whether the provided HF names correspond to a fused QKV weight."""

        names = list(hf_weight_names)
        if len(names) != 3:
            return False

        required = {"q_proj", "k_proj", "v_proj"}
        discovered = {token for name in names for token in required if token in name}
        return discovered == required

    def _is_gdn_in_proj_split(self, hf_weight_names: Iterable[str]) -> bool:
        """Check whether the provided HF names correspond to split GDN in_proj weights."""

        names = list(hf_weight_names)
        if len(names) != 4:
            return False
        required = set(GDN_IN_PROJ_KEYS)
        discovered = {token for name in names for token in required if token in name}
        return discovered == required and all("linear_attn" in name for name in names)

    def _is_fused_fc1_gate_up(
        self,
        base_hf_weight_names: Iterable[str],
        linear_out_tensor: torch.Tensor,
        base_weight_shape: Optional[torch.Size] = None,
    ) -> bool:
        """Detect fused FC1 (gate/up) adapters based on names and tensor shape."""

        names = list(base_hf_weight_names)
        has_gate_up = (
            bool(names)
            and len(names) % 2 == 0
            and all(("gate_proj" in name or "up_proj" in name) for name in names)
            and any("gate_proj" in name for name in names)
            and any("up_proj" in name for name in names)
        )
        if not has_gate_up:
            return False

        if linear_out_tensor.ndim != 2 or linear_out_tensor.shape[0] % 2 != 0:
            return False

        if base_weight_shape is not None and linear_out_tensor.shape[0] != 2 * base_weight_shape[0]:
            return False

        return True

    def _infer_qkv_projection_from_name(self, hf_name: str) -> Optional[str]:
        """Return q_proj/k_proj/v_proj identifier based on the HF name."""

        if "q_proj" in hf_name:
            return "q_proj"
        if "k_proj" in hf_name:
            return "k_proj"
        if "v_proj" in hf_name:
            return "v_proj"
        return None

    def _infer_gdn_in_proj_projection_from_name(self, hf_name: str) -> Optional[str]:
        """Return in_proj_qkv/z/b/a identifier based on the HF name."""

        for projection_key in GDN_IN_PROJ_KEYS:
            if projection_key in hf_name:
                return projection_key
        return None

    def _infer_hf_expert_idx(self, hf_name: str) -> Optional[int]:
        """Return the expert index embedded in an HF MoE weight name."""

        match = re.search(r"\bexperts\.(\d+)\b", hf_name)
        if match is None:
            return None
        return int(match.group(1))

    def _split_qkv_linear_out_weight(
        self,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        linear_out_weight: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Split a fused LoRA linear_out tensor for QKV adapters."""

        model = megatron_model[0] if isinstance(megatron_model, list) else megatron_model
        # Pass the LoRA rank as feature_dim so split_qkv_weights doesn't
        # mistake it for an FP8 compressed hidden_size.
        feature_dim = linear_out_weight.shape[-1] if linear_out_weight.ndim == 2 else None
        q_out, k_out, v_out = split_qkv_weights(model.config, linear_out_weight, feature_dim=feature_dim)
        return {"q_proj": q_out, "k_proj": k_out, "v_proj": v_out}

    def _split_gdn_in_proj_linear_out_weight(
        self,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        linear_out_weight: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Split a fused LoRA linear_out tensor for GDN in_proj adapters."""

        model = megatron_model[0] if isinstance(megatron_model, list) else megatron_model
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        feature_dim = linear_out_weight.shape[1]
        qkvz, ba = split_gdn_linear_weights(
            model.config,
            linear_out_weight,
            tp_size=tp_size,
            feature_dim=feature_dim,
        )
        qkv, z, b, a = _split_gdn_grouped_to_separate(model.config, qkvz, ba, feature_dim=feature_dim)
        return {"in_proj_qkv": qkv, "in_proj_z": z, "in_proj_b": b, "in_proj_a": a}

    def _build_lora_hf_names(self, base_hf_weight_names: List[str]) -> tuple[List[str], List[str]]:
        """Build LoRA A/B names for a list of HF base parameter names."""

        linear_in_hf_names = [
            self._make_lora_param_name(base_name, ".linear_in.weight") for base_name in base_hf_weight_names
        ]
        linear_out_hf_names = [
            self._make_lora_param_name(base_name, ".linear_out.weight") for base_name in base_hf_weight_names
        ]
        return linear_in_hf_names, linear_out_hf_names

    def _collect_packed_expert_adapter_tensors(
        self,
        linear_in_tensor: torch.Tensor,
        linear_out_tensor: torch.Tensor,
        expert_linear_in_gathered: Optional[List[torch.Tensor]],
        expert_linear_out_gathered: Optional[List[torch.Tensor]],
        num_moe_experts: int,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Collect one LoRA A/B tensor per expert for grouped expert exports."""

        per_expert_linear_in: List[torch.Tensor] = []
        per_expert_linear_out: List[torch.Tensor] = []
        if linear_in_tensor.ndim > 2 or linear_out_tensor.ndim > 2:
            # Already carries local expert dim; concatenate across EP ranks if needed.
            linear_in_all = (
                torch.cat(expert_linear_in_gathered, dim=0)
                if expert_linear_in_gathered is not None
                else linear_in_tensor
            )
            linear_out_all = (
                torch.cat(expert_linear_out_gathered, dim=0)
                if expert_linear_out_gathered is not None
                else linear_out_tensor
            )
            per_expert_linear_in = list(linear_in_all)
            per_expert_linear_out = list(linear_out_all)
            return per_expert_linear_in, per_expert_linear_out

        for expert_idx in range(num_moe_experts):
            per_expert_linear_in.append(
                self._select_expert_adapter_weight(
                    linear_in_tensor,
                    expert_linear_in_gathered,
                    expert_idx,
                    num_moe_experts,
                )
            )
            per_expert_linear_out.append(
                self._select_expert_adapter_weight(
                    linear_out_tensor,
                    expert_linear_out_gathered,
                    expert_idx,
                    num_moe_experts,
                )
            )
        return per_expert_linear_in, per_expert_linear_out

    def _build_packed_expert_linear_out_by_base(
        self,
        megatron_model: List[MegatronModel],
        base_hf_weight_names: List[str],
        per_expert_linear_out: List[torch.Tensor],
        is_expert: bool,
    ) -> Dict[str, torch.Tensor]:
        """Build per-base stacked LoRA-B tensors for packed grouped-expert export."""

        if not per_expert_linear_out:
            return {}

        # Handle fused adapters (qkv/gate_up/gdn in_proj) by splitting per-expert then stacking.
        per_base_linear_out = self._get_fused_adapter_linear_out_slices(
            megatron_model,
            base_hf_weight_names,
            per_expert_linear_out[0],
            is_expert=is_expert,
        )
        if per_base_linear_out is None:
            stacked = torch.stack(per_expert_linear_out, dim=0)
            return {base_name: stacked for base_name in base_hf_weight_names}

        per_base_stacks: Dict[str, List[torch.Tensor]] = {name: [] for name in base_hf_weight_names}
        for expert_out in per_expert_linear_out:
            per_base = self._get_fused_adapter_linear_out_slices(
                megatron_model,
                base_hf_weight_names,
                expert_out,
                is_expert=is_expert,
            )
            assert per_base is not None, "Expected fused adapter split for expert LoRA"
            for base_name in base_hf_weight_names:
                per_base_stacks[base_name].append(per_base[base_name])

        return {base_name: torch.stack(parts, dim=0) for base_name, parts in per_base_stacks.items()}

    def _split_fused_fc1_linear_out_weight(
        self,
        linear_out_weight: torch.Tensor,
        *,
        is_expert: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split fused FC1 LoRA linear_out into gate/up with TP-aware ordering."""

        tp_size = (
            parallel_state.get_expert_tensor_parallel_world_size()
            if is_expert
            else parallel_state.get_tensor_model_parallel_world_size()
        )
        if tp_size <= 1:
            return torch.chunk(linear_out_weight, 2, dim=0)

        shard_size = linear_out_weight.shape[0] // tp_size
        if shard_size * tp_size != linear_out_weight.shape[0] or shard_size % 2 != 0:
            return torch.chunk(linear_out_weight, 2, dim=0)

        shards = torch.split(linear_out_weight, shard_size, dim=0)
        gate_parts = []
        up_parts = []
        for shard in shards:
            gate_shard, up_shard = torch.chunk(shard, 2, dim=0)
            gate_parts.append(gate_shard)
            up_parts.append(up_shard)
        gate = torch.cat(gate_parts, dim=0)
        up = torch.cat(up_parts, dim=0)
        return gate, up

    def _gather_expert_adapter_weight(
        self,
        weight: torch.Tensor,
    ) -> Optional[List[torch.Tensor]]:
        """Gather expert-sharded adapter weights across EP ranks when needed."""
        ep_size = parallel_state.get_expert_model_parallel_world_size()
        if ep_size <= 1:
            return None
        assert weight.ndim < 3

        gathered = [torch.empty_like(weight) for _ in range(ep_size)]
        torch.distributed.all_gather(gathered, weight, group=parallel_state.get_expert_model_parallel_group())
        return gathered

    def _gather_dsv4_grouped_expert_oft_weight(
        self,
        weight: torch.Tensor,
        *,
        proj: Optional[str] = None,
    ) -> torch.Tensor:
        """Gather grouped DSV4 compact OFT tensors into HF expert layout.

        DSV4 grouped ``w1``/``w3`` OFT rotates the full hidden input and is
        replicated across expert-TP ranks. ``w2`` rotates the row-parallel
        intermediate input, so its block dimension is expert-TP sharded and
        must be gathered before yielding full HF/SGLang expert tensors.
        """
        assert weight.ndim == 3

        if proj == "w2":
            etp_group = parallel_state.get_expert_tensor_parallel_group()
            etp_size = etp_group.size() if etp_group is not None else 1
            if etp_size > 1:
                gathered = [torch.empty_like(weight) for _ in range(etp_size)]
                torch.distributed.all_gather(gathered, weight, group=etp_group)
                weight = torch.cat(gathered, dim=1)

        ep_size = parallel_state.get_expert_model_parallel_world_size()
        if ep_size <= 1:
            return weight

        gathered = [torch.empty_like(weight) for _ in range(ep_size)]
        torch.distributed.all_gather(gathered, weight, group=parallel_state.get_expert_model_parallel_group())
        return torch.cat(gathered, dim=0)

    def _select_expert_adapter_weight(
        self,
        weight: torch.Tensor,
        gathered: List[torch.Tensor],
        expert_idx: int,
        num_experts: int,
    ) -> torch.Tensor:
        """Select the per-expert adapter weight slice if present."""

        assert weight.ndim < 3

        ep_size = parallel_state.get_expert_model_parallel_world_size()
        if ep_size <= 1:
            return weight

        num_experts_per_rank = num_experts // ep_size
        rank = expert_idx // num_experts_per_rank
        return gathered[rank]

    def _megatron_global_adapters_info_all_pp_ranks(
        self, megatron_model: Union[MegatronModel, List[MegatronModel]]
    ) -> List[tuple[str, str, bool, bool, int, int, int, int]]:
        """Get all adapters' information tuple:
         (global_base_name, local_base_prefix, input_is_parallel, base_linear_is_parallel, alpha, dim, pp_rank, vp_stage)
        across all pipeline parallel ranks."""
        # Cache the result after first call
        if hasattr(self, "_cached_param_objects_adapter"):
            return self._cached_param_objects_adapter

        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        from megatron.bridge.models.conversion.model_bridge import _megatron_local_name_to_global

        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pp_rank = get_pg_rank(pp_group)
        model_config = unwrap_model(megatron_model)[0].config
        global_param_objects: List[tuple[str, str, bool, bool, int, int, int, int]] = []

        for vp_stage, model in enumerate(megatron_model):
            for local_param_name, _ in itertools.chain(model.named_parameters(), persistent_buffers(model)):  # type: ignore[name-defined]
                if "_extra_state" in local_param_name:
                    continue
                local_param_name = self._unwrap_name(local_param_name)
                global_param_name = _megatron_local_name_to_global(
                    megatron_model, model_config, local_param_name, vp_stage
                )
                # only collect linear_in.weight for deduplication
                if not self._is_adapter_param_name(global_param_name) or not global_param_name.endswith(
                    ".linear_in.weight"
                ):
                    continue

                local_base_prefix = local_param_name.partition(".adapter.")[0]
                global_base_name = global_param_name[: -len(".linear_in.weight")]
                adapter, to_wrap = self._get_adapter_wrap_module(local_base_prefix, megatron_model, vp_stage)
                if isinstance(adapter, ModuleDict):
                    adapter_name = local_param_name.removeprefix(local_base_prefix + ".adapter.").split(".")[0]
                    adapter = adapter[adapter_name]
                if isinstance(adapter, ParallelLinearAdapter):
                    input_is_parallel = adapter.input_is_parallel
                    base_linear_is_parallel = True
                else:
                    attrs = get_adapter_attributes_from_linear(to_wrap)
                    input_is_parallel = attrs.input_is_parallel
                    base_linear_is_parallel = attrs.base_linear_is_parallel
                global_param_objects.append(
                    (
                        global_base_name,
                        local_base_prefix,
                        input_is_parallel,
                        base_linear_is_parallel,
                        adapter.alpha,
                        adapter.dim,
                        pp_rank,
                        vp_stage,
                    )
                )

        gathered_global_param_objects = [None] * pp_group.size()
        torch.distributed.all_gather_object(gathered_global_param_objects, global_param_objects, group=pp_group)

        # flatten the list, sort it and remove duplicates
        # the order matters here, casually re-order will cause a hang.
        flattened_names = list(set(sum(gathered_global_param_objects, [])))

        # the order cannot be changed, this sync for all ranks for conversion
        # change this might cause a hang
        gathered_global_param_objects = sorted(flattened_names, key=lambda x: extract_sort_key(x[0]))

        self._cached_param_objects_adapter = gathered_global_param_objects

        return gathered_global_param_objects

    def _construct_adapters_names(self, prefix: str, adapter_key: Optional[str]) -> tuple[str, str]:
        """Build linear_in/linear_out parameter names for an adapter.

        Args:
            prefix: Base module prefix without any adapter suffix (global or local, depending on caller).
            adapter_key: Optional adapter identifier used by CanonicalLoRA (e.g. ``adapter_q``). ``None`` for
                standard single-adapter LoRA modules.

        Returns:
            Tuple ``(linear_in_name, linear_out_name)`` containing the parameter names for the adapter's
            input and output projection weights.
        """
        linear_in_name, linear_out_name = prefix + ".adapter", prefix + ".adapter"
        if adapter_key is not None:
            linear_in_name += f".{adapter_key}"
            linear_out_name += f".{adapter_key}"
        linear_in_name += ".linear_in.weight"
        linear_out_name += ".linear_out.weight"
        return linear_in_name, linear_out_name

    def build_adapter_conversion_tasks(
        self, megatron_model: Union[MegatronModel, List[MegatronModel]]
    ) -> Dict[str, List[AdapterWeightConversionTask]]:
        """Construct adapter merge tasks keyed by their base parameter.

        The returned dict is keyed by the *global* LoRA-wrapped parameter name
        (e.g., ``decoder.layers.0.mlp.linear_fc1.to_wrap.weight``). Each value
        contains the adapter tasks (canonical or regular) that should be
        merged into that base weight.
        """

        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        adapters_info = self._megatron_global_adapters_info_all_pp_ranks(megatron_model)
        tasks_by_base: Dict[str, List[AdapterWeightConversionTask]] = defaultdict(list)  # type: ignore[name-defined]

        from megatron.bridge.models.conversion.model_bridge import WeightConversionTask

        # `MegatronModelBridge` mixes in this class and provides `mapping_registry`.
        assert hasattr(self, "mapping_registry"), "MegatronModelBridge must define mapping_registry"
        mapping_registry = self.mapping_registry()  # type: ignore[attr-defined]

        for (
            global_base_name,
            local_base_prefix,
            input_is_parallel,
            base_linear_is_parallel,
            alpha,
            dim,
            pp_rank,
            vp_stage,
        ) in adapters_info:
            # global_base_name example: decoder.layers.0.mlp.linear_fc1.adapter.adapter_q
            global_base_prefix, _, adapter_suffix = global_base_name.partition(".adapter")

            adapter_key = None
            if adapter_suffix:
                key_token = adapter_suffix.split(".")[-1]
                if key_token.startswith("adapter_"):
                    adapter_key = key_token

            global_linear_in_name, global_linear_out_name = self._construct_adapters_names(
                global_base_prefix, adapter_key
            )
            # In case the adapter doesn't exist locally, we use the global names
            local_linear_in_name, local_linear_out_name = global_linear_in_name, global_linear_out_name

            base_suffix = ".weight"
            if is_expert_linear(global_base_prefix) and ".local_experts." not in global_base_prefix:
                # To get expert layer hf mapping properly
                base_suffix = ".weight0"

            hf_linear_in_name = self._resolve_hf_adapter_param_name(
                mapping_registry, global_base_prefix, ".linear_in.weight", base_suffix, adapter_key
            )
            hf_linear_out_name = self._resolve_hf_adapter_param_name(
                mapping_registry, global_base_prefix, ".linear_out.weight", base_suffix, adapter_key
            )

            linear_in_module, linear_in_weight = None, None
            linear_out_module, linear_out_weight = None, None
            if parallel_state.get_pipeline_model_parallel_rank() == pp_rank:
                adapter, _ = self._get_adapter_wrap_module(local_base_prefix, megatron_model, vp_stage)
                if isinstance(adapter, ModuleDict):
                    adapter = adapter[adapter_key]
                linear_in_module, linear_in_weight = adapter.linear_in, adapter.linear_in.weight
                linear_out_module, linear_out_weight = adapter.linear_out, adapter.linear_out.weight
                local_linear_in_name, local_linear_out_name = self._construct_adapters_names(
                    local_base_prefix, adapter_key
                )

            # Pick mapping strategies based on base layer parallelism
            if base_linear_is_parallel:
                linear_in_mapping_cls = RowParallelMapping if input_is_parallel else ColumnParallelMapping
                linear_out_mapping_cls = ColumnParallelMapping
            else:
                linear_in_mapping_cls = ReplicatedMapping
                linear_out_mapping_cls = ReplicatedMapping

            linear_in_task = WeightConversionTask(
                param_name=local_linear_in_name,
                global_param_name=global_linear_in_name,
                mapping=linear_in_mapping_cls(
                    megatron_param=local_linear_in_name,
                    hf_param=hf_linear_in_name,
                ),
                pp_rank=pp_rank,
                vp_stage=vp_stage,
                megatron_module=linear_in_module,
                param_weight=linear_in_weight,
            )

            linear_out_task = WeightConversionTask(
                param_name=local_linear_out_name,
                global_param_name=global_linear_out_name,
                mapping=linear_out_mapping_cls(
                    megatron_param=local_linear_out_name,
                    hf_param=hf_linear_out_name,
                ),
                pp_rank=pp_rank,
                vp_stage=vp_stage,
                megatron_module=linear_out_module,
                param_weight=linear_out_weight,
            )

            tasks_by_base[global_base_prefix].append(
                AdapterWeightConversionTask(
                    global_base_prefix=global_base_prefix,
                    adapter_key=adapter_key,
                    alpha=alpha,
                    dim=dim,
                    linear_in_task=linear_in_task,
                    linear_out_task=linear_out_task,
                )
            )

        return tasks_by_base

    def materialize_adapter_weights(self, adapter_tasks: List[AdapterWeightConversionTask]) -> List[AdapterWeight]:
        """Run adapter merge tasks to gather full adapter weights."""

        from megatron.bridge.models.conversion.model_bridge import MegatronWeightTuple

        materialized: List[AdapterWeight] = []
        for adapter_task in adapter_tasks:
            linear_in_dict = adapter_task.linear_in_task.mapping.megatron_to_hf(
                adapter_task.linear_in_task.param_weight, adapter_task.linear_in_task.megatron_module
            )
            linear_in_tensor = next(iter(linear_in_dict.values()))

            linear_out_dict = adapter_task.linear_out_task.mapping.megatron_to_hf(
                adapter_task.linear_out_task.param_weight, adapter_task.linear_out_task.megatron_module
            )
            linear_out_tensor = next(iter(linear_out_dict.values()))

            materialized.append(
                AdapterWeight(
                    global_base_prefix=adapter_task.global_base_prefix,
                    adapter_key=adapter_task.adapter_key,
                    alpha=adapter_task.alpha,
                    dim=adapter_task.dim,
                    linear_in_weight=MegatronWeightTuple(
                        adapter_task.linear_in_task.param_name,
                        linear_in_tensor,
                        adapter_task.linear_in_task.vp_stage,
                    ),
                    linear_out_weight=MegatronWeightTuple(
                        adapter_task.linear_out_task.param_name,
                        linear_out_tensor,
                        adapter_task.linear_out_task.vp_stage,
                    ),
                )
            )

        return materialized

    def stream_adapter_weights_megatron_to_hf(
        self,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        cpu: bool = True,
        show_progress: bool = True,
    ) -> Iterable[HFWeightTuple]:
        """Stream only adapter weights without merging them into base tensors."""

        # Local import avoids circular dependency while ensuring runtime access.
        from megatron.bridge.models.conversion.model_bridge import HFWeightTuple

        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        num_moe_experts = megatron_model[0].config.num_moe_experts
        adapter_tasks_by_base = self.build_adapter_conversion_tasks(megatron_model)
        adapter_tasks = list(itertools.chain.from_iterable(adapter_tasks_by_base.values()))
        if not adapter_tasks:
            return

        assert hasattr(self, "mapping_registry"), "MegatronModelBridge must define mapping_registry"
        mapping_registry = self.mapping_registry()  # type: ignore[attr-defined]

        for adapter_task in self._with_progress_tracking(adapter_tasks, "Streaming adapter weights", show_progress):
            adapter_weight = self.materialize_adapter_weights([adapter_task])[0]

            linear_in_tensor = adapter_weight.linear_in_weight.weight
            linear_out_tensor = adapter_weight.linear_out_weight.weight
            is_expert = is_expert_linear(adapter_task.global_base_prefix)
            is_grouped_expert = is_expert and ".local_experts." not in adapter_task.global_base_prefix
            expert_linear_in_gathered = None
            expert_linear_out_gathered = None
            if is_grouped_expert:
                expert_linear_in_gathered = self._gather_expert_adapter_weight(
                    linear_in_tensor,
                )
                expert_linear_out_gathered = self._gather_expert_adapter_weight(
                    linear_out_tensor,
                )

            base_suffixes = [".weight"]
            if is_grouped_expert:
                base_suffixes = [f".weight{expert_num}" for expert_num in range(num_moe_experts)]

            # If the HF base names don't include experts.N, emit packed expert weights
            # (stacked along dim 0) once per HF name instead of duplicating per expert.
            packed_expert = False
            base_hf_weight_names: List[str] = []
            if is_grouped_expert and base_suffixes:
                base_hf_weight_names = self._get_base_hf_param_names_for_adapter(
                    mapping_registry,
                    adapter_task.global_base_prefix,
                    adapter_task.adapter_key,
                    base_suffixes[0],
                )
                if base_hf_weight_names and not any(
                    re.search(r"experts\.(\d+)", name) for name in base_hf_weight_names
                ):
                    packed_expert = True

            if packed_expert:
                linear_in_hf_names, linear_out_hf_names = self._build_lora_hf_names(base_hf_weight_names)
                per_expert_linear_in, per_expert_linear_out = self._collect_packed_expert_adapter_tensors(
                    linear_in_tensor,
                    linear_out_tensor,
                    expert_linear_in_gathered,
                    expert_linear_out_gathered,
                    num_moe_experts,
                )

                if not per_expert_linear_in or not per_expert_linear_out:
                    raise ValueError(
                        f"Expected to find per-expert adapter weights for grouped expert "
                        f"linear layer but none found, global_base_prefix={adapter_task.global_base_prefix}"
                    )
                linear_in_stacked = torch.stack(per_expert_linear_in, dim=0)
                if cpu:
                    linear_in_stacked = linear_in_stacked.cpu()

                if adapter_task.adapter_key is None:
                    linear_out_by_base = self._build_packed_expert_linear_out_by_base(
                        megatron_model,
                        base_hf_weight_names,
                        per_expert_linear_out,
                        is_expert=is_expert_linear(adapter_task.global_base_prefix),
                    )
                else:
                    shared_linear_out = torch.stack(per_expert_linear_out, dim=0)
                    linear_out_by_base = {base_name: shared_linear_out for base_name in base_hf_weight_names}

                for index, base_name in enumerate(base_hf_weight_names):
                    linear_out_stacked = linear_out_by_base[base_name]
                    if cpu:
                        linear_out_stacked = linear_out_stacked.cpu()
                    yield HFWeightTuple(linear_in_hf_names[index], linear_in_stacked)
                    yield HFWeightTuple(linear_out_hf_names[index], linear_out_stacked)

                continue

            for base_suffix in base_suffixes:
                current_linear_in_tensor = linear_in_tensor
                current_linear_out_tensor = linear_out_tensor
                if is_grouped_expert:
                    expert_idx = int(base_suffix[len(".weight") :])
                    current_linear_in_tensor = self._select_expert_adapter_weight(
                        linear_in_tensor,
                        expert_linear_in_gathered,
                        expert_idx,
                        num_moe_experts,
                    )
                    current_linear_out_tensor = self._select_expert_adapter_weight(
                        linear_out_tensor,
                        expert_linear_out_gathered,
                        expert_idx,
                        num_moe_experts,
                    )

                if cpu:
                    current_linear_in_tensor = current_linear_in_tensor.cpu()
                    current_linear_out_tensor = current_linear_out_tensor.cpu()

                base_hf_weight_names = self._get_base_hf_param_names_for_adapter(
                    mapping_registry,
                    adapter_task.global_base_prefix,
                    adapter_task.adapter_key,
                    base_suffix,
                )
                linear_in_hf_names, linear_out_hf_names = self._build_lora_hf_names(base_hf_weight_names)
                if adapter_task.adapter_key is None:
                    # Handle fused adapters (e.g., gate/up or q/k/v) by splitting the fused tensor
                    # into per-base slices keyed by the HF weight names.
                    # Example: base_hf_weight_names = ["...gate_proj.weight", "...up_proj.weight"]
                    per_base_linear_out = self._get_fused_adapter_linear_out_slices(
                        megatron_model,
                        base_hf_weight_names,
                        current_linear_out_tensor,
                        is_expert=is_expert_linear(adapter_task.global_base_prefix),
                    )
                    if per_base_linear_out is not None:
                        for index, base_name in enumerate(base_hf_weight_names):
                            current_linear_out_tensor = per_base_linear_out.get(base_name)
                            assert current_linear_out_tensor is not None, "unknown projection name"

                            yield HFWeightTuple(linear_in_hf_names[index], current_linear_in_tensor)
                            yield HFWeightTuple(linear_out_hf_names[index], current_linear_out_tensor)
                        continue

                yield HFWeightTuple(linear_in_hf_names[0], current_linear_in_tensor)
                yield HFWeightTuple(linear_out_hf_names[0], current_linear_out_tensor)

    def _get_fused_adapter_linear_out_slices(
        self,
        megatron_model: List[MegatronModel],
        base_hf_weight_names: List[str],
        linear_out_tensor: torch.Tensor,
        is_expert: bool = False,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Return per-base-name linear_out slices for fused adapters, else None.

        This supports fused QKV adapters (split into q/k/v) and fused FC1 adapters
        (split into gate/up along dim=0). The returned dict is keyed by the HF
        base weight name (e.g. `...q_proj.weight` or `...gate_proj.weight`).
        """

        if self._is_fused_qkv(base_hf_weight_names):
            qkv_linear_out_weights = self._split_qkv_linear_out_weight(megatron_model, linear_out_tensor)
            per_base: Dict[str, torch.Tensor] = {}
            for base_name in base_hf_weight_names:
                projection_key = self._infer_qkv_projection_from_name(base_name)
                if projection_key is None:
                    continue
                per_base[base_name] = qkv_linear_out_weights[projection_key]
            return per_base

        if self._is_gdn_in_proj_split(base_hf_weight_names):
            gdn_linear_out_weights = self._split_gdn_in_proj_linear_out_weight(megatron_model, linear_out_tensor)
            per_base = {}
            for base_name in base_hf_weight_names:
                projection_key = self._infer_gdn_in_proj_projection_from_name(base_name)
                if projection_key is None:
                    raise ValueError(f"Unknown GDN in_proj base weight name: {base_name}")
                per_base[base_name] = gdn_linear_out_weights[projection_key]
            return per_base

        is_fused_fc1 = self._is_fused_fc1_gate_up(base_hf_weight_names, linear_out_tensor)
        if is_fused_fc1:
            gate_weight, up_weight = self._split_fused_fc1_linear_out_weight(
                linear_out_tensor,
                is_expert=is_expert,
            )
            per_base = {}
            for base_name in base_hf_weight_names:
                if "gate_proj" in base_name:
                    per_base[base_name] = gate_weight
                elif "up_proj" in base_name:
                    per_base[base_name] = up_weight
                else:
                    raise ValueError(f"Unknown fused-fc1 base weight name: {base_name}")
            return per_base

        return None

    def _merge_lora_adapter_weights(
        self,
        megatron_model: List[MegatronModel],
        converted_weights_dict: Dict[str, torch.Tensor],
        adapter_weights: List[AdapterWeight],
    ) -> Dict[str, torch.Tensor]:
        """Merge LoRA adapter weights back into the base tensor for HF export."""

        if not converted_weights_dict:
            # Nothing to merge on this rank (e.g., non-owning PP rank or filtered mapping).
            return converted_weights_dict

        if len(adapter_weights) > 1 and all(
            w.adapter_key in ADAPTER_NAME_MAP.values() for w in adapter_weights if w.adapter_key
        ):
            return self._merge_canonical_adapter_from_weights(megatron_model, converted_weights_dict, adapter_weights)

        assert len(adapter_weights) == 1, "Expected a single adapter weight for standard LoRA merging"

        adapter_weight = adapter_weights[0]
        alpha, dim = adapter_weight.alpha, adapter_weight.dim
        linear_in_weight = adapter_weight.linear_in_weight.weight
        linear_out_weight = adapter_weight.linear_out_weight.weight
        num_moe_experts = megatron_model[0].config.num_moe_experts
        is_expert = is_expert_linear(adapter_weight.global_base_prefix)
        is_grouped_expert = is_expert and ".local_experts." not in adapter_weight.global_base_prefix
        expert_linear_in_gathered = None
        expert_linear_out_gathered = None
        if is_grouped_expert:
            expert_linear_in_gathered = self._gather_expert_adapter_weight(linear_in_weight)
            expert_linear_out_gathered = self._gather_expert_adapter_weight(linear_out_weight)

        base_weight = next(iter(converted_weights_dict.values()))
        base_weight_shape = base_weight.shape
        weight_names = converted_weights_dict.keys()
        if self._is_gdn_in_proj_split(weight_names):
            # GDN in_proj LoRA is defined on the fused Megatron tensor; split it into
            # the four HF tensors (qkv/z/b/a) before merging.
            config = unwrap_model(megatron_model)[0].config
            hidden_size = config.hidden_size
            qk_dim = config.linear_key_head_dim * config.linear_num_key_heads
            v_dim = config.linear_value_head_dim * config.linear_num_value_heads
            num_v_heads = config.linear_num_value_heads
            fused_dim0 = 2 * qk_dim + 2 * v_dim + 2 * num_v_heads

            base_device = base_weight.device
            linear_out_on_base = (
                linear_out_weight if linear_out_weight.device == base_device else linear_out_weight.to(base_device)
            )
            linear_in_on_base = (
                linear_in_weight if linear_in_weight.device == base_device else linear_in_weight.to(base_device)
            )
            dummy_base = torch.zeros((fused_dim0, hidden_size), device=base_device, dtype=base_weight.dtype)
            lora_weight = LoRAMerge().merge(dummy_base, linear_out_on_base, linear_in_on_base, alpha, dim)

            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            qkvz, ba = split_gdn_linear_weights(config, lora_weight, tp_size=tp_size)
            qkv, z, b, a = _split_gdn_grouped_to_separate(config, qkvz, ba)
            gdn_slices = {"in_proj_qkv": qkv, "in_proj_z": z, "in_proj_b": b, "in_proj_a": a}

            for hf_name, base_tensor in list(converted_weights_dict.items()):
                projection_key = self._infer_gdn_in_proj_projection_from_name(hf_name)
                if projection_key is None:
                    raise ValueError(f"Unknown GDN in_proj weight name: {hf_name}")
                converted_weights_dict[hf_name] = base_tensor + gdn_slices[projection_key]

            return converted_weights_dict
        is_fused_fc1 = self._is_fused_fc1_gate_up(weight_names, linear_out_weight, base_weight_shape)
        is_fused_qkv = self._is_fused_qkv(weight_names) and not is_expert
        qkv_linear_out_weights = (
            self._split_qkv_linear_out_weight(megatron_model, linear_out_weight) if is_fused_qkv else None
        )
        fc1_gate_weight = fc1_up_weight = None
        if is_fused_fc1 and not is_expert:
            fc1_gate_weight, fc1_up_weight = self._split_fused_fc1_linear_out_weight(
                linear_out_weight,
                is_expert=is_expert,
            )

        for hf_name, base_weight in list(converted_weights_dict.items()):
            current_linear_in_weight = linear_in_weight
            current_linear_out_weight = linear_out_weight
            if is_grouped_expert:
                expert_idx = self._infer_hf_expert_idx(hf_name)
                if expert_idx is not None:
                    current_linear_in_weight = self._select_expert_adapter_weight(
                        linear_in_weight,
                        expert_linear_in_gathered,
                        expert_idx,
                        num_moe_experts,
                    )
                    current_linear_out_weight = self._select_expert_adapter_weight(
                        linear_out_weight,
                        expert_linear_out_gathered,
                        expert_idx,
                        num_moe_experts,
                    )
            if is_fused_fc1:
                if is_expert:
                    fc1_gate_weight, fc1_up_weight = self._split_fused_fc1_linear_out_weight(
                        current_linear_out_weight,
                        is_expert=is_expert,
                    )
                if "gate_proj" in hf_name:
                    current_linear_out_weight = fc1_gate_weight
                elif "up_proj" in hf_name:
                    current_linear_out_weight = fc1_up_weight
                else:
                    raise ValueError(f"Unknown weight name: {hf_name}")
            elif is_fused_qkv and qkv_linear_out_weights is not None:
                projection_key = self._infer_qkv_projection_from_name(hf_name)
                if projection_key is None:
                    raise ValueError(f"Unknown weight name: {hf_name}")
                current_linear_out_weight = qkv_linear_out_weights[projection_key]

            merged_weight = self._merge_single_adapter_weight(
                base_weight, alpha, dim, current_linear_in_weight, current_linear_out_weight
            )
            converted_weights_dict[hf_name] = merged_weight

        return converted_weights_dict

    def _merge_single_adapter_weight(
        self,
        base_weight: torch.Tensor,
        alpha: int,
        dim: int,
        linear_in_weight: torch.Tensor,
        linear_out_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Merge a single adapter's weights with base weight.

        The merge is performed in float32 to avoid precision loss from
        bfloat16 matmul (adapter weights are often stored in bf16).
        The result is cast back to the original base weight dtype.
        """

        orig_dtype = base_weight.dtype
        merger = LoRAMerge()
        base_device = base_weight.device
        linear_out_on_base = linear_out_weight.to(device=base_device, dtype=torch.float32)
        linear_in_on_base = linear_in_weight.to(device=base_device, dtype=torch.float32)
        merged = merger.merge(
            base_weight.float(),
            linear_out_on_base,
            linear_in_on_base,
            alpha,
            dim,
        )
        return merged.to(orig_dtype)

    def _merge_canonical_adapter_from_weights(
        self,
        megatron_model: List[MegatronModel],
        converted_weights_dict: Dict[str, torch.Tensor],
        adapter_weights: List[AdapterWeight],
    ) -> Dict[str, torch.Tensor]:
        """Merge CanonicalLoRA adapters using pre-materialized adapter weights."""

        adapter_lookup = {aw.adapter_key: aw for aw in adapter_weights}
        expert_linear_in_gathered: Dict[str, List[torch.Tensor]] = {}
        expert_linear_out_gathered: Dict[str, List[torch.Tensor]] = {}
        base_prefix = adapter_weights[0].global_base_prefix
        num_moe_experts = megatron_model[0].config.num_moe_experts
        is_expert = is_expert_linear(base_prefix)
        is_grouped_expert = is_expert and ".local_experts." not in base_prefix
        if is_grouped_expert:
            for adapter_key, adapter_weight in adapter_lookup.items():
                expert_linear_in_gathered[adapter_key] = self._gather_expert_adapter_weight(
                    adapter_weight.linear_in_weight.weight,
                )
                expert_linear_out_gathered[adapter_key] = self._gather_expert_adapter_weight(
                    adapter_weight.linear_out_weight.weight,
                )

        for hf_name, base_weight in converted_weights_dict.items():
            target_adapter = None
            target_adapter_key = None
            for suffix, adapter_key in ADAPTER_NAME_MAP.items():
                if hf_name.endswith(suffix):
                    target_adapter = adapter_lookup.get(adapter_key)
                    target_adapter_key = adapter_key
                    break

            if target_adapter is None:
                raise ValueError(f"Adapter name mapping not found for {hf_name}")

            linear_in_weight = target_adapter.linear_in_weight.weight
            linear_out_weight = target_adapter.linear_out_weight.weight
            if is_grouped_expert:
                expert_idx = self._infer_hf_expert_idx(hf_name)
                if expert_idx is not None:
                    linear_in_weight = self._select_expert_adapter_weight(
                        linear_in_weight,
                        expert_linear_in_gathered.get(target_adapter_key),
                        expert_idx,
                        num_moe_experts,
                    )
                    linear_out_weight = self._select_expert_adapter_weight(
                        linear_out_weight,
                        expert_linear_out_gathered.get(target_adapter_key),
                        expert_idx,
                        num_moe_experts,
                    )

            merged_weight = self._merge_single_adapter_weight(
                base_weight,
                target_adapter.alpha,
                target_adapter.dim,
                linear_in_weight,
                linear_out_weight,
            )
            converted_weights_dict[hf_name] = merged_weight

        return converted_weights_dict

    # -------------------------------------------------------------------------
    # OFT adapter conversion
    # -------------------------------------------------------------------------

    def _get_oft_adapter_wrap_module(
        self,
        local_base_prefix: str,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        vp_stage: int,
        slice_name: Optional[str] = None,
        expert_idx: Optional[int] = None,
    ) -> Optional[torch.nn.Module]:
        """Locate the OFTRotationModule adapter for a given base prefix.

        ``get_module_and_param_from_name`` returns ``(parent_module, target_attr)``
        where *target_attr* is the OFTLinear (or TEOFTLayerNormColumnParallelLinear)
        wrapper that holds the ``.adapter`` attribute.

        For CanonicalOFT split wrappers (``OFTLinearSplitQKV`` /
        ``OFTLinearSplitFC1UpGate``) the rotation modules live at
        ``adapter_q`` / ``adapter_k`` / ``adapter_v`` / ``adapter_gate`` /
        ``adapter_up``; ``slice_name`` selects which one to return.
        """

        from megatron.bridge.peft.canonical_oft import GroupedOFTRotation
        from megatron.bridge.peft.oft_layers import OFTRotationModule

        _, wrapper = get_module_and_param_from_name(megatron_model, local_base_prefix, vp_stage)
        if slice_name is not None:
            sub = getattr(wrapper, f"adapter_{slice_name}", None)
            # OFTLinearGroupedSplitFC1UpGate now stores ``adapter_gate`` /
            # ``adapter_up`` as a single ``GroupedOFTRotation`` whose ``oft_r``
            # is 3D ``(num_local_experts, num_blocks, n_elements)``. When the
            # task carries ``expert_idx`` (legacy per-expert layout), index into
            # the module list; otherwise return the grouped container itself.
            if expert_idx is not None and not isinstance(sub, OFTRotationModule):
                try:
                    sub = sub[expert_idx]
                except (TypeError, IndexError):
                    return None
            if isinstance(sub, (OFTRotationModule, GroupedOFTRotation)):
                return sub
            return None
        adapter = getattr(wrapper, "adapter", None)
        if isinstance(adapter, OFTRotationModule):
            return adapter
        return None

    def _megatron_global_oft_adapters_info_all_pp_ranks(
        self, megatron_model: Union[MegatronModel, List[MegatronModel]]
    ) -> List[OFTAdapterConversionTask]:
        """Collect OFT adapter information across all pipeline parallel ranks.

        Returns a sorted, deduplicated list of :class:`OFTAdapterConversionTask` entries,
        one per adapted module across the entire model.
        """

        if hasattr(self, "_cached_oft_adapter_info"):
            return self._cached_oft_adapter_info

        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        from megatron.bridge.models.conversion.model_bridge import _megatron_local_name_to_global

        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pp_rank = get_pg_rank(pp_group)
        model_config = unwrap_model(megatron_model)[0].config
        local_info: List[tuple] = []

        for vp_stage, model in enumerate(megatron_model):
            for local_param_name, param in itertools.chain(model.named_parameters(), persistent_buffers(model)):
                if "_extra_state" in local_param_name:
                    continue
                local_param_name = self._unwrap_name(local_param_name)
                global_param_name = _megatron_local_name_to_global(
                    megatron_model, model_config, local_param_name, vp_stage
                )

                dsv4_grouped_match = _DSV4_GROUPED_EXPERT_OFT_PARAM_RE.match(global_param_name)
                if dsv4_grouped_match is not None:
                    dsv4_grouped_global_base_prefix = dsv4_grouped_match.group("base")
                    local_base_prefix = local_param_name[: -len("_oft_r")]
                    if param.dim() != 3:
                        raise ValueError(
                            f"Expected grouped DSV4 OFT param {global_param_name} to be 3D, got {tuple(param.shape)}"
                        )
                    local_info.append((
                        dsv4_grouped_global_base_prefix,
                        local_base_prefix,
                        True,
                        False,
                        _infer_oft_block_size_from_n_elements(int(param.shape[-1])),
                        int(param.shape[1]),
                        False,
                        pp_rank,
                        vp_stage,
                        None,
                        None,
                    ))
                    continue

                if not self._is_adapter_param_name(global_param_name) or not global_param_name.endswith(".oft_r"):
                    continue

                slice_name = self._parse_canonical_oft_slice(global_param_name)
                expert_idx: Optional[int] = None
                if slice_name is not None:
                    sub_token = f".adapter_{slice_name}."
                    local_base_prefix = local_param_name.partition(sub_token)[0]
                    suffix_after_slice = global_param_name.split(sub_token, 1)[1]
                    grouped_match = re.match(r"^(\d+)\.oft_r$", suffix_after_slice)
                    if grouped_match is not None:
                        expert_idx = int(grouped_match.group(1))
                        strip = f".adapter_{slice_name}.{expert_idx}.oft_r"
                    else:
                        strip = f".adapter_{slice_name}.oft_r"
                    global_base_prefix = global_param_name[: -len(strip)]
                    global_base_prefix = _canonical_oft_export_base_prefix(global_base_prefix, slice_name)
                else:
                    local_base_prefix = local_param_name.partition(".adapter.")[0]
                    global_base_prefix = global_param_name[: -len(".adapter.oft_r")]

                adapter = self._get_oft_adapter_wrap_module(
                    local_base_prefix,
                    megatron_model,
                    vp_stage,
                    slice_name=slice_name,
                    expert_idx=expert_idx,
                )
                if adapter is None:
                    continue

                local_info.append((
                    global_base_prefix,
                    local_base_prefix,
                    adapter.is_expert,
                    adapter.input_is_parallel,
                    adapter.block_size,
                    adapter.r,
                    adapter.block_share,
                    pp_rank,
                    vp_stage,
                    slice_name,
                    expert_idx,
                ))

        gathered_info = [None] * pp_group.size()
        torch.distributed.all_gather_object(gathered_info, local_info, group=pp_group)

        flattened = list(set(sum(gathered_info, [])))
        sorted_info = sorted(flattened, key=_oft_adapter_info_sort_key)

        result = [
            OFTAdapterConversionTask(
                global_base_prefix=t[0],
                local_base_prefix=t[1],
                is_expert=t[2],
                input_is_parallel=t[3],
                block_size=t[4],
                r=t[5],
                block_share=t[6],
                pp_rank=t[7],
                vp_stage=t[8],
                slice_name=t[9],
                expert_idx=t[10],
            )
            for t in sorted_info
        ]

        self._cached_oft_adapter_info = result
        return result

    def _gather_oft_r_across_tp(
        self,
        oft_r: torch.Tensor,
        input_is_parallel: bool,
        block_share: bool,
        is_expert: bool,
    ) -> torch.Tensor:
        """All-gather TP-sharded ``oft_r`` blocks along dim 0 for RowParallel layers.

        For ColumnParallel base layers (``input_is_parallel=False``) or when
        ``block_share=True``, ``oft_r`` is already replicated and returned as-is.
        """

        if not input_is_parallel or block_share:
            return oft_r

        tp_group = (
            parallel_state.get_expert_tensor_parallel_group()
            if is_expert
            else parallel_state.get_tensor_model_parallel_group()
        )
        tp_size = tp_group.size()
        if tp_size <= 1:
            return oft_r

        gathered = [torch.empty_like(oft_r) for _ in range(tp_size)]
        torch.distributed.all_gather(gathered, oft_r, group=tp_group)
        return torch.cat(gathered, dim=0)

    def stream_oft_adapter_weights_megatron_to_hf(
        self,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        cpu: bool = True,
        show_progress: bool = True,
    ) -> Iterable["HFWeightTuple"]:
        """Stream OFT adapter weights (``oft_r``) from Megatron to HuggingFace format.

        Yields ``HFWeightTuple(hf_name, tensor)`` for each adapted layer. For fused
        Megatron layers (QKV, gate/up), the same ``oft_r`` is emitted once per
        sub-projection so that each HF module receives its own copy.

        The method handles:
        * **Tensor-parallel gathering** – ``oft_r`` blocks sharded across TP ranks
          for RowParallel base layers are concatenated along the blocks dimension.
        * **Pipeline-parallel broadcast** – the owning PP rank broadcasts the
          gathered ``oft_r`` to all other PP ranks.
        * **Expert-parallel gathering** – for grouped MoE experts, ``oft_r`` is
          collected from all EP ranks and the correct per-expert slice is selected.

        Args:
            megatron_model: The Megatron model (or list of VP chunks).
            cpu: Move tensors to CPU before yielding. Default ``True``.
            show_progress: Display a progress bar. Default ``True``.

        Yields:
            ``HFWeightTuple`` with the HF adapter parameter name (e.g.
            ``model.layers.0.self_attn.q_proj.oft_R``) and the corresponding tensor.
        """

        from megatron.bridge.models.conversion.model_bridge import HFWeightTuple

        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        num_moe_experts = megatron_model[0].config.num_moe_experts
        tasks = self._megatron_global_oft_adapters_info_all_pp_ranks(megatron_model)
        if not tasks:
            return

        assert hasattr(self, "mapping_registry"), "MegatronModelBridge must define mapping_registry"
        mapping_registry = self.mapping_registry()

        pp_group = parallel_state.get_pipeline_model_parallel_group()
        my_pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        pp_global_ranks = torch.distributed.get_process_group_ranks(pp_group)

        # Determine parameter dtype from model config for buffer allocation
        model_config = unwrap_model(megatron_model)[0].config
        if model_config.bf16:
            param_dtype = torch.bfloat16
        elif model_config.fp16:
            param_dtype = torch.float16
        else:
            param_dtype = torch.float32

        for task in self._with_progress_tracking(tasks, "Streaming OFT adapter weights", show_progress):
            n_elements = task.block_size * (task.block_size - 1) // 2
            num_blocks_local = 1 if task.block_share else task.r
            dsv4_grouped_match = _DSV4_GROUPED_EXPERT_OFT_BASE_PREFIX_RE.match(task.global_base_prefix)
            dsv4_grouped_expert_projection = (
                dsv4_grouped_match.group("proj") if dsv4_grouped_match is not None else None
            )
            is_dsv4_grouped_expert = dsv4_grouped_expert_projection is not None

            # ------------------------------------------------------------------
            # Step 1: obtain the full (TP-gathered) oft_r on the owning PP rank
            # ------------------------------------------------------------------
            if my_pp_rank == task.pp_rank:
                if is_dsv4_grouped_expert:
                    _, oft_r_param = get_module_and_param_from_name(
                        megatron_model, f"{task.local_base_prefix}_oft_r", task.vp_stage
                    )
                    oft_r_tensor = oft_r_param.data
                else:
                    adapter = self._get_oft_adapter_wrap_module(
                        task.local_base_prefix,
                        megatron_model,
                        task.vp_stage,
                        slice_name=task.slice_name,
                        expert_idx=task.expert_idx,
                    )
                    oft_r_tensor = self._gather_oft_r_across_tp(
                        adapter.oft_r.data, task.input_is_parallel, task.block_share, task.is_expert
                    )
            else:
                # Allocate a receive buffer with the correct gathered shape
                is_canonical_split_grouped_recv = (
                    task.slice_name is not None and task.expert_idx is None
                )
                if is_dsv4_grouped_expert or is_canonical_split_grouped_recv:
                    ep_size = parallel_state.get_expert_model_parallel_world_size()
                    num_experts_per_rank = num_moe_experts // max(ep_size, 1)
                    total_blocks = num_blocks_local
                    oft_r_tensor = torch.zeros(
                        num_experts_per_rank,
                        total_blocks,
                        n_elements,
                        device=torch.cuda.current_device(),
                        dtype=param_dtype,
                    )
                elif task.input_is_parallel and not task.block_share:
                    tp_size = (
                        parallel_state.get_expert_tensor_parallel_world_size()
                        if task.is_expert
                        else parallel_state.get_tensor_model_parallel_world_size()
                    )
                    total_blocks = num_blocks_local * tp_size
                    oft_r_tensor = torch.zeros(
                        total_blocks, n_elements, device=torch.cuda.current_device(), dtype=param_dtype
                    )
                else:
                    total_blocks = num_blocks_local
                    oft_r_tensor = torch.zeros(
                        total_blocks, n_elements, device=torch.cuda.current_device(), dtype=param_dtype
                    )

            # ------------------------------------------------------------------
            # Step 2: broadcast from the owning PP rank to all other PP ranks
            # ------------------------------------------------------------------
            if pp_group.size() > 1:
                src_global = pp_global_ranks[task.pp_rank]
                torch.distributed.broadcast(oft_r_tensor, src=src_global, group=pp_group)

            # ------------------------------------------------------------------
            # Step 3: handle expert-parallel gathering for grouped MoE experts
            # ------------------------------------------------------------------
            if is_dsv4_grouped_expert:
                expert_oft_r = self._gather_dsv4_grouped_expert_oft_weight(
                    oft_r_tensor,
                    proj=dsv4_grouped_expert_projection,
                )
                dsv4_grouped_layer_match = _DSV4_GROUPED_EXPERT_OFT_LAYER_RE.search(task.global_base_prefix)
                for expert_idx, current_oft_r in enumerate(expert_oft_r):
                    if cpu:
                        current_oft_r = current_oft_r.cpu()
                    if dsv4_grouped_layer_match is not None:
                        hf_oft_name = (
                            f"layers.{dsv4_grouped_layer_match.group('layer')}.ffn.experts."
                            f"{expert_idx}.{dsv4_grouped_layer_match.group('proj')}.oft_R"
                        )
                        yield HFWeightTuple(hf_oft_name, current_oft_r)
                continue

            # CanonicalOFT grouped split FC1 (gate / up) — single 3D
            # ``oft_r`` of shape ``(num_local_experts, num_blocks, n_elements)``.
            # EP-gather across the expert-parallel group, then emit one HF
            # tuple per global expert with the matching slice leaf
            # (``gate_proj`` / ``up_proj``).
            is_canonical_split_grouped = (
                task.slice_name is not None
                and task.expert_idx is None
                and oft_r_tensor.ndim == 3
            )
            if is_canonical_split_grouped:
                expert_oft_r = self._gather_dsv4_grouped_expert_oft_weight(
                    oft_r_tensor, proj=None,
                )
                leaf = _CANONICAL_OFT_SLICE_TO_HF_LEAF[task.slice_name]
                for expert_idx in range(expert_oft_r.shape[0]):
                    current_oft_r = expert_oft_r[expert_idx]
                    if cpu:
                        current_oft_r = current_oft_r.cpu()
                    base_hf_weight_names = self._get_base_hf_param_names_for_adapter(
                        mapping_registry,
                        task.global_base_prefix,
                        None,
                        f".weight{expert_idx}",
                    )
                    base_hf_weight_names = [
                        name for name in base_hf_weight_names if f".{leaf}." in name
                    ]
                    for base_name in base_hf_weight_names:
                        hf_oft_name = self._make_oft_param_name(base_name)
                        if hf_oft_name is not None:
                            yield HFWeightTuple(hf_oft_name, current_oft_r)
                continue

            is_dsv4_native_expert = _is_dsv4_native_expert_oft_base_prefix(
                task.global_base_prefix
            )
            export_base_prefix = task.global_base_prefix
            if is_dsv4_native_expert:
                export_base_prefix = _globalize_dsv4_native_expert_oft_base_prefix(
                    export_base_prefix,
                    num_moe_experts,
                )
            # Per-expert tasks already point at a single rotation — skip the
            # legacy shared-R fan-out gather/select path.
            is_grouped_expert = (
                task.is_expert
                and ".local_experts." not in task.global_base_prefix
                and not is_dsv4_native_expert
                and task.expert_idx is None
            )
            expert_oft_r_gathered = None
            if is_grouped_expert:
                expert_oft_r_gathered = self._gather_expert_adapter_weight(oft_r_tensor)

            base_suffixes = [".weight"]
            if is_grouped_expert:
                base_suffixes = [f".weight{n}" for n in range(num_moe_experts)]
            elif task.expert_idx is not None:
                base_suffixes = [f".weight{task.expert_idx}"]

            # ------------------------------------------------------------------
            # Step 4: map to HF names and yield
            # ------------------------------------------------------------------
            for base_suffix in base_suffixes:
                current_oft_r = oft_r_tensor
                if is_grouped_expert:
                    expert_idx = int(base_suffix[len(".weight") :])
                    current_oft_r = self._select_expert_adapter_weight(
                        oft_r_tensor, expert_oft_r_gathered, expert_idx, num_moe_experts
                    )

                if cpu:
                    current_oft_r = current_oft_r.cpu()

                base_hf_weight_names = self._get_base_hf_param_names_for_adapter(
                    mapping_registry,
                    export_base_prefix,
                    None,
                    base_suffix,
                )

                # CanonicalOFT split adapters: each task represents one slice
                # (q/k/v/gate/up) and must emit only the matching HF projection.
                if task.slice_name is not None:
                    leaf = _CANONICAL_OFT_SLICE_TO_HF_LEAF[task.slice_name]
                    base_hf_weight_names = [
                        name for name in base_hf_weight_names if f".{leaf}." in name
                    ]

                # Legacy shared-R OFT on a GROUPED MoE expert fused gate/up:
                # the serve side (sglang FusedMoE) stores ONE fused ``w13_oft_r``
                # rotation per expert and detects the fused layout as
                # ``gate_proj.oft_R`` present + ``up_proj.oft_R`` ABSENT. Emit
                # only the gate projection (the shared rotation, applied to the
                # fused gate/up input); sending both gate+up would be read as the
                # split layout and mis-route to the unregistered w1/w3 buffers.
                # Scoped to grouped experts so dense/QKV fan-out is unchanged.
                if is_grouped_expert and task.slice_name is None:
                    _gate_only = [
                        name for name in base_hf_weight_names if ".up_proj." not in name
                    ]
                    if _gate_only:
                        base_hf_weight_names = _gate_only

                # For fused Megatron layers (QKV or gate/up) with the legacy
                # shared-R OFT, the same rotation applies to every
                # sub-projection – emit oft_r for each.
                for base_name in base_hf_weight_names:
                    hf_oft_name = self._make_oft_param_name(base_name)
                    if hf_oft_name is not None:
                        yield HFWeightTuple(hf_oft_name, current_oft_r)

    # ------------------------------------------------------------------
    # OFT merge helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_oft_rotation_matrix(
        oft_r: torch.Tensor,
        block_size: int,
        in_features: int,
        block_share: bool,
    ) -> torch.Tensor:
        """Reconstruct the full block-diagonal orthogonal rotation from ``oft_r``.

        Uses the Cayley transform with a 5-term Neumann series approximation,
        matching :meth:`OFTRotationModule._cayley_batch`.

        Args:
            oft_r: Skew-symmetric parameter vectors ``(num_blocks, n_elements)``.
            block_size: Size of each orthogonal block.
            in_features: Full (un-sharded) input dimension of the base linear layer.
            block_share: Whether all blocks share a single set of parameters.

        Returns:
            ``(in_features, in_features)`` block-diagonal orthogonal matrix.
        """

        num_blocks = oft_r.shape[0]

        # Build skew-symmetric matrices
        rows, cols = torch.triu_indices(block_size, block_size, 1)
        Q = torch.zeros(num_blocks, block_size, block_size, device=oft_r.device, dtype=oft_r.dtype)
        Q[:, rows, cols] = oft_r
        Q = Q - Q.transpose(-2, -1)

        # Cayley transform via Neumann series: R ≈ I + 2Q + 2Q² + 2Q³ + Q⁴
        eye = torch.eye(block_size, device=oft_r.device, dtype=oft_r.dtype)
        R = eye.unsqueeze(0).expand(num_blocks, -1, -1).clone()
        R.add_(Q, alpha=2.0)
        Q_sq = torch.bmm(Q, Q)
        R.add_(Q_sq, alpha=2.0)
        Q_power = torch.bmm(Q_sq, Q)
        R.add_(Q_power, alpha=2.0)
        Q_power = torch.bmm(Q_power, Q)
        R.add_(Q_power)

        # Expand block_share: repeat the single block across all blocks
        if block_share:
            effective_r = in_features // block_size
            R = R.repeat(effective_r, 1, 1)

        return torch.block_diag(*[R[i] for i in range(R.shape[0])])

    def _merge_oft_adapter_weights(
        self,
        megatron_model: List[MegatronModel],
        converted_weights_dict: Dict[str, torch.Tensor],
        oft_r: torch.Tensor,
        block_size: int,
        block_share: bool,
    ) -> Dict[str, torch.Tensor]:
        """Merge an OFT rotation into the base weights for HF export.

        Applies ``W_merged = W @ R`` where R is the full block-diagonal orthogonal
        matrix reconstructed from ``oft_r``.  For fused layers (QKV, gate/up) the same
        rotation is applied to every sub-projection weight in *converted_weights_dict*.

        Unlike LoRA merge, OFT merge does **not** require an all-gather because the
        rotation either operates on the full input (ColumnParallel) or on a TP-local
        shard (RowParallel).  By the time this method is called the ``oft_r`` should
        already be the full (gathered) parameter.

        Args:
            megatron_model: The Megatron model (or list of VP chunks).
            converted_weights_dict: Dict mapping HF weight names to their
                already-converted base tensors.
            oft_r: The full (TP-gathered) ``oft_r`` parameter tensor.
            block_size: Size of each orthogonal block.
            block_share: Whether all blocks share parameters.

        Returns:
            The same *converted_weights_dict* with values replaced by merged weights.
        """

        first_weight = next(iter(converted_weights_dict.values()))
        in_features = first_weight.shape[1]

        R = self._compute_oft_rotation_matrix(oft_r, block_size, in_features, block_share)

        for hf_name, base_weight in list(converted_weights_dict.items()):
            R_dev = R.to(base_weight.device, base_weight.dtype)
            converted_weights_dict[hf_name] = base_weight @ R_dev

        return converted_weights_dict


_HF_LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight")
_HF_OFT_SUFFIXES = (".oft_R",)


def infer_target_modules_from_adapter_weights(
    adapter_weight_names: Iterable[str],
    peft_config: Optional[PEFT] = None,
) -> List[str]:
    """Derive HF ``target_modules`` from the HF-format adapter weight names.

    Given names like ``model.layers.0.self_attn.q_proj.lora_A.weight`` or
    ``model.layers.0.self_attn.q_proj.oft_R``, this extracts the unique module
    identifiers (``q_proj``, ``gate_proj``, ...) that the ``peft`` library
    expects in ``adapter_config.json``.

    When ``peft_config`` is provided, only the suffixes for that adapter family
    (LoRA / DoRA → ``lora_A``/``lora_B``; OFT → ``oft_R``) are matched, so the
    inference is unambiguous when both adapter types coexist on disk. When
    ``peft_config`` is omitted, both families are matched.
    """
    from megatron.bridge.peft.oft import OFT

    if peft_config is not None and isinstance(peft_config, OFT):
        suffixes = _HF_OFT_SUFFIXES
    elif peft_config is not None:
        suffixes = _HF_LORA_SUFFIXES
    else:
        suffixes = (*_HF_LORA_SUFFIXES, *_HF_OFT_SUFFIXES)

    modules: set[str] = set()
    for name in adapter_weight_names:
        for suffix in suffixes:
            if name.endswith(suffix):
                base = name[: -len(suffix)]
                module_name = base.rsplit(".", 1)[-1]
                modules.add(module_name)
                break
    return sorted(modules)


def build_adapter_config_dict(
    peft_config: PEFT,
    target_modules: List[str],
    base_model_name_or_path: Optional[str] = None,
) -> Dict[str, object]:
    """Build an HF PEFT-compatible ``adapter_config.json`` dictionary.

    The returned dict can be serialised directly with ``json.dump`` and is
    loadable by ``peft.PeftModel.from_pretrained`` without any runtime
    dependency on the ``peft`` pip package.

    The strategy is to inherit every field from the megatron-bridge
    ``peft_config`` dataclass via :func:`dataclasses.asdict`, then perform
    the small set of name remappings between Megatron and HuggingFace (e.g.
    ``dim`` → ``r``, ``alpha`` → ``lora_alpha``, ``block_size`` →
    ``oft_block_size``) and add HF-required boilerplate fields
    (``peft_type``, ``task_type`` …) that the Megatron dataclass does not
    carry. Megatron-only training knobs that HF PEFT does not understand
    (e.g. ``lora_A_init_method``, ``lora_dtype``) are dropped.

    Supports LoRA / DoRA configs and OFT configs (detected by class).
    """
    import dataclasses

    from megatron.bridge.peft.dora import DoRA
    from megatron.bridge.peft.oft import OFT

    # ------------------------------------------------------------------
    # 1. Inherit everything we can from the dataclass.
    # ------------------------------------------------------------------
    config: Dict[str, object] = {}
    if dataclasses.is_dataclass(peft_config):
        config.update(dataclasses.asdict(peft_config))
        # JSON does not support sets — normalise to a sorted list.
        for k, v in list(config.items()):
            if isinstance(v, (set, frozenset)):
                config[k] = sorted(v)

    # ------------------------------------------------------------------
    # 2. Common HF PEFT fields shared by all adapter families.
    # ------------------------------------------------------------------
    config["base_model_name_or_path"] = base_model_name_or_path or ""
    # ``target_modules`` must be the HF-format names (q_proj, gate_proj, …),
    # not the Megatron-format names that the dataclass holds — overwrite
    # unconditionally.
    config["target_modules"] = target_modules
    config.setdefault("task_type", "CAUSAL_LM")
    config.setdefault("inference_mode", True)
    config.setdefault("auto_mapping", None)
    config.setdefault("revision", None)
    config.setdefault("modules_to_save", None)

    if isinstance(peft_config, OFT):
        # ----- OFT-specific remapping & HF defaults (peft >= 0.18) ----
        config["peft_type"] = "OFT"
        # Megatron uses ``block_size``; HF PEFT uses ``oft_block_size``.
        if "block_size" in config:
            config["oft_block_size"] = config.pop("block_size")
        config.setdefault("num_cayley_neumann_terms", 5)
        config.setdefault("use_cayley_neumann", True)
        config.setdefault("init_weights", True)
        config.setdefault("bias", "none")
        config.setdefault("fan_in_fan_out", False)
        config.setdefault("layers_pattern", None)
        config.setdefault("layers_to_transform", None)
    else:
        # ----- LoRA / DoRA remapping & HF defaults --------------------
        config["peft_type"] = "LORA"
        # Megatron field names → HF field names.
        if "dim" in config:
            config["r"] = config.pop("dim")
        if "alpha" in config:
            config["lora_alpha"] = config.pop("alpha")
        if "dropout" in config:
            config["lora_dropout"] = config.pop("dropout")
        # Drop Megatron-only training knobs that HF PEFT does not accept.
        for k in (
            "dropout_position",
            "lora_A_init_method",
            "lora_B_init_method",
            "lora_dtype",
            "a2a_experimental",
        ):
            config.pop(k, None)
        config.setdefault("init_lora_weights", True)
        config.setdefault("rank_pattern", {})
        config.setdefault("alpha_pattern", {})
        config.setdefault("use_dora", isinstance(peft_config, DoRA))
        config.setdefault("use_rslora", False)
        config.setdefault("bias", "none")
        config.setdefault("fan_in_fan_out", False)

    return config
