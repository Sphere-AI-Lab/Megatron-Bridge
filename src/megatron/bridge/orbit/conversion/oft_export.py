# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""OFT adapter export/streaming for HF conversion (orbit fork).

OFT is a peer of LoRA: where upstream exports LoRA adapters via
``AutoBridge.export_adapter_weights`` / ``save_hf_adapter``, orbit exports OFT
adapters via the free functions :func:`export_oft_adapter_weights` /
:func:`save_hf_oft_adapter` defined at the bottom of this module. They compose
the architecture's registered model-bridge class with
:class:`OrbitOFTExportMixin` (mixin-first MRO, see
:func:`oft_export_bridge_for`) — no upstream file is edited.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, TypeVar, Union

import torch
from megatron.core import parallel_state
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_pg_rank, unwrap_model

from megatron.bridge.models.conversion.utils import (
    extract_sort_key,
    get_module_and_param_from_name,
    persistent_buffers,
)


if TYPE_CHECKING:
    # Runtime code lazy-imports HFWeightTuple inside the stream method;
    # importing model_bridge at module level here would be circular
    # (peft_bridge -> oft_export -> model_bridge -> peft_bridge).
    from megatron.bridge.models.conversion.model_bridge import HFWeightTuple  # noqa: F401

MegatronModel = TypeVar("MegatronModel", bound=MegatronModule)


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


def _globalize_dsv4_native_expert_oft_base_prefix(
    global_base_prefix: str,
    num_moe_experts: int,
    ep_rank: Optional[int] = None,
) -> str:
    """Map EP-local native DSV4 expert prefixes to global expert IDs for HF export.

    ``ep_rank`` names the rank whose local expert this prefix belongs to;
    ``None`` means this process's own EP rank. The explicit form lets rank 0
    compute peer ranks' global names when emitting their gathered tensors.
    """

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

    if ep_rank is None:
        ep_rank = parallel_state.get_expert_model_parallel_rank()
    global_expert_id = ep_rank * num_experts_per_rank + expert_id
    return f"{match.group('head')}{global_expert_id}{match.group('tail')}"


def _gather_dsv4_native_expert_variants(
    global_base_prefix: str,
    oft_r_tensor: torch.Tensor,
    num_moe_experts: int,
) -> list[tuple[str, torch.Tensor]]:
    """All-gather a native per-expert rotation across the EP group.

    Every rank walks only its own local expert modules, but only rank 0 keeps
    the yielded tensors for the safetensors file -- so a per-expert rotation
    that lives on a nonzero EP rank must be brought over before the yield.
    Renaming the prefix to the global expert id (which this path always did)
    makes the file *look* complete while every nonzero-rank expert is silently
    absent from it. Returns one (globalized prefix, tensor) pair per EP rank;
    all ranks must call this (it is a collective).
    """
    ep_size = parallel_state.get_expert_model_parallel_world_size()
    if ep_size <= 1:
        return [
            (
                _globalize_dsv4_native_expert_oft_base_prefix(global_base_prefix, num_moe_experts),
                oft_r_tensor,
            )
        ]

    gathered = [torch.empty_like(oft_r_tensor) for _ in range(ep_size)]
    torch.distributed.all_gather(gathered, oft_r_tensor, group=parallel_state.get_expert_model_parallel_group())
    return [
        (
            _globalize_dsv4_native_expert_oft_base_prefix(global_base_prefix, num_moe_experts, ep_rank=peer),
            gathered[peer],
        )
        for peer in range(ep_size)
    ]


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

_CANONICAL_OFT_SLICE_SORT_ORDER = {slice_name: index for index, slice_name in enumerate(("q", "k", "v", "gate", "up"))}


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


class OrbitOFTExportMixin:
    """OFT export methods mixed into ``MegatronPeftBridge``.

    Methods are verbatim from the pre-restructure ``MegatronPeftBridge``; the
    adapter-info cache stays on the bridge instance
    (``self._cached_oft_adapter_info``).

    The two predicate overrides below widen upstream's LoRA-shaped name
    handling for OFT; because this mixin precedes the bridge class in the MRO
    (see :func:`oft_export_bridge_for`), they take effect without editing
    upstream's ``MegatronPeftBridge``.
    """

    def _get_lora_unwrapped_name(self, megatron_param: str) -> str:
        """Remove `.to_wrap` (LoRA) or `._orig_module` (OFT) from PEFT parameter names."""
        return megatron_param.replace(".to_wrap.", ".").replace("._orig_module.", ".")

    def _is_adapter_param_name(self, param_name: str) -> bool:
        """Return True if the parameter only belongs to a PEFT adapter.

        Matches both the upstream ``.adapter.`` form and the CanonicalOFT split
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
        where *target_attr* is an OFT wrapper that holds the ``.adapter``
        attribute.

        For CanonicalOFT split wrappers (``OFTLinearSplitQKV`` /
        ``OFTLinearSplitFC1UpGate``) the rotation modules live at
        ``adapter_q`` / ``adapter_k`` / ``adapter_v`` / ``adapter_gate`` /
        ``adapter_up``; ``slice_name`` selects which one to return.
        """

        from megatron.bridge.orbit.oft.canonical_oft import GroupedOFTRotation
        from megatron.bridge.orbit.oft.oft_layers import OFTRotationModule

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
                    local_info.append(
                        (
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
                        )
                    )
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

                local_info.append(
                    (
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
                    )
                )

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
                is_canonical_split_grouped_recv = task.slice_name is not None and task.expert_idx is None
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
                task.slice_name is not None and task.expert_idx is None and oft_r_tensor.ndim == 3
            )
            if is_canonical_split_grouped:
                expert_oft_r = self._gather_dsv4_grouped_expert_oft_weight(
                    oft_r_tensor,
                    proj=None,
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
                    base_hf_weight_names = [name for name in base_hf_weight_names if f".{leaf}." in name]
                    for base_name in base_hf_weight_names:
                        hf_oft_name = self._make_oft_param_name(base_name)
                        if hf_oft_name is not None:
                            yield HFWeightTuple(hf_oft_name, current_oft_r)
                continue

            is_dsv4_native_expert = _is_dsv4_native_expert_oft_base_prefix(task.global_base_prefix)
            if is_dsv4_native_expert:
                # Collective: every rank contributes its local expert's rotation
                # and receives every peer's, so rank 0 can emit the full expert
                # set instead of silently writing only its own EP slice.
                emit_variants = _gather_dsv4_native_expert_variants(
                    task.global_base_prefix,
                    oft_r_tensor,
                    num_moe_experts,
                )
            else:
                emit_variants = [(task.global_base_prefix, oft_r_tensor)]
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
            # Step 4: map to HF names and yield (once per EP variant for native
            # per-expert tasks; a single pass otherwise)
            # ------------------------------------------------------------------
            for export_base_prefix, emit_oft_r in emit_variants:
                for base_suffix in base_suffixes:
                    current_oft_r = emit_oft_r
                    if is_grouped_expert:
                        expert_idx = int(base_suffix[len(".weight") :])
                        current_oft_r = self._select_expert_adapter_weight(
                            emit_oft_r, expert_oft_r_gathered, expert_idx, num_moe_experts
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
                        base_hf_weight_names = [name for name in base_hf_weight_names if f".{leaf}." in name]

                    # Legacy shared-R OFT on a GROUPED MoE expert fused gate/up:
                    # the serve side (sglang FusedMoE) stores ONE fused ``w13_oft_r``
                    # rotation per expert and detects the fused layout as
                    # ``gate_proj.oft_R`` present + ``up_proj.oft_R`` ABSENT. Emit
                    # only the gate projection (the shared rotation, applied to the
                    # fused gate/up input); sending both gate+up would be read as the
                    # split layout and mis-route to the unregistered w1/w3 buffers.
                    # Scoped to grouped experts so dense/QKV fan-out is unchanged.
                    if is_grouped_expert and task.slice_name is None:
                        _gate_only = [name for name in base_hf_weight_names if ".up_proj." not in name]
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


# ---------------------------------------------------------------------------
# Orbit-side OFT export API (peer of AutoBridge.save_hf_adapter for LoRA).
#
# Upstream's export path is LoRA-only; instead of editing it, orbit composes
# the registered model-bridge class with OrbitOFTExportMixin (mixin-first MRO)
# and exposes free-function entrypoints below.
# ---------------------------------------------------------------------------

_HF_OFT_SUFFIXES = (".oft_R",)

_OFT_BRIDGE_CLASS_CACHE: Dict[type, type] = {}


def oft_export_bridge_for(auto_bridge) -> "MegatronModule":
    """Return the model bridge for ``auto_bridge``'s architecture, composed with the OFT mixin.

    Mirrors upstream's ``_get_model_bridge_impl`` construction (fresh instance,
    ``hf_pretrained``/``hf_config`` attached) with ``OrbitOFTExportMixin``
    placed first in the MRO.
    """
    from megatron.bridge.models.conversion import model_bridge as model_bridge_mod

    base = model_bridge_mod.get_model_bridge(auto_bridge._causal_lm_architecture)
    base_cls = type(base)
    oft_cls = _OFT_BRIDGE_CLASS_CACHE.get(base_cls)
    if oft_cls is None:
        oft_cls = type(f"OrbitOFT{base_cls.__name__}", (OrbitOFTExportMixin, base_cls), {})
        _OFT_BRIDGE_CLASS_CACHE[base_cls] = oft_cls

    bridge = oft_cls()
    hf_pretrained = getattr(auto_bridge, "hf_pretrained", None)
    if hf_pretrained is not None:
        bridge.hf_pretrained = hf_pretrained
        bridge.hf_config = hf_pretrained.config if hasattr(hf_pretrained, "config") else hf_pretrained
    return bridge


def export_oft_adapter_weights(auto_bridge, model, cpu: bool = False, show_progress: bool = True):
    """Export only OFT adapter weights from a Megatron model in HuggingFace format.

    Unlike LoRA's ``lora_A``/``lora_B`` matrices, OFT adapters consist of a
    single rotation parameter per adapted layer; fused Megatron layers (QKV,
    gate/up) emit it once per sub-projection. All ranks must consume the
    returned generator — the gather uses TP/PP/EP collectives.

    Yields:
        HFWeightTuple: ``(param_name, weight_tensor)`` for OFT adapter parameters.
    """
    bridge = oft_export_bridge_for(auto_bridge)
    return bridge.stream_oft_adapter_weights_megatron_to_hf(model, cpu=cpu, show_progress=show_progress)


def infer_oft_target_modules(adapter_weight_names: Iterable[str]) -> List[str]:
    """Derive HF ``target_modules`` from HF-format OFT adapter weight names."""
    modules: set[str] = set()
    for name in adapter_weight_names:
        for suffix in _HF_OFT_SUFFIXES:
            if name.endswith(suffix):
                modules.add(name[: -len(suffix)].rsplit(".", 1)[-1])
    return sorted(modules)


def build_oft_adapter_config_dict(
    peft_config,
    target_modules: List[str],
    base_model_name_or_path: Optional[str] = None,
) -> Dict[str, object]:
    """Build a HF-PEFT-compatible ``adapter_config.json`` dict for an OFT adapter.

    Inherits every field from the megatron-bridge ``peft_config`` dataclass,
    remaps Megatron names to HF names (``block_size`` → ``oft_block_size``) and
    adds the HF boilerplate fields (``peft_type`` …) the dataclass lacks. The
    result loads with ``peft.PeftModel.from_pretrained`` without depending on
    the ``peft`` pip package here.
    """
    import dataclasses as _dataclasses

    def _json_sanitize(value):
        # JSON has no sets/tuples; normalize them anywhere in the tree.
        if isinstance(value, (set, frozenset)):
            return sorted(value)
        if isinstance(value, tuple):
            return [_json_sanitize(v) for v in value]
        if isinstance(value, list):
            return [_json_sanitize(v) for v in value]
        if isinstance(value, dict):
            return {k: _json_sanitize(v) for k, v in value.items()}
        return value

    config: Dict[str, object] = {}
    if _dataclasses.is_dataclass(peft_config):
        config.update(_dataclasses.asdict(peft_config))

    config["base_model_name_or_path"] = base_model_name_or_path or ""
    # ``target_modules`` must be the HF-format names (q_proj, gate_proj, ...).
    config["target_modules"] = target_modules
    config.setdefault("task_type", "CAUSAL_LM")
    config.setdefault("inference_mode", True)
    config.setdefault("auto_mapping", None)
    config.setdefault("revision", None)
    config.setdefault("modules_to_save", None)

    config["peft_type"] = "OFT"
    if "block_size" in config:
        config["oft_block_size"] = config.pop("block_size")
    config.setdefault("num_cayley_neumann_terms", 5)
    config.setdefault("use_cayley_neumann", True)
    config.setdefault("init_weights", True)
    config.setdefault("bias", "none")
    config.setdefault("fan_in_fan_out", False)
    config.setdefault("layers_pattern", None)
    config.setdefault("layers_to_transform", None)
    return _json_sanitize(config)


def save_hf_oft_adapter(
    auto_bridge,
    model,
    path,
    peft_config,
    base_model_name_or_path: Optional[str] = None,
    show_progress: bool = True,
) -> None:
    """Save OFT adapter weights as a HuggingFace PEFT-compatible directory.

    The output directory contains ``adapter_config.json`` and
    ``adapter_model.safetensors`` and loads directly with
    ``peft.PeftModel.from_pretrained(base_model, path)``. Peer of upstream's
    ``AutoBridge.save_hf_adapter`` (which handles LoRA / DoRA).
    """
    import json
    from pathlib import Path

    import torch.distributed as dist
    from safetensors.torch import save_file

    is_distributed = dist.is_available() and dist.is_initialized()
    is_rank0 = (not is_distributed) or dist.get_rank() == 0

    if is_distributed:
        dist.barrier()

    # The OFT exporter keeps tensors on GPU during the per-rank gather because
    # it relies on collective broadcasts; non-rank0 ranks must still consume
    # the generator to participate in the TP/PP/EP collectives.
    adapter_state: Dict[str, torch.Tensor] = {}
    for name, tensor in export_oft_adapter_weights(auto_bridge, model, cpu=False, show_progress=show_progress):
        if is_rank0:
            adapter_state[f"base_model.model.{name}"] = tensor.detach().cpu()

    if is_rank0:
        if not adapter_state:
            raise RuntimeError(
                "No adapter weights were found on the model. "
                "Ensure the model has OFT adapters applied before calling save_hf_oft_adapter()."
            )

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        if base_model_name_or_path is None:
            hf_pretrained = getattr(auto_bridge, "hf_pretrained", None)
            base_model_name_or_path = str(
                getattr(hf_pretrained, "model_name_or_path", None) or getattr(hf_pretrained, "name_or_path", "")
            )

        target_modules = infer_oft_target_modules(adapter_state.keys())
        adapter_config = build_oft_adapter_config_dict(
            peft_config,
            target_modules=target_modules,
            base_model_name_or_path=base_model_name_or_path,
        )
        with open(save_dir / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=2, sort_keys=True)

        save_file(adapter_state, str(save_dir / "adapter_model.safetensors"))

    if is_distributed:
        dist.barrier()
