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

import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from importlib import metadata as importlib_metadata
from typing import TYPE_CHECKING, Callable, Iterable, TypeVar

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
    ".oft_r": ".oft_R.weight",
}
HF_OFT_EMBEDDING_SUFFIX = ".oft_embedding_R.weight"

GDN_IN_PROJ_KEYS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
_DSV4_NATIVE_EXPERT_OFT_BASE_PREFIX_RE = re.compile(r"^(?P<head>.*\.mlp\.experts\.)(?P<expert>\d+)(?P<tail>\.w[123])$")
_DSV4_GROUPED_EXPERT_OFT_PARAM_RE = re.compile(r"^(?P<base>.*\.mlp\.(?P<proj>w[123]))_oft_r$")
_DSV4_GROUPED_EXPERT_OFT_BASE_PREFIX_RE = re.compile(r"^(?P<base>.*\.mlp\.(?P<proj>w[123]))$")
_DSV4_GROUPED_EXPERT_OFT_LAYER_RE = re.compile(r"(?:^|\.)decoder\.layers\.(?P<layer>\d+)\.mlp\.(?P<proj>w[123])$")
_DSV4_PROJ_TO_HF_LEAF = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}


def _infer_oft_block_size_from_n_elements(n_elements: int) -> int:
    """Infer OFT block size from compact skew-vector length."""
    discriminant = 1 + 8 * n_elements
    root = math.isqrt(discriminant)
    block_size = (1 + root) // 2
    if root * root != discriminant or block_size * (block_size - 1) // 2 != n_elements:
        raise ValueError(f"Cannot infer OFT block size from n_elements={n_elements}")
    return block_size


def _distributed_error(stage: str, local_error: str | None) -> None:
    """Raise one coordinated error before any OFT tensor collective.

    OFT export is a world-rank operation. Discovery and mapping are deliberately
    exchanged through the default process group so a rank-local Python failure
    cannot leave a peer waiting in a later PP/TP/EP tensor collective.
    """

    dist = torch.distributed
    if not dist.is_available() or not dist.is_initialized():
        if local_error is not None:
            raise RuntimeError(f"{stage} failed: {local_error}")
        return

    payload = None if local_error is None else {"rank": dist.get_rank(), "error": local_error[:2000]}
    gathered: list[dict[str, object] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, payload)
    failures = [item for item in gathered if item is not None]
    if failures:
        details = "; ".join(f"rank {item['rank']}: {item['error']}" for item in failures)
        raise RuntimeError(f"{stage} failed: {details}")


def _agree_oft_task_plan(tasks: list["OFTAdapterConversionTask"]) -> bool:
    """Require every world rank to discover the same ordered OFT task plan."""

    dist = torch.distributed
    if not dist.is_available() or not dist.is_initialized():
        return bool(tasks)

    local_contract = (len(tasks), _contract_digest([_oft_task_contract(task) for task in tasks]))
    gathered: list[tuple[int, str] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_contract)
    reference = gathered[0]
    if reference is None:
        raise RuntimeError("OFT task-plan agreement received no plan from rank 0")
    mismatches = [rank for rank, task_contract in enumerate(gathered) if task_contract != reference]
    if mismatches:
        raise RuntimeError(
            f"OFT task plan differs from rank 0 on ranks {mismatches}; "
            f"rank 0 contract is count={reference[0]}, digest={reference[1]}"
        )
    return reference[0] > 0


def _exception_summary(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _contract_digest(value: object) -> str:
    """Return a stable fixed-size digest for JSON-serializable export metadata."""

    encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bounded_error_summary(errors: list[str], *, limit: int = 8) -> str | None:
    """Bound rank diagnostics before exchanging them through an object collective."""

    if not errors:
        return None
    shown = [error[:1000] for error in errors[:limit]]
    if len(errors) > limit:
        shown.append(f"... {len(errors) - limit} additional errors omitted")
    return "; ".join(shown)


def _expected_local_expert_count(num_moe_experts: int) -> int:
    """Return the exact local expert count for the active EP topology."""

    ep_size = int(parallel_state.get_expert_model_parallel_world_size())
    if ep_size < 1:
        raise ValueError(f"expert model parallel size must be positive, got {ep_size}")
    if num_moe_experts < 1:
        raise ValueError(f"num_moe_experts must be positive for expert OFT export, got {num_moe_experts}")
    if num_moe_experts % ep_size != 0:
        raise ValueError(f"num_moe_experts={num_moe_experts} must be divisible by ep_size={ep_size}")
    return num_moe_experts // ep_size


def _receive_spec_for_task(task: "OFTAdapterConversionTask") -> tuple[torch.dtype, torch.device]:
    """Resolve a rank-local receive dtype/device from an agreed task record."""

    supported_dtypes = {
        str(torch.float16): torch.float16,
        str(torch.bfloat16): torch.bfloat16,
        str(torch.float32): torch.float32,
        str(torch.float64): torch.float64,
    }
    dtype = supported_dtypes.get(task.tensor_dtype)
    if dtype is None:
        raise ValueError(f"unsupported OFT adapter dtype {task.tensor_dtype!r}")
    if task.device_type == "cpu":
        return dtype, torch.device("cpu")
    if task.device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("OFT adapter task requires CUDA, but CUDA is unavailable on this rank")
        return dtype, torch.device("cuda", torch.cuda.current_device())
    raise ValueError(f"unsupported OFT adapter device type {task.device_type!r}")


def _materialize_oft_task_outputs(
    outputs: list[tuple[str, torch.Tensor]],
    *,
    cpu: bool,
) -> list[tuple[str, torch.Tensor]]:
    """Materialize one task's outputs before any later tensor collective."""

    if not cpu:
        return outputs

    materialized: list[tuple[str, torch.Tensor]] = []
    local_error: str | None = None
    try:
        materialized = [(name, tensor.detach().to(device="cpu").contiguous().clone()) for name, tensor in outputs]
    except Exception as exc:
        local_error = _exception_summary(exc)
    _distributed_error("OFT task CPU materialization", local_error)
    return materialized


def _is_dsv4_native_expert_oft_base_prefix(global_base_prefix: str) -> bool:
    """Return whether ``global_base_prefix`` is a native DSV4 per-expert OFT prefix."""

    return _DSV4_NATIVE_EXPERT_OFT_BASE_PREFIX_RE.match(global_base_prefix) is not None


def _globalize_dsv4_native_expert_oft_base_prefix(
    global_base_prefix: str,
    num_moe_experts: int,
    ep_rank: int | None = None,
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

    num_experts_per_rank = _expected_local_expert_count(num_moe_experts)
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
    slice_name: str | None = None
    # Set when the slice adapter is a per-expert ``nn.ModuleList`` (grouped split FC1):
    # the task owns one expert's rotation and emits a single per-expert HF name.
    expert_idx: int | None = None
    # PEFT stores embedding rotations under ``oft_embedding_R`` rather than
    # the linear layer's ``oft_R`` parameter name.
    is_embedding: bool = False
    # A non-None value identifies an explicit leading local-expert tensor axis.
    grouped_expert_count: int | None = None
    # String metadata stays process-safe while making receive buffers exact.
    tensor_dtype: str = ""
    device_type: str = ""


def _oft_task_contract(task: OFTAdapterConversionTask) -> tuple[object, ...]:
    """Convert one task to deterministic primitive metadata for agreement."""

    return (
        task.global_base_prefix,
        task.local_base_prefix,
        task.is_expert,
        task.input_is_parallel,
        task.block_size,
        task.r,
        task.block_share,
        task.pp_rank,
        task.vp_stage,
        task.slice_name,
        task.expert_idx,
        task.is_embedding,
        task.grouped_expert_count,
        task.tensor_dtype,
        task.device_type,
    )


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


class OFTExportFormat(str, Enum):
    """Consumer-specific layouts for exported OFT adapter weights."""

    SGLANG = "sglang"
    HF_PEFT = "hf_peft"

    def __str__(self) -> str:
        """Return the serialized string value, matching Python 3.11 StrEnum."""
        return self.value


def _dsv4_grouped_oft_name(
    *,
    layer: str,
    expert_idx: int,
    projection: str,
    export_format: OFTExportFormat,
) -> str:
    """Return the consumer-specific name for one grouped DSV4 rotation."""

    if export_format == OFTExportFormat.HF_PEFT:
        leaf = _DSV4_PROJ_TO_HF_LEAF[projection]
        return f"model.layers.{layer}.mlp.experts.{expert_idx}.{leaf}.oft_R.weight"
    return f"layers.{layer}.ffn.experts.{expert_idx}.{projection}.oft_R.weight"


def _filter_legacy_grouped_oft_weight_names(
    base_hf_weight_names: list[str],
    *,
    export_format: OFTExportFormat,
) -> list[str]:
    """Choose the legacy grouped-MoE gate/up naming convention for a target.

    SGLang recognizes one shared fused ``w13_oft_r`` from a gate key with no up
    key. HF PEFT instead attaches OFT independently to both unfused projections,
    so its adapter directory needs both names even though the tensors are equal.
    """
    if export_format == OFTExportFormat.HF_PEFT:
        return base_hf_weight_names
    gate_only = [name for name in base_hf_weight_names if ".up_proj." not in name]
    return gate_only or base_hf_weight_names


def _canonical_oft_export_base_prefix(global_base_prefix: str, slice_name: str | None) -> str:
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
    is_embedding = info[11] if len(info) > 11 else False
    grouped_expert_count = info[12] if len(info) > 12 else None
    tensor_dtype = info[13] if len(info) > 13 else ""
    device_type = info[14] if len(info) > 14 else ""
    slice_order = _CANONICAL_OFT_SLICE_SORT_ORDER.get(slice_name, -1)
    expert_order = -1 if expert_idx is None else int(expert_idx)
    return (
        extract_sort_key(info[0]),
        slice_order,
        expert_order,
        info[1],
        info[7],
        info[8],
        is_embedding,
        -1 if grouped_expert_count is None else int(grouped_expert_count),
        tensor_dtype,
        device_type,
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
    def _parse_canonical_oft_slice(param_name: str) -> str | None:
        """If ``param_name`` is a CanonicalOFT split-adapter param, return its
        slice name (``"q"`` / ``"k"`` / ``"v"`` / ``"gate"`` / ``"up"``).
        Otherwise return ``None``."""
        for slice_name in _CANONICAL_OFT_SLICE_TO_HF_LEAF:
            if f".adapter_{slice_name}." in param_name:
                return slice_name
        return None

    def _make_oft_param_name(
        self,
        base_name: str,
        megatron_oft_suffix: str = ".oft_r",
        *,
        is_embedding: bool = False,
        export_format: OFTExportFormat = OFTExportFormat.SGLANG,
    ) -> str | None:
        """Translate a base HF weight name into its OFT-specific counterpart.

        Example:
            ``model.layers.0.self_attn.q_proj.weight`` → ``model.layers.0.self_attn.q_proj.oft_R.weight``
        """
        hf_suffix = (
            HF_OFT_EMBEDDING_SUFFIX
            if is_embedding and megatron_oft_suffix == ".oft_r"
            else MEGATRON_TO_HF_OFT_SUFFIX.get(megatron_oft_suffix)
        )
        if hf_suffix is None:
            return None
        if base_name.endswith(".weight"):
            return base_name[: -len(".weight")] + hf_suffix
        # Some grouped-expert mappings name a tensor container without the
        # final ``.weight`` component (for example Qwen3.5 gate_up_proj).
        # Neither PEFT nor SGLang can attach an OFT rotation to that bare
        # parameter: both consumers require concrete per-module/per-expert
        # names. Fail closed instead of emitting a plausible but misrouted key.
        raise ValueError(f"{export_format.value} OFT export requires a module weight mapping, got {base_name!r}")

    def _gather_dsv4_grouped_expert_oft_weight(
        self,
        weight: torch.Tensor,
        *,
        proj: str | None = None,
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
        megatron_model: MegatronModel | list[MegatronModel],
        vp_stage: int,
        slice_name: str | None = None,
        expert_idx: int | None = None,
    ) -> torch.nn.Module | None:
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

    def _local_oft_adapter_info_for_parameter(
        self,
        *,
        megatron_model: list[MegatronModel],
        model_config,
        local_param_name: str,
        param: torch.Tensor,
        vp_stage: int,
        pp_rank: int,
        local_to_global,
    ) -> tuple | None:
        """Build one dependency-free discovery record for an OFT parameter."""

        from megatron.bridge.orbit.oft.oft_layers import OFTVocabParallelEmbedding

        if "_extra_state" in local_param_name:
            return None
        local_param_name = self._unwrap_name(local_param_name)
        global_param_name = local_to_global(megatron_model, model_config, local_param_name, vp_stage)

        dsv4_grouped_match = _DSV4_GROUPED_EXPERT_OFT_PARAM_RE.match(global_param_name)
        if dsv4_grouped_match is not None:
            if param.dim() != 3:
                raise ValueError(
                    f"Expected grouped DSV4 OFT param {global_param_name} to be 3D, got {tuple(param.shape)}"
                )
            return (
                dsv4_grouped_match.group("base"),
                local_param_name[: -len("_oft_r")],
                True,
                False,
                _infer_oft_block_size_from_n_elements(int(param.shape[-1])),
                int(param.shape[1]),
                False,
                pp_rank,
                vp_stage,
                None,
                None,
                False,
                int(param.shape[0]),
                str(param.dtype),
                param.device.type,
            )

        if not self._is_adapter_param_name(global_param_name) or not global_param_name.endswith(".oft_r"):
            return None

        slice_name = self._parse_canonical_oft_slice(global_param_name)
        expert_idx: int | None = None
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
            raise RuntimeError(
                f"Discovered OFT parameter {global_param_name!r}, but could not resolve "
                f"its adapter wrapper at {local_base_prefix!r}"
            )

        _, wrapper = get_module_and_param_from_name(megatron_model, local_base_prefix, vp_stage)
        grouped_expert_count = int(param.shape[0]) if param.ndim == 3 else None
        return (
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
            isinstance(wrapper, OFTVocabParallelEmbedding),
            grouped_expert_count,
            str(param.dtype),
            param.device.type,
        )

    def _megatron_global_oft_adapters_info_all_pp_ranks(
        self, megatron_model: MegatronModel | list[MegatronModel]
    ) -> list[OFTAdapterConversionTask]:
        """Collect OFT adapter information across all pipeline parallel ranks.

        Returns a sorted, deduplicated list of :class:`OFTAdapterConversionTask` entries,
        one per adapted module across the entire model.
        """

        if hasattr(self, "_cached_oft_adapter_info"):
            return self._cached_oft_adapter_info

        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        local_info: list[tuple] = []
        discovery_error: str | None = None
        try:
            from megatron.bridge.models.conversion.model_bridge import _megatron_local_name_to_global

            pp_group = parallel_state.get_pipeline_model_parallel_group()
            pp_rank = get_pg_rank(pp_group)
            model_config = unwrap_model(megatron_model)[0].config
            for vp_stage, model in enumerate(megatron_model):
                for local_param_name, param in itertools.chain(model.named_parameters(), persistent_buffers(model)):
                    info = self._local_oft_adapter_info_for_parameter(
                        megatron_model=megatron_model,
                        model_config=model_config,
                        local_param_name=local_param_name,
                        param=param,
                        vp_stage=vp_stage,
                        pp_rank=pp_rank,
                        local_to_global=_megatron_local_name_to_global,
                    )
                    if info is not None:
                        local_info.append(info)
        except Exception as exc:
            discovery_error = _exception_summary(exc)

        _distributed_error("OFT adapter discovery", discovery_error)

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
                is_embedding=t[11],
                grouped_expert_count=t[12],
                tensor_dtype=t[13],
                device_type=t[14],
            )
            for t in sorted_info
        ]

        self._cached_oft_adapter_info = result
        return result

    def _planned_hf_names_for_oft_task(
        self,
        task: OFTAdapterConversionTask,
        mapping_registry,
        num_moe_experts: int,
        export_format: OFTExportFormat,
    ) -> list[str]:
        """Resolve every HF key a task will emit without touching tensor data."""

        if task.is_expert:
            ep_size = int(parallel_state.get_expert_model_parallel_world_size())
            expected_local_experts = _expected_local_expert_count(num_moe_experts)
            is_rank_local_layout = ".local_experts." in task.global_base_prefix or task.expert_idx is not None
            if ep_size > 1 and is_rank_local_layout:
                raise ValueError(
                    f"rank-local expert OFT layout {task.global_base_prefix!r} is not safe to export with "
                    f"ep_size={ep_size}"
                )
            if task.grouped_expert_count is not None and task.grouped_expert_count != expected_local_experts:
                raise ValueError(
                    f"grouped expert axis for {task.global_base_prefix!r} has "
                    f"{task.grouped_expert_count} local experts, expected {expected_local_experts}"
                )

        dsv4_grouped_match = _DSV4_GROUPED_EXPERT_OFT_BASE_PREFIX_RE.match(task.global_base_prefix)
        if dsv4_grouped_match is not None:
            layer_match = _DSV4_GROUPED_EXPERT_OFT_LAYER_RE.search(task.global_base_prefix)
            if layer_match is None:
                return []
            return [
                _dsv4_grouped_oft_name(
                    layer=layer_match.group("layer"),
                    expert_idx=expert_idx,
                    projection=layer_match.group("proj"),
                    export_format=export_format,
                )
                for expert_idx in range(num_moe_experts)
            ]

        is_canonical_split_grouped = task.grouped_expert_count is not None and task.slice_name is not None
        if is_canonical_split_grouped:
            leaf = _CANONICAL_OFT_SLICE_TO_HF_LEAF[task.slice_name]
            names: list[str] = []
            for expert_idx in range(num_moe_experts):
                base_names = self._get_base_hf_param_names_for_adapter(
                    mapping_registry,
                    task.global_base_prefix,
                    None,
                    f".weight{expert_idx}",
                )
                for base_name in base_names:
                    if f".{leaf}." not in base_name:
                        continue
                    hf_name = self._make_oft_param_name(
                        base_name,
                        is_embedding=task.is_embedding,
                        export_format=export_format,
                    )
                    if hf_name is not None:
                        names.append(hf_name)
            return names

        is_dsv4_native_expert = _is_dsv4_native_expert_oft_base_prefix(task.global_base_prefix)
        if is_dsv4_native_expert:
            ep_size = parallel_state.get_expert_model_parallel_world_size()
            export_prefixes = [
                _globalize_dsv4_native_expert_oft_base_prefix(
                    task.global_base_prefix,
                    num_moe_experts,
                    ep_rank=ep_rank,
                )
                for ep_rank in range(ep_size)
            ]
        else:
            export_prefixes = [task.global_base_prefix]

        is_grouped_expert = (
            task.is_expert
            and ".local_experts." not in task.global_base_prefix
            and not is_dsv4_native_expert
            and task.expert_idx is None
        )
        if is_grouped_expert:
            base_suffixes = [f".weight{expert_idx}" for expert_idx in range(num_moe_experts)]
        elif task.expert_idx is not None:
            base_suffixes = [f".weight{task.expert_idx}"]
        else:
            base_suffixes = [".weight"]

        names = []
        for export_prefix in export_prefixes:
            for base_suffix in base_suffixes:
                base_names = self._get_base_hf_param_names_for_adapter(
                    mapping_registry,
                    export_prefix,
                    None,
                    base_suffix,
                )
                if task.slice_name is not None:
                    leaf = _CANONICAL_OFT_SLICE_TO_HF_LEAF[task.slice_name]
                    base_names = [name for name in base_names if f".{leaf}." in name]
                if is_grouped_expert and task.slice_name is None:
                    base_names = _filter_legacy_grouped_oft_weight_names(
                        base_names,
                        export_format=export_format,
                    )
                for base_name in base_names:
                    hf_name = self._make_oft_param_name(
                        base_name,
                        is_embedding=task.is_embedding,
                        export_format=export_format,
                    )
                    if hf_name is not None:
                        names.append(hf_name)
        return names

    def _validate_oft_export_preflight(
        self,
        megatron_model: list[MegatronModel],
        tasks: list[OFTAdapterConversionTask],
        mapping_registry,
        num_moe_experts: int,
        export_format: OFTExportFormat,
    ) -> list[str]:
        """Validate wrappers, compact geometry, and final names before tensors move."""

        local_errors: list[str] = []
        planned_names: list[str] = []
        try:
            for task in tasks:
                task_names = self._planned_hf_names_for_oft_task(
                    task,
                    mapping_registry,
                    num_moe_experts,
                    export_format,
                )
                if not task_names:
                    local_errors.append(f"{task.global_base_prefix!r} maps to no HF OFT parameter")
                planned_names.extend(task_names)

            seen_names: set[str] = set()
            duplicate_names: set[str] = set()
            for name in planned_names:
                if name in seen_names:
                    duplicate_names.add(name)
                seen_names.add(name)
            if duplicate_names:
                examples = sorted(duplicate_names)[:8]
                local_errors.append(f"{len(duplicate_names)} duplicate HF OFT output keys; examples: {examples}")
        except Exception as exc:
            local_errors.append(_exception_summary(exc))

        my_pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        for task in tasks:
            if my_pp_rank != task.pp_rank:
                continue
            try:
                dsv4_grouped = _DSV4_GROUPED_EXPERT_OFT_BASE_PREFIX_RE.match(task.global_base_prefix) is not None
                canonical_grouped = task.grouped_expert_count is not None and task.slice_name is not None
                grouped_layout = task.grouped_expert_count is not None
                if grouped_layout and not (dsv4_grouped or canonical_grouped):
                    raise ValueError("unsupported OFT tensor with a leading expert axis")
                if dsv4_grouped:
                    _, param = get_module_and_param_from_name(
                        megatron_model,
                        f"{task.local_base_prefix}_oft_r",
                        task.vp_stage,
                    )
                    oft_r = param.data
                else:
                    adapter = self._get_oft_adapter_wrap_module(
                        task.local_base_prefix,
                        megatron_model,
                        task.vp_stage,
                        slice_name=task.slice_name,
                        expert_idx=task.expert_idx,
                    )
                    if adapter is None:
                        raise RuntimeError(f"adapter wrapper {task.local_base_prefix!r} is unresolved")
                    oft_r = adapter.oft_r.data

                expected_ndim = 3 if grouped_layout else 2
                if oft_r.ndim != expected_ndim:
                    raise ValueError(f"expected {expected_ndim}D oft_r, got shape {tuple(oft_r.shape)}")
                if task.tensor_dtype != str(oft_r.dtype):
                    raise ValueError(
                        f"oft_r dtype {oft_r.dtype} does not match discovered dtype {task.tensor_dtype!r}"
                    )
                if task.device_type != oft_r.device.type:
                    raise ValueError(
                        f"oft_r device {oft_r.device.type!r} does not match discovered device {task.device_type!r}"
                    )
                if grouped_layout:
                    expected_local_experts = _expected_local_expert_count(num_moe_experts)
                    if int(oft_r.shape[0]) != expected_local_experts:
                        raise ValueError(
                            f"oft_r expert axis {oft_r.shape[0]} does not match expected local expert count "
                            f"{expected_local_experts}"
                        )
                if task.block_size < 2 or task.r < 1:
                    raise ValueError(f"invalid OFT geometry block_size={task.block_size}, r={task.r}")
                expected_elements = task.block_size * (task.block_size - 1) // 2
                if int(oft_r.shape[-1]) != expected_elements:
                    raise ValueError(
                        f"oft_r last dimension {oft_r.shape[-1]} does not encode block_size={task.block_size}"
                    )
                expected_blocks = 1 if task.block_share else task.r
                if int(oft_r.shape[-2]) != expected_blocks:
                    raise ValueError(
                        f"oft_r block dimension {oft_r.shape[-2]} does not match expected {expected_blocks}"
                    )
                if not torch.is_floating_point(oft_r) or not bool(torch.isfinite(oft_r).all().item()):
                    raise ValueError("oft_r must contain finite floating-point values")
                if task.is_embedding and (task.is_expert or task.input_is_parallel or task.slice_name is not None):
                    raise ValueError("embedding OFT task has incompatible expert/parallel/slice metadata")
            except Exception as exc:
                local_errors.append(f"{task.global_base_prefix!r}: {_exception_summary(exc)}")

        dist = torch.distributed
        if not dist.is_available() or not dist.is_initialized():
            if local_errors:
                raise RuntimeError(f"OFT export preflight failed: {'; '.join(local_errors)}")
            return planned_names

        local_record = (
            _bounded_error_summary(local_errors),
            len(planned_names),
            _contract_digest(planned_names),
        )
        world_records: list[tuple[str | None, int, str] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(world_records, local_record)
        if any(record is None for record in world_records):
            empty_ranks = [rank for rank, record in enumerate(world_records) if record is None]
            raise RuntimeError(f"OFT export preflight received empty contracts from ranks {empty_ranks}")
        records = [record for record in world_records if record is not None]
        failures = [f"rank {rank}: {record[0]}" for rank, record in enumerate(records) if record[0]]
        reference = records[0]
        name_mismatches = [rank for rank, record in enumerate(records) if record[1:] != reference[1:]]
        if failures or name_mismatches:
            details = failures
            if name_mismatches:
                details.append(
                    f"HF key plan differs from rank 0 on ranks {name_mismatches}; "
                    f"rank 0 contract is count={reference[1]}, digest={reference[2]}"
                )
            raise RuntimeError(f"OFT export preflight failed: {'; '.join(details)}")

        return planned_names

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
        megatron_model: MegatronModel | list[MegatronModel],
        cpu: bool = True,
        show_progress: bool = True,
        export_format: OFTExportFormat = OFTExportFormat.SGLANG,
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
            export_format: Consumer-specific adapter layout. SGLang uses one
                fused gate key; HF PEFT needs equal gate and up keys.

        Yields:
            ``HFWeightTuple`` with the HF adapter parameter name (e.g.
            ``model.layers.0.self_attn.q_proj.oft_R.weight``) and the corresponding tensor.
        """

        from megatron.bridge.models.conversion.model_bridge import HFWeightTuple

        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        num_moe_experts = 0
        normalized_export_format = OFTExportFormat.SGLANG
        model_setup_error: str | None = None
        try:
            normalized_export_format = OFTExportFormat(export_format)
            if not megatron_model:
                raise ValueError("OFT export requires at least one model chunk")
            model_config = unwrap_model(megatron_model)[0].config
            raw_num_moe_experts = getattr(model_config, "num_moe_experts", None)
            num_moe_experts = 0 if raw_num_moe_experts is None else int(raw_num_moe_experts)
            if num_moe_experts < 0:
                raise ValueError(f"num_moe_experts must be nonnegative, got {num_moe_experts}")
        except Exception as exc:
            model_setup_error = _exception_summary(exc)
        _distributed_error("OFT export model setup", model_setup_error)
        export_format = normalized_export_format

        tasks = self._megatron_global_oft_adapters_info_all_pp_ranks(megatron_model)
        if not _agree_oft_task_plan(tasks):
            return

        setup_error: str | None = None
        mapping_registry = None
        pp_group = None
        my_pp_rank = 0
        pp_global_ranks: list[int] = []
        receive_specs: dict[OFTAdapterConversionTask, tuple[torch.dtype, torch.device, tuple[int, ...]]] = {}
        try:
            if not hasattr(self, "mapping_registry"):
                raise RuntimeError("MegatronModelBridge must define mapping_registry")
            mapping_registry = self.mapping_registry()
            pp_group = parallel_state.get_pipeline_model_parallel_group()
            my_pp_rank = parallel_state.get_pipeline_model_parallel_rank()
            pp_global_ranks = torch.distributed.get_process_group_ranks(pp_group)
            pp_size = pp_group.size()
            if len(pp_global_ranks) != pp_size:
                raise RuntimeError(f"pipeline group reports size {pp_size}, but has global ranks {pp_global_ranks}")

            for task in tasks:
                if not 0 <= task.pp_rank < pp_size:
                    raise ValueError(
                        f"OFT task {task.global_base_prefix!r} has invalid PP owner {task.pp_rank} for size {pp_size}"
                    )
                dtype, device = _receive_spec_for_task(task)
                n_elements = task.block_size * (task.block_size - 1) // 2
                num_blocks = 1 if task.block_share else task.r
                if task.grouped_expert_count is not None:
                    shape = (task.grouped_expert_count, num_blocks, n_elements)
                elif task.input_is_parallel and not task.block_share:
                    tp_size = (
                        parallel_state.get_expert_tensor_parallel_world_size()
                        if task.is_expert
                        else parallel_state.get_tensor_model_parallel_world_size()
                    )
                    if tp_size < 1:
                        raise ValueError(f"tensor parallel size must be positive, got {tp_size}")
                    shape = (num_blocks * tp_size, n_elements)
                else:
                    shape = (num_blocks, n_elements)
                receive_specs[task] = (dtype, device, shape)
        except Exception as exc:
            setup_error = _exception_summary(exc)
        _distributed_error("OFT export collective setup", setup_error)

        planned_names = self._validate_oft_export_preflight(
            megatron_model,
            tasks,
            mapping_registry,
            num_moe_experts,
            export_format,
        )
        emitted_names: list[str] = []

        for task in self._with_progress_tracking(tasks, "Streaming OFT adapter weights", show_progress):
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
                receive_dtype, receive_device, receive_shape = receive_specs[task]
                oft_r_tensor = torch.zeros(receive_shape, device=receive_device, dtype=receive_dtype)

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
                task_outputs: list[tuple[str, torch.Tensor]] = []
                for expert_idx, current_oft_r in enumerate(expert_oft_r):
                    if dsv4_grouped_layer_match is not None:
                        hf_oft_name = _dsv4_grouped_oft_name(
                            layer=dsv4_grouped_layer_match.group("layer"),
                            expert_idx=expert_idx,
                            projection=dsv4_grouped_layer_match.group("proj"),
                            export_format=export_format,
                        )
                        task_outputs.append((hf_oft_name, current_oft_r))
                for hf_oft_name, current_oft_r in _materialize_oft_task_outputs(task_outputs, cpu=cpu):
                    emitted_names.append(hf_oft_name)
                    yield HFWeightTuple(hf_oft_name, current_oft_r)
                continue

            # CanonicalOFT grouped split FC1 (gate / up) — single 3D
            # ``oft_r`` of shape ``(num_local_experts, num_blocks, n_elements)``.
            # EP-gather across the expert-parallel group, then emit one HF
            # tuple per global expert with the matching slice leaf
            # (``gate_proj`` / ``up_proj``).
            is_canonical_split_grouped = task.grouped_expert_count is not None and task.slice_name is not None
            if is_canonical_split_grouped:
                expert_oft_r = self._gather_dsv4_grouped_expert_oft_weight(
                    oft_r_tensor,
                    proj=None,
                )
                leaf = _CANONICAL_OFT_SLICE_TO_HF_LEAF[task.slice_name]
                task_outputs = []
                for expert_idx in range(expert_oft_r.shape[0]):
                    current_oft_r = expert_oft_r[expert_idx]
                    base_hf_weight_names = self._get_base_hf_param_names_for_adapter(
                        mapping_registry,
                        task.global_base_prefix,
                        None,
                        f".weight{expert_idx}",
                    )
                    base_hf_weight_names = [name for name in base_hf_weight_names if f".{leaf}." in name]
                    for base_name in base_hf_weight_names:
                        hf_oft_name = self._make_oft_param_name(
                            base_name,
                            is_embedding=task.is_embedding,
                            export_format=export_format,
                        )
                        if hf_oft_name is not None:
                            task_outputs.append((hf_oft_name, current_oft_r))
                for hf_oft_name, current_oft_r in _materialize_oft_task_outputs(task_outputs, cpu=cpu):
                    emitted_names.append(hf_oft_name)
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
            task_outputs = []
            for export_base_prefix, emit_oft_r in emit_variants:
                for base_suffix in base_suffixes:
                    current_oft_r = emit_oft_r
                    if is_grouped_expert:
                        expert_idx = int(base_suffix[len(".weight") :])
                        current_oft_r = self._select_expert_adapter_weight(
                            emit_oft_r, expert_oft_r_gathered, expert_idx, num_moe_experts
                        )

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
                    # ``gate_proj.oft_R.weight`` present + ``up_proj.oft_R.weight`` ABSENT. Emit
                    # only the gate projection (the shared rotation, applied to the
                    # fused gate/up input); sending both gate+up would be read as the
                    # split layout and mis-route to the unregistered w1/w3 buffers.
                    # Scoped to grouped experts so dense/QKV fan-out is unchanged.
                    if is_grouped_expert and task.slice_name is None:
                        base_hf_weight_names = _filter_legacy_grouped_oft_weight_names(
                            base_hf_weight_names,
                            export_format=export_format,
                        )

                    # For fused Megatron layers (QKV or gate/up) with the legacy
                    # shared-R OFT, the same rotation applies to every
                    # sub-projection – emit oft_r for each.
                    for base_name in base_hf_weight_names:
                        hf_oft_name = self._make_oft_param_name(
                            base_name,
                            is_embedding=task.is_embedding,
                            export_format=export_format,
                        )
                        if hf_oft_name is not None:
                            task_outputs.append((hf_oft_name, current_oft_r))

            for hf_oft_name, current_oft_r in _materialize_oft_task_outputs(task_outputs, cpu=cpu):
                emitted_names.append(hf_oft_name)
                yield HFWeightTuple(hf_oft_name, current_oft_r)

        stream_error = None
        if emitted_names != planned_names:
            mismatch_index = next(
                (
                    index
                    for index, (planned, emitted) in enumerate(zip(planned_names, emitted_names))
                    if planned != emitted
                ),
                min(len(planned_names), len(emitted_names)),
            )
            stream_error = (
                "emitted HF keys differ from the validated plan at index "
                f"{mismatch_index} (planned_count={len(planned_names)}, emitted_count={len(emitted_names)})"
            )
        _distributed_error("OFT export stream verification", stream_error)

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
        megatron_model: list[MegatronModel],
        converted_weights_dict: dict[str, torch.Tensor],
        oft_r: torch.Tensor,
        block_size: int,
        block_share: bool,
    ) -> dict[str, torch.Tensor]:
        """Merge an OFT rotation into the base weights for HF export.

        Applies ``W_merged = W @ R.T`` where R is the full block-diagonal orthogonal
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
            converted_weights_dict[hf_name] = base_weight @ R_dev.transpose(-1, -2)

        return converted_weights_dict


# ---------------------------------------------------------------------------
# Orbit-side OFT export API (peer of AutoBridge.save_hf_adapter for LoRA).
#
# Upstream's export path is LoRA-only; instead of editing it, orbit composes
# the registered model-bridge class with OrbitOFTExportMixin (mixin-first MRO)
# and exposes free-function entrypoints below.
# ---------------------------------------------------------------------------

_HF_OFT_SUFFIXES = (".oft_R.weight", HF_OFT_EMBEDDING_SUFFIX)

_OFT_BRIDGE_CLASS_CACHE: dict[type, type] = {}


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


def export_oft_adapter_weights(
    auto_bridge,
    model,
    cpu: bool = False,
    show_progress: bool = True,
    export_format: OFTExportFormat = OFTExportFormat.SGLANG,
):
    """Export only OFT adapter weights from a Megatron model in HuggingFace format.

    Unlike LoRA's ``lora_A``/``lora_B`` matrices, OFT adapters consist of a
    single rotation parameter per adapted layer; fused Megatron layers (QKV,
    gate/up) emit it once per sub-projection. All ranks must consume the
    returned generator — the gather uses TP/PP/EP collectives.

    Yields:
        HFWeightTuple: ``(param_name, weight_tensor)`` for OFT adapter parameters.
    """
    bridge = oft_export_bridge_for(auto_bridge)
    return bridge.stream_oft_adapter_weights_megatron_to_hf(
        model,
        cpu=cpu,
        show_progress=show_progress,
        export_format=export_format,
    )


def infer_oft_target_modules(adapter_weight_names: Iterable[str]) -> list[str]:
    """Derive exact PEFT module paths from HF-format OFT adapter names.

    Leaf-only targets (for example ``q_proj``) would broaden a layer-specific
    Bridge adapter to every matching module in the HF model. PEFT accepts exact
    paths, so preserve the entire path after its serialization-only prefix.
    """
    modules: set[str] = set()
    for name in adapter_weight_names:
        for suffix in _HF_OFT_SUFFIXES:
            if name.endswith(suffix):
                module_name = name[: -len(suffix)]
                if module_name.startswith("base_model.model."):
                    module_name = module_name[len("base_model.model.") :]
                if module_name:
                    modules.add(module_name)
                break
    return sorted(modules)


def _installed_peft_version() -> str:
    """Return a PEFT version compatible with Bridge's emitted parameterization."""

    try:
        version = importlib_metadata.version("peft")
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError("HF OFT export requires the project's peft dependency") from exc

    from packaging.version import InvalidVersion, Version

    try:
        parsed = Version(version)
    except InvalidVersion as exc:
        raise RuntimeError(f"Cannot serialize an invalid installed peft version: {version!r}") from exc
    if parsed < Version("0.18.0"):
        raise RuntimeError(f"HF OFT export requires peft>=0.18.0 for Cayley-Neumann compatibility; found {version}")
    return version


def build_oft_adapter_config_dict(
    peft_config,
    target_modules: list[str],
    base_model_name_or_path: str | None = None,
) -> dict[str, object]:
    """Build a HF-PEFT-compatible ``adapter_config.json`` dict for an OFT adapter.

    Bridge OFT objects also inherit matcher caches, checkpoint policy, and VLM
    freeze controls. Serializing the dataclass wholesale leaks those private
    fields into PEFT's schema, so construct the accepted PEFT 0.19 fields
    explicitly.
    """

    def _json_sanitize(value):
        # JSON has no sets/tuples; normalize them anywhere in the tree.
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, (set, frozenset)):
            return sorted(value)
        if isinstance(value, tuple):
            return [_json_sanitize(v) for v in value]
        if isinstance(value, list):
            return [_json_sanitize(v) for v in value]
        if isinstance(value, dict):
            return {k: _json_sanitize(v) for k, v in value.items()}
        return value

    config: dict[str, object] = {
        "task_type": "CAUSAL_LM",
        "peft_type": "OFT",
        "auto_mapping": None,
        "peft_version": _installed_peft_version(),
        "base_model_name_or_path": base_model_name_or_path or "",
        "revision": None,
        "inference_mode": True,
        "r": int(getattr(peft_config, "r", 0)),
        "oft_block_size": int(getattr(peft_config, "block_size", 0)),
        "module_dropout": float(getattr(peft_config, "module_dropout", 0.0)),
        "target_modules": list(target_modules),
        "fan_in_fan_out": False,
        "bias": "none",
        # Exact target paths already encode Bridge's matcher selection. Carrying
        # Bridge patterns or layer filters would apply a second, incompatible
        # filter inside PEFT.
        "exclude_modules": None,
        "init_weights": True,
        "layers_to_transform": None,
        "layers_pattern": None,
        "modules_to_save": None,
        "coft": bool(getattr(peft_config, "coft", False)),
        "eps": float(getattr(peft_config, "eps", 6e-5)),
        "block_share": bool(getattr(peft_config, "block_share", False)),
        "use_cayley_neumann": True,
        "num_cayley_neumann_terms": 5,
    }
    return _json_sanitize(config)


def _validate_serialized_oft_state_geometry(
    adapter_state: dict[str, torch.Tensor],
    peft_config,
) -> None:
    """Validate final compact tensors against the one serialized PEFT config."""

    configured_r = int(getattr(peft_config, "r", 0))
    configured_block_size = int(getattr(peft_config, "block_size", 0))
    block_share = bool(getattr(peft_config, "block_share", False))
    if (configured_r > 0) == (configured_block_size > 0):
        raise ValueError("exactly one of r and block_size must be positive")
    for name, tensor in adapter_state.items():
        if tensor.ndim != 2:
            raise ValueError(f"{name!r} must be a 2D compact OFT tensor, got {tuple(tensor.shape)}")
        if tensor.shape[0] < 1:
            raise ValueError(f"{name!r} must contain at least one OFT block")
        block_size = _infer_oft_block_size_from_n_elements(int(tensor.shape[-1]))
        if block_size < 2:
            raise ValueError(f"{name!r} encodes degenerate block_size={block_size}")
        if configured_block_size > 0 and block_size != configured_block_size:
            raise ValueError(
                f"{name!r} encodes block_size={block_size}, not configured block_size={configured_block_size}"
            )
        expected_blocks = 1 if block_share else configured_r
        if configured_r > 0 and int(tensor.shape[0]) != expected_blocks:
            raise ValueError(
                f"{name!r} contains {tensor.shape[0]} blocks, not the configured PEFT count {expected_blocks}"
            )
        if block_share and int(tensor.shape[0]) != 1:
            raise ValueError(f"{name!r} is block-shared but stores {tensor.shape[0]} blocks")
        if not torch.is_floating_point(tensor) or not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name!r} must contain finite floating-point values")


def _run_rank0_stage(
    stage: str,
    *,
    is_distributed: bool,
    is_rank0: bool,
    operation: Callable[[], None],
) -> None:
    """Run one filesystem stage on rank zero and broadcast its terminal result."""

    outcome: dict[str, object] | None = None
    if is_rank0:
        try:
            operation()
            outcome = {"ok": True}
        except Exception as exc:
            outcome = {"ok": False, "error_type": type(exc).__name__, "message": str(exc)}

    if is_distributed:
        payload = [outcome]
        torch.distributed.broadcast_object_list(payload, src=0)
        outcome = payload[0]

    if outcome is None:
        raise RuntimeError(f"{stage} failed: rank-zero outcome was not broadcast")
    if not outcome["ok"]:
        message = f"{stage} failed: {outcome['error_type']}: {outcome['message']}"
        if outcome["error_type"] == "FileExistsError":
            raise FileExistsError(message)
        raise RuntimeError(message)


def _publish_hf_oft_adapter_directory(
    save_dir,
    adapter_state: dict[str, torch.Tensor],
    adapter_config: dict[str, object],
) -> None:
    """Build both adapter files in a sibling and publish them together."""

    import os
    import shutil
    import tempfile
    from pathlib import Path

    from safetensors import safe_open
    from safetensors.torch import save_file

    save_dir = Path(save_dir)
    if save_dir.exists() or save_dir.is_symlink():
        raise FileExistsError(f"OFT adapter destination already exists: {save_dir}")
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{save_dir.name}.oft-staging-", dir=str(save_dir.parent)))
    published = False
    try:
        config_path = staging_dir / "adapter_config.json"
        weights_path = staging_dir / "adapter_model.safetensors"
        with config_path.open("w", encoding="utf-8") as config_file:
            json.dump(adapter_config, config_file, indent=2, sort_keys=True)
            config_file.write("\n")
        save_file(adapter_state, str(weights_path))

        with config_path.open(encoding="utf-8") as config_file:
            if json.load(config_file) != adapter_config:
                raise RuntimeError("staged adapter_config.json did not round-trip")
        with safe_open(str(weights_path), framework="pt", device="cpu") as weights_file:
            if set(weights_file.keys()) != set(adapter_state):
                raise RuntimeError("staged safetensors keys do not match the export plan")

        if save_dir.exists() or save_dir.is_symlink():
            raise FileExistsError(f"OFT adapter destination appeared during export: {save_dir}")
        os.rename(staging_dir, save_dir)
        published = True
    finally:
        if not published:
            shutil.rmtree(staging_dir, ignore_errors=True)


def save_hf_oft_adapter(
    auto_bridge,
    model,
    path,
    peft_config,
    base_model_name_or_path: str | None = None,
    show_progress: bool = True,
) -> None:
    """Save OFT adapter weights as a HuggingFace PEFT-compatible directory.

    The output directory contains ``adapter_config.json`` and
    ``adapter_model.safetensors`` and loads directly with
    ``peft.PeftModel.from_pretrained(base_model, path)``. Peer of upstream's
    ``AutoBridge.save_hf_adapter`` (which handles LoRA / DoRA). ``path`` must
    not already exist: the complete directory is published in one rename so
    readers cannot observe a config/weights pair from different exports.
    """
    from pathlib import Path

    import torch.distributed as dist

    is_distributed = dist.is_available() and dist.is_initialized()
    is_rank0 = (not is_distributed) or dist.get_rank() == 0
    save_dir = Path(path)

    def _check_destination() -> None:
        if save_dir.exists() or save_dir.is_symlink():
            raise FileExistsError(f"OFT adapter destination already exists: {save_dir}")

    _run_rank0_stage(
        "OFT adapter destination preflight",
        is_distributed=is_distributed,
        is_rank0=is_rank0,
        operation=_check_destination,
    )

    # Every rank must consume the generator to participate in TP/PP/EP
    # collectives. Each task is copied to CPU under a coordinated error check
    # before the exporter advances to the next task.
    adapter_state: dict[str, torch.Tensor] = {}
    seen_names: set[str] = set()
    for name, tensor in export_oft_adapter_weights(
        auto_bridge,
        model,
        cpu=True,
        show_progress=show_progress,
        export_format=OFTExportFormat.HF_PEFT,
    ):
        final_name = f"base_model.model.{name}"
        if final_name in seen_names:
            raise RuntimeError(f"duplicate OFT adapter output key: {final_name}")
        seen_names.add(final_name)
        if is_rank0:
            # The exporter coordinates and completes each task's host copy
            # before advancing to any later tensor collective.
            adapter_state[final_name] = tensor.detach()

    if not seen_names:
        raise RuntimeError(
            "No adapter weights were found on the model. "
            "Ensure the model has OFT adapters applied before calling save_hf_oft_adapter()."
        )

    def _validate_and_publish() -> None:
        nonlocal base_model_name_or_path

        # Fused source modules intentionally emit one rotation under multiple
        # HF keys. Materialize independent CPU storage because safetensors
        # rejects aliases even when every logical key is valid.
        for name, tensor in adapter_state.items():
            adapter_state[name] = tensor.detach().to(device="cpu").contiguous().clone()
        cpu_adapter_state = adapter_state
        _validate_serialized_oft_state_geometry(cpu_adapter_state, peft_config)

        if base_model_name_or_path is None:
            hf_pretrained = getattr(auto_bridge, "hf_pretrained", None)
            base_model_name_or_path = str(
                getattr(hf_pretrained, "model_name_or_path", None) or getattr(hf_pretrained, "name_or_path", "")
            )

        target_modules = infer_oft_target_modules(cpu_adapter_state.keys())
        if not target_modules:
            raise RuntimeError("exported weights contain no recognized OFT adapter keys")
        adapter_config = build_oft_adapter_config_dict(
            peft_config,
            target_modules=target_modules,
            base_model_name_or_path=base_model_name_or_path,
        )
        _publish_hf_oft_adapter_directory(save_dir, cpu_adapter_state, adapter_config)

    _run_rank0_stage(
        "OFT adapter publication",
        is_distributed=is_distributed,
        is_rank0=is_rank0,
        operation=_validate_and_publish,
    )
