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

"""Compare logical and physical tensor metadata across model checkpoints.

Security:
    PyTorch distributed-checkpoint ``.metadata`` files are pickle payloads.
    Only inspect checkpoints from trusted sources because loading a malicious
    pickle can execute arbitrary code.
"""

from __future__ import annotations

import math
import pickle
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.distributed.checkpoint.metadata import BytesStorageMetadata, TensorStorageMetadata


@dataclass(frozen=True)
class DTypeSummary:
    """Physical tensor rollup for a single dtype."""

    tensor_count: int = 0
    numel: int = 0
    num_bytes: int = 0


@dataclass(frozen=True)
class TensorCollectionSummary:
    """Physical rollup of a tensor collection, broken down by dtype name."""

    tensor_count: int
    total_numel: int
    total_num_bytes: int
    dtype_stats: dict[str, DTypeSummary]


@dataclass(frozen=True)
class LogicalCategorySummary:
    """Logical tensor rollup for one category, counting packed weights at their unpacked size."""

    tensor_count: int = 0
    numel: int = 0


@dataclass(frozen=True)
class LogicalTensorSummary:
    """Logical rollup of a tensor collection, broken down by category (``dense``, ``int4_quantized``, ...)."""

    tensor_count: int
    total_numel: int
    category_stats: dict[str, LogicalCategorySummary]


_HF_CONFIG_FIELD_NAMES = {
    "hidden_size": ("hidden_size",),
    "num_layers": ("num_hidden_layers", "num_layers"),
    "ffn_hidden_size": ("intermediate_size", "ffn_hidden_size"),
    "num_attention_heads": ("num_attention_heads",),
    "num_query_groups": ("num_key_value_heads", "num_query_groups"),
    "num_moe_experts": ("num_local_experts", "num_moe_experts", "num_experts", "n_routed_experts"),
    "vocab_size": ("vocab_size",),
    "gated_linear_unit": ("gated_linear_unit",),
}

_MEGATRON_CONFIG_FIELD_NAMES = {
    "hidden_size": ("hidden_size",),
    "num_layers": ("num_layers", "num_hidden_layers"),
    "ffn_hidden_size": ("ffn_hidden_size", "intermediate_size"),
    "num_attention_heads": ("num_attention_heads",),
    "num_query_groups": ("num_query_groups", "num_key_value_heads"),
    "num_moe_experts": ("num_moe_experts", "num_local_experts", "num_experts"),
    "vocab_size": ("vocab_size",),
    "gated_linear_unit": ("gated_linear_unit",),
}

_NESTED_HF_CONFIG_ATTR_NAMES = (
    "text_config",
    "language_config",
    "llm_config",
    "decoder_config",
)


@dataclass(frozen=True)
class _QuantizedFormatSpec:
    category: str
    packed_patterns: tuple[re.Pattern[str], ...]
    metadata_patterns: tuple[re.Pattern[str], ...]
    packed_logical_numel_multiplier: int
    scale_group_size: int | None = None


_QUANTIZED_FORMAT_SPECS = (
    _QuantizedFormatSpec(
        category="int4_quantized",
        packed_patterns=(
            re.compile(r"^(?P<base>.+)\.weight_packed$"),
            re.compile(r"^(?P<base>.+)_packed$"),
        ),
        metadata_patterns=(
            re.compile(r"^(?P<base>.+)\.weight_scale$"),
            re.compile(r"^(?P<base>.+)\.weight_shape$"),
            re.compile(r"^(?P<base>.+)_scale$"),
            re.compile(r"^(?P<base>.+)_shape$"),
        ),
        packed_logical_numel_multiplier=8,
        scale_group_size=32,
    ),
)


def _model_key_filter(key: str) -> bool:
    return (
        key == "model"
        or key.startswith("model.")
        or (key.startswith("model") and len(key) > 5 and key[5].isdigit() and "." in key)
    )


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _numel_from_shape(shape: Iterable[int]) -> int:
    return math.prod(shape)


def _summarize_tensor_metadata_items(
    items: Iterable[tuple[str, TensorStorageMetadata]],
) -> TensorCollectionSummary:
    dtype_stats: dict[str, DTypeSummary] = {}
    tensor_count = 0
    total_numel = 0
    total_num_bytes = 0

    for _, entry in items:
        tensor_count += 1
        numel = _numel_from_shape(entry.size)
        num_bytes = numel * _dtype_element_size(entry.properties.dtype)
        total_numel += numel
        total_num_bytes += num_bytes

        dtype_key = _dtype_name(entry.properties.dtype)
        current = dtype_stats.get(dtype_key, DTypeSummary())
        dtype_stats[dtype_key] = DTypeSummary(
            tensor_count=current.tensor_count + 1,
            numel=current.numel + numel,
            num_bytes=current.num_bytes + num_bytes,
        )

    return TensorCollectionSummary(
        tensor_count=tensor_count,
        total_numel=total_numel,
        total_num_bytes=total_num_bytes,
        dtype_stats=dict(sorted(dtype_stats.items())),
    )


def summarize_torch_dist_state_dict_metadata(
    state_dict_metadata: Mapping[str, TensorStorageMetadata | BytesStorageMetadata],
) -> TensorCollectionSummary:
    """Roll up tensor entries of a torch-dist ``state_dict_metadata`` by dtype.

    Restricts to model-prefixed keys when any are present, so optimizer and RNG
    state do not distort the comparison; otherwise summarizes every tensor entry.
    """
    tensor_items = [
        (str(key), value) for key, value in state_dict_metadata.items() if isinstance(value, TensorStorageMetadata)
    ]

    model_tensor_items = [(key, value) for key, value in tensor_items if _model_key_filter(key)]
    if model_tensor_items:
        return _summarize_tensor_metadata_items(model_tensor_items)

    return _summarize_tensor_metadata_items(tensor_items)


def _match_quantized_packed_base(key: str) -> tuple[_QuantizedFormatSpec, str] | None:
    for spec in _QUANTIZED_FORMAT_SPECS:
        for pattern in spec.packed_patterns:
            match = pattern.match(key)
            if match is not None:
                return spec, match.group("base")
    return None


def _match_quantized_metadata_base(
    key: str, known_quantized_bases: Mapping[str, _QuantizedFormatSpec]
) -> tuple[_QuantizedFormatSpec, str] | None:
    for spec in _QUANTIZED_FORMAT_SPECS:
        for pattern in spec.metadata_patterns:
            match = pattern.match(key)
            if match is None:
                continue
            base = match.group("base")
            known_spec = known_quantized_bases.get(base)
            if known_spec == spec:
                return spec, base
    return None


def _logical_numel_from_quantized_entry(
    spec: _QuantizedFormatSpec,
    key: str,
    shape: tuple[int, ...],
) -> int | None:
    packed_match = _match_quantized_packed_base(key)
    if packed_match is not None:
        _, _ = packed_match
        if len(shape) == 2:
            return shape[0] * shape[1] * spec.packed_logical_numel_multiplier
        return None

    if (
        spec.scale_group_size is not None
        and len(shape) == 2
        and (key.endswith(".weight_scale") or key.endswith("_scale"))
    ):
        return shape[0] * shape[1] * spec.scale_group_size

    return None


def _summarize_logical_named_entries(
    items: Iterable[tuple[str, torch.dtype, tuple[int, ...]]],
) -> LogicalTensorSummary:
    normalized_items = [(key, dtype, tuple(shape)) for key, dtype, shape in items]
    quantized_bases: dict[str, _QuantizedFormatSpec] = {}
    for key, _, _ in normalized_items:
        packed_match = _match_quantized_packed_base(key)
        if packed_match is not None:
            spec, base = packed_match
            quantized_bases[base] = spec

    logical_entries: dict[tuple[str, str], int] = {}

    for key, _, shape in normalized_items:
        packed_match = _match_quantized_packed_base(key)
        if packed_match is not None:
            spec, base = packed_match
            logical_numel = _logical_numel_from_quantized_entry(spec, key, shape)
            if logical_numel is not None:
                logical_entries[(spec.category, base)] = logical_numel
            continue

        metadata_match = _match_quantized_metadata_base(key, quantized_bases)
        if metadata_match is not None:
            spec, base = metadata_match
            logical_entries.setdefault((spec.category, base), 0)
            fallback_numel = _logical_numel_from_quantized_entry(spec, key, shape)
            if fallback_numel is not None and logical_entries[(spec.category, base)] == 0:
                logical_entries[(spec.category, base)] = fallback_numel
            continue

        logical_entries[("dense", key)] = _numel_from_shape(shape)

    category_stats: dict[str, LogicalCategorySummary] = {}
    total_numel = 0
    tensor_count = 0
    for (category, _), logical_numel in sorted(logical_entries.items()):
        tensor_count += 1
        total_numel += logical_numel
        current = category_stats.get(category, LogicalCategorySummary())
        category_stats[category] = LogicalCategorySummary(
            tensor_count=current.tensor_count + 1,
            numel=current.numel + logical_numel,
        )

    return LogicalTensorSummary(
        tensor_count=tensor_count,
        total_numel=total_numel,
        category_stats=dict(sorted(category_stats.items())),
    )


def summarize_logical_torch_dist_state_dict_metadata(
    state_dict_metadata: Mapping[str, TensorStorageMetadata | BytesStorageMetadata],
) -> LogicalTensorSummary:
    """Roll up a torch-dist ``state_dict_metadata`` by logical category.

    Uses the same model-key preference as
    :func:`summarize_torch_dist_state_dict_metadata`, but counts packed
    quantized weights at their dequantized element count.
    """
    tensor_items = [
        (str(key), value.properties.dtype, tuple(value.size))
        for key, value in state_dict_metadata.items()
        if isinstance(value, TensorStorageMetadata)
    ]
    model_tensor_items = [(key, dtype, shape) for key, dtype, shape in tensor_items if _model_key_filter(key)]
    if model_tensor_items:
        return _summarize_logical_named_entries(model_tensor_items)
    return _summarize_logical_named_entries(tensor_items)


def summarize_logical_torch_dist_checkpoint_metadata(checkpoint_path: str | Path) -> LogicalTensorSummary:
    """Read a torch-dist checkpoint's ``.metadata`` and roll it up by logical category.

    Warning:
        ``.metadata`` is loaded with :mod:`pickle`. Only pass checkpoints from
        trusted sources.

    Raises:
        ValueError: If the metadata pickle carries no ``state_dict_metadata``.
    """
    metadata_path = Path(checkpoint_path) / ".metadata"
    with metadata_path.open("rb") as metadata_file:
        metadata = pickle.load(metadata_file)

    state_dict_metadata = getattr(metadata, "state_dict_metadata", None)
    if state_dict_metadata is None:
        raise ValueError(f"Checkpoint metadata at {metadata_path} does not contain state_dict_metadata")

    return summarize_logical_torch_dist_state_dict_metadata(state_dict_metadata)


def summarize_torch_dist_checkpoint_metadata(checkpoint_path: str | Path) -> TensorCollectionSummary:
    """Read a torch-dist checkpoint's ``.metadata`` and roll it up by dtype.

    Warning:
        ``.metadata`` is loaded with :mod:`pickle`. Only pass checkpoints from
        trusted sources.

    Raises:
        ValueError: If the metadata pickle carries no ``state_dict_metadata``.
    """
    metadata_path = Path(checkpoint_path) / ".metadata"
    with metadata_path.open("rb") as metadata_file:
        metadata = pickle.load(metadata_file)

    state_dict_metadata = getattr(metadata, "state_dict_metadata", None)
    if state_dict_metadata is None:
        raise ValueError(f"Checkpoint metadata at {metadata_path} does not contain state_dict_metadata")

    return summarize_torch_dist_state_dict_metadata(state_dict_metadata)


def _batched(iterable: Iterable[str], batch_size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _iter_state_mapping_tensors(state: Mapping[str, Any], chunk_size: int) -> Iterable[tuple[str, torch.Tensor]]:
    source = getattr(state, "source", None)
    if source is not None and hasattr(source, "load_tensors"):
        for key_batch in _batched(state.keys(), chunk_size):
            loaded = source.load_tensors(key_batch)
            for key in key_batch:
                tensor = loaded.get(key)
                if torch.is_tensor(tensor):
                    yield key, tensor
        return

    for key, value in state.items():
        if torch.is_tensor(value):
            yield str(key), value


def iter_named_tensors(node: Any, *, prefix: str = "", chunk_size: int = 128) -> Iterable[tuple[str, torch.Tensor]]:
    """Recursively yield ``(dotted_name, tensor)`` pairs from nested mappings and sequences.

    When a mapping exposes a lazy ``source`` with ``load_tensors``, keys are
    fetched in batches of ``chunk_size`` so the whole state need not be resident.
    """
    if torch.is_tensor(node):
        if prefix:
            yield prefix, node
        return

    if isinstance(node, Mapping):
        if getattr(node, "source", None) is not None and hasattr(getattr(node, "source"), "load_tensors"):
            yield from _iter_state_mapping_tensors(node, chunk_size)
            return

        for key, value in node.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_named_tensors(value, prefix=child_prefix, chunk_size=chunk_size)
        return

    if isinstance(node, (list, tuple)):
        for idx, value in enumerate(node):
            child_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            yield from iter_named_tensors(value, prefix=child_prefix, chunk_size=chunk_size)


def summarize_tensor_collection(node: Any, *, chunk_size: int = 128) -> TensorCollectionSummary:
    """Walk a live tensor collection and roll it up by dtype."""
    dtype_stats: dict[str, DTypeSummary] = {}
    tensor_count = 0
    total_numel = 0
    total_num_bytes = 0

    for _, tensor in iter_named_tensors(node, chunk_size=chunk_size):
        tensor_count += 1
        numel = tensor.numel()
        num_bytes = numel * tensor.element_size()
        total_numel += numel
        total_num_bytes += num_bytes

        dtype_key = _dtype_name(tensor.dtype)
        current = dtype_stats.get(dtype_key, DTypeSummary())
        dtype_stats[dtype_key] = DTypeSummary(
            tensor_count=current.tensor_count + 1,
            numel=current.numel + numel,
            num_bytes=current.num_bytes + num_bytes,
        )

    return TensorCollectionSummary(
        tensor_count=tensor_count,
        total_numel=total_numel,
        total_num_bytes=total_num_bytes,
        dtype_stats=dict(sorted(dtype_stats.items())),
    )


def summarize_logical_tensor_collection(node: Any, *, chunk_size: int = 128) -> LogicalTensorSummary:
    """Walk a live tensor collection and roll it up by logical category."""
    items = [
        (key, tensor.dtype, tuple(tensor.shape)) for key, tensor in iter_named_tensors(node, chunk_size=chunk_size)
    ]
    return _summarize_logical_named_entries(items)


def _iter_hf_config_candidates(config: Any) -> Iterable[Any]:
    yield config

    seen = {id(config)}
    for attr_name in _NESTED_HF_CONFIG_ATTR_NAMES:
        if not hasattr(config, attr_name):
            continue
        nested_config = getattr(config, attr_name)
        if nested_config is None or id(nested_config) in seen:
            continue
        seen.add(id(nested_config))
        yield nested_config


def extract_config_fields(config: Any, field_names: Mapping[str, tuple[str, ...]]) -> dict[str, Any]:
    """Resolve normalized config fields from a config and its nested sub-configs.

    For each public name, the first alias that resolves to a non-``None`` value
    wins, searching the config itself before its ``text_config`` /
    ``language_config`` / ``llm_config`` / ``decoder_config`` children.
    """
    fields: dict[str, Any] = {}
    for candidate in _iter_hf_config_candidates(config):
        for public_name, aliases in field_names.items():
            if public_name in fields:
                continue
            for alias in aliases:
                if hasattr(candidate, alias):
                    value = getattr(candidate, alias)
                    if value is not None:
                        fields[public_name] = value
                        break
    return dict(sorted(fields.items()))


def extract_hf_config_fields(config: Any) -> dict[str, Any]:
    """Extract the normalized comparison fields using the HF alias table."""
    return extract_config_fields(config, _HF_CONFIG_FIELD_NAMES)


def extract_megatron_config_fields(config: Any) -> dict[str, Any]:
    """Extract the normalized comparison fields using the Megatron alias table."""
    return extract_config_fields(config, _MEGATRON_CONFIG_FIELD_NAMES)


def compare_tensor_summaries(
    hf_summary: TensorCollectionSummary,
    megatron_summary: TensorCollectionSummary,
) -> list[str]:
    """Describe total and per-dtype element-count differences; empty list means they agree."""
    mismatches: list[str] = []

    if hf_summary.total_numel != megatron_summary.total_numel:
        mismatches.append(
            f"total numel mismatch: hf={hf_summary.total_numel}, megatron={megatron_summary.total_numel}"
        )

    hf_dtypes = set(hf_summary.dtype_stats)
    megatron_dtypes = set(megatron_summary.dtype_stats)
    for dtype_key in sorted(hf_dtypes | megatron_dtypes):
        hf_dtype_summary = hf_summary.dtype_stats.get(dtype_key, DTypeSummary())
        megatron_dtype_summary = megatron_summary.dtype_stats.get(dtype_key, DTypeSummary())
        if hf_dtype_summary.numel != megatron_dtype_summary.numel:
            mismatches.append(
                f"dtype numel mismatch for {dtype_key}: "
                f"hf={hf_dtype_summary.numel}, megatron={megatron_dtype_summary.numel}"
            )

    return mismatches


def compare_logical_tensor_summaries(
    hf_summary: LogicalTensorSummary,
    megatron_summary: LogicalTensorSummary,
) -> list[str]:
    """Describe total and per-category logical element-count differences."""
    mismatches: list[str] = []

    if hf_summary.total_numel != megatron_summary.total_numel:
        mismatches.append(
            f"total logical numel mismatch: hf={hf_summary.total_numel}, megatron={megatron_summary.total_numel}"
        )

    hf_categories = set(hf_summary.category_stats)
    megatron_categories = set(megatron_summary.category_stats)
    for category in sorted(hf_categories | megatron_categories):
        hf_category = hf_summary.category_stats.get(category, LogicalCategorySummary())
        megatron_category = megatron_summary.category_stats.get(category, LogicalCategorySummary())
        if hf_category.numel != megatron_category.numel:
            mismatches.append(
                f"logical numel mismatch for {category}: hf={hf_category.numel}, megatron={megatron_category.numel}"
            )

    return mismatches


def compare_config_fields(hf_fields: Mapping[str, Any], megatron_fields: Mapping[str, Any]) -> list[str]:
    """Describe differing values for config fields present on both sides."""
    mismatches: list[str] = []

    for field_name in sorted(set(hf_fields) & set(megatron_fields)):
        if hf_fields[field_name] != megatron_fields[field_name]:
            mismatches.append(
                f"config mismatch for {field_name}: hf={hf_fields[field_name]}, megatron={megatron_fields[field_name]}"
            )

    return mismatches


def format_num_bytes(num_bytes: int) -> str:
    """Render a byte count with a binary unit suffix."""
    suffixes = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    value = float(num_bytes)
    for suffix in suffixes:
        if value < 1024.0 or suffix == suffixes[-1]:
            return f"{value:.2f} {suffix}"
        value /= 1024.0
    return f"{num_bytes} B"
