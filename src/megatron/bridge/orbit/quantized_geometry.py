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

"""Strict geometry checks shared by packed INT4 and NVFP4 checkpoint schemas."""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class SwiGLUFactoryGeometry:
    """Validated fused geometry recovered from a two-way SwiGLU factory."""

    sharded_tensor: Any
    sub_tensors: tuple[Any, Any]
    local_out: int
    global_out: int
    out_offset: int
    axis_fragmentations: tuple[int, ...]
    same_key_splits: bool


def validate_sharded_weight_metadata(
    sharded_tensor: Any,
    *,
    key: str,
) -> None:
    """Validate ranks needed to interpret the final two axes as a weight."""

    local_shape = tuple(getattr(sharded_tensor, "local_shape", ()))
    global_shape = tuple(getattr(sharded_tensor, "global_shape", ()))
    global_offset = tuple(getattr(sharded_tensor, "global_offset", ()))
    raw_axis_fragmentations = getattr(sharded_tensor, "axis_fragmentations", None)
    if raw_axis_fragmentations is None:
        raise ValueError(f"Invalid quantized checkpoint geometry for {key}: axis_fragmentations must be present")
    axis_fragmentations = tuple(raw_axis_fragmentations)
    prepend = _nonnegative_index(
        getattr(sharded_tensor, "prepend_axis_num", None),
        name="prepend axis",
        key=key,
    )
    if len(local_shape) != 2:
        raise ValueError(
            f"Invalid quantized checkpoint geometry for {key}: local weight rank must be 2, got {local_shape}"
        )
    expected_global_rank = prepend + 2
    if not (
        len(global_shape) == expected_global_rank
        and len(global_offset) == expected_global_rank
        and len(axis_fragmentations) == expected_global_rank
    ):
        raise ValueError(
            f"Invalid quantized checkpoint geometry for {key}: expected global metadata rank "
            f"{expected_global_rank}, got shape={len(global_shape)}, offset={len(global_offset)}, "
            f"fragmentation={len(axis_fragmentations)}"
        )
    for axis in range(prepend):
        global_size = _positive_index(global_shape[axis], name=f"prepended axis {axis} size", key=key)
        offset = _nonnegative_index(global_offset[axis], name=f"prepended axis {axis} offset", key=key)
        fragments = _positive_index(
            axis_fragmentations[axis],
            name=f"prepended axis {axis} fragmentation",
            key=key,
        )
        if offset >= global_size or fragments != global_size:
            raise ValueError(
                f"Invalid quantized checkpoint geometry for {key}: prepended axis {axis} describes "
                f"an implicit unit shard, so offset {offset} must be below size {global_size} and "
                f"fragmentation must equal {global_size}, got {fragments}"
            )


def resolve_dense_layer_index(
    sharded_tensor: Any,
    *,
    key: str,
) -> int | None:
    """Return the global layer index carried by one prepended metadata axis."""

    validate_sharded_weight_metadata(sharded_tensor, key=key)
    prepend = _nonnegative_index(
        getattr(sharded_tensor, "prepend_axis_num", None),
        name="prepend axis",
        key=key,
    )
    if prepend == 0:
        return None

    global_shape = tuple(sharded_tensor.global_shape)
    global_offset = tuple(sharded_tensor.global_offset)
    if prepend != 1:
        raise ValueError(
            f"Invalid quantized checkpoint geometry for {key}: a dense layer weight requires exactly "
            f"one prepended layer axis, got {prepend}"
        )
    if "layers." not in key:
        raise ValueError(
            f"Invalid quantized checkpoint geometry for {key}: cannot unroll a stacked layer "
            "without a 'layers.' key segment"
        )
    layer_offset = _nonnegative_index(global_offset[0], name="layer offset", key=key)
    if layer_offset >= _positive_index(global_shape[0], name="global layer count", key=key):
        raise ValueError(
            f"Invalid quantized checkpoint geometry for {key}: layer offset {layer_offset} "
            f"is outside global layer count {global_shape[0]}"
        )
    return layer_offset


def resolve_expert_layer_index(sharded_tensor: Any, *, key: str) -> int | None:
    """Return the layer coordinate preceding a grouped expert coordinate."""

    validate_sharded_weight_metadata(sharded_tensor, key=key)
    prepend = _nonnegative_index(
        getattr(sharded_tensor, "prepend_axis_num", None),
        name="prepend axis",
        key=key,
    )
    if prepend <= 1:
        return None
    if prepend != 2:
        raise ValueError(
            f"Invalid quantized checkpoint geometry for {key}: grouped expert weights support "
            f"only (expert) or (layer, expert) prepended axes, got {prepend} axes"
        )
    if "layers." not in key:
        raise ValueError(
            f"Invalid quantized checkpoint geometry for {key}: a prepended expert layer axis "
            "requires a 'layers.' key segment"
        )
    return _nonnegative_index(sharded_tensor.global_offset[0], name="layer offset", key=key)


def rewrite_dense_layer_key(key: str, global_layer_index: int | None) -> str:
    """Replace an enclosing local layer index with its checkpoint-global index."""

    if global_layer_index is None:
        return key
    rewritten, substitutions = re.subn(
        r"((?:^|\.)layers\.)(?:\d+\.)?",
        rf"\g<1>{global_layer_index}.",
        key,
        count=1,
    )
    if substitutions != 1:
        raise ValueError(
            f"Invalid quantized checkpoint geometry for {key}: cannot apply global layer index {global_layer_index}"
        )
    return rewritten


def _positive_index(value: Any, *, name: str, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid quantized checkpoint geometry for {key}: {name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(
            f"Invalid quantized checkpoint geometry for {key}: {name} must be a positive integer, got {value!r}"
        ) from exc
    if result <= 0:
        raise ValueError(f"Invalid quantized checkpoint geometry for {key}: {name} must be positive, got {result}")
    return result


def _nonnegative_index(value: Any, *, name: str, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid quantized checkpoint geometry for {key}: {name} must be a non-negative integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(
            f"Invalid quantized checkpoint geometry for {key}: {name} must be a non-negative integer, got {value!r}"
        ) from exc
    if result < 0:
        raise ValueError(f"Invalid quantized checkpoint geometry for {key}: {name} must be non-negative, got {result}")
    return result


def validate_quantized_shard_geometry(
    *,
    key: str,
    local_shape: Iterable[int],
    global_shape: Iterable[int],
    global_offset: Iterable[int],
    axis_fragmentations: Iterable[int],
    packing_factor: int,
    group_size: int,
) -> None:
    """Validate one logical 2-D weight shard before packed-grid division.

    Quantized checkpoint tensors cannot use the dense checkpoint loader's
    overlap/zero-fill compatibility mode. Their payload and scale grids are
    meaningful only when every shard is a regular, exactly aligned tile.
    """

    packing_factor = _positive_index(packing_factor, name="packing_factor", key=key)
    group_size = _positive_index(group_size, name="group_size", key=key)
    local_shape = tuple(local_shape)
    global_shape = tuple(global_shape)
    global_offset = tuple(global_offset)
    axis_fragmentations = tuple(axis_fragmentations)

    for name, value in (
        ("local_shape", local_shape),
        ("global_shape", global_shape),
        ("global_offset", global_offset),
        ("axis_fragmentations", axis_fragmentations),
    ):
        if len(value) != 2:
            raise ValueError(f"Invalid quantized checkpoint geometry for {key}: {name} must have rank 2, got {value}")

    local_out = _positive_index(local_shape[0], name="local output rows", key=key)
    local_in = _positive_index(local_shape[1], name="local input width", key=key)
    global_out = _positive_index(global_shape[0], name="global output rows", key=key)
    global_in = _positive_index(global_shape[1], name="global input width", key=key)
    out_offset = _nonnegative_index(global_offset[0], name="output offset", key=key)
    in_offset = _nonnegative_index(global_offset[1], name="input offset", key=key)
    out_fragments = _positive_index(axis_fragmentations[0], name="output fragmentation", key=key)
    in_fragments = _positive_index(axis_fragmentations[1], name="input fragmentation", key=key)

    for local, global_, offset, fragments, axis_name in (
        (local_out, global_out, out_offset, out_fragments, "output"),
        (local_in, global_in, in_offset, in_fragments, "input"),
    ):
        if offset + local > global_:
            raise ValueError(
                f"Invalid quantized checkpoint geometry for {key}: {axis_name} shard "
                f"[{offset}, {offset + local}) exceeds global size {global_}"
            )
        if global_ % local != 0:
            raise ValueError(
                f"Invalid quantized checkpoint geometry for {key}: global {axis_name} size {global_} "
                f"is not an exact multiple of local size {local}"
            )
        expected_fragments = global_ // local
        if fragments != expected_fragments:
            raise ValueError(
                f"Invalid quantized checkpoint geometry for {key}: {axis_name} fragmentation {fragments} "
                f"does not match the exact shard grid {expected_fragments}"
            )
        if offset % local != 0:
            raise ValueError(
                f"Invalid quantized checkpoint geometry for {key}: {axis_name} offset {offset} "
                f"is not aligned to local size {local}"
            )

    for value, divisor, label in (
        (local_in, packing_factor, "local input width"),
        (global_in, packing_factor, "global input width"),
        (in_offset, packing_factor, "input offset"),
        (local_in, group_size, "local input width"),
        (global_in, group_size, "global input width"),
        (in_offset, group_size, "input offset"),
    ):
        if value % divisor != 0:
            raise ValueError(
                f"Invalid quantized checkpoint geometry for {key}: {label} {value} must be divisible by "
                f"{'packing factor' if divisor == packing_factor else 'group_size'}={divisor}"
            )


def reconstruct_swiglu_factory_geometry(factory: Any, *, key: str) -> SwiGLUFactoryGeometry:
    """Validate and reconstruct a factory that splits one fused weight in two."""

    built = factory.build()
    sub_tensors = list(built) if isinstance(built, list) else list(built.values())
    if len(sub_tensors) != 2:
        raise ValueError(
            f"Unexpected SwiGLU factory for {key}: expected exactly two gate/up shards, got {len(sub_tensors)}"
        )
    gate, up = sub_tensors

    data_shape = tuple(getattr(getattr(factory, "data", None), "shape", ()))
    if len(data_shape) != 2:
        raise ValueError(f"Unexpected SwiGLU factory data rank for {key}: expected rank 2, got {data_shape}")

    for role, sub_tensor in (("gate", gate), ("up", up)):
        validate_sharded_weight_metadata(sub_tensor, key=f"{key} ({role})")

    gate_local = tuple(gate.local_shape)
    up_local = tuple(up.local_shape)
    if gate_local != up_local:
        raise ValueError(f"Unexpected SwiGLU factory shapes for {key}: gate={gate_local}, up={up_local}")
    if data_shape != (gate_local[0] * 2, gate_local[1]):
        raise ValueError(
            f"Unexpected SwiGLU factory output shapes for {key}: fused={data_shape}, split={gate_local}; "
            "the split must be exactly two equal output halves"
        )

    prepend = gate.prepend_axis_num
    if up.prepend_axis_num != prepend:
        raise ValueError(f"Unexpected SwiGLU factory prepend axes for {key}: gate={prepend}, up={up.prepend_axis_num}")
    out_axis = prepend
    gate_global = tuple(gate.global_shape)
    up_global = tuple(up.global_shape)
    gate_offset = tuple(gate.global_offset)
    up_offset = tuple(up.global_offset)
    gate_fragments = tuple(gate.axis_fragmentations)
    up_fragments = tuple(up.axis_fragmentations)
    split_local_out = _positive_index(gate_local[0], name="split local output rows", key=key)
    if gate_offset[out_axis] % split_local_out != 0:
        raise ValueError(
            f"Unexpected SwiGLU factory output offset for {key}: {gate_offset[out_axis]} is not aligned "
            f"to split output size {split_local_out}"
        )

    factory_key = getattr(factory, "key", None)
    same_key_splits = gate.key == factory_key and up.key == factory_key
    distinct_key_splits = gate.key != up.key and gate.key != factory_key and up.key != factory_key
    if not (same_key_splits or distinct_key_splits):
        raise ValueError(
            f"Unexpected SwiGLU factory keys for {key}: gate={gate.key!r}, up={up.key!r}, factory={factory_key!r}"
        )

    if same_key_splits:
        if gate_global != up_global or gate_fragments != up_fragments:
            raise ValueError(f"Unexpected same-key SwiGLU metadata mismatch for {key}")
        for axis in range(len(gate_global)):
            if axis != out_axis and gate_offset[axis] != up_offset[axis]:
                raise ValueError(f"Unexpected same-key SwiGLU non-output offsets for {key}")
        split_global_out = _positive_index(gate_global[out_axis], name="global output rows", key=key)
        if split_global_out % 2 != 0:
            raise ValueError(f"Unexpected SwiGLU global output for {key}: {split_global_out} must be even")
        if up_offset[out_axis] != gate_offset[out_axis] + split_global_out // 2:
            raise ValueError(
                f"Unexpected same-key SwiGLU output offsets for {key}: "
                f"gate={gate_offset[out_axis]}, up={up_offset[out_axis]}"
            )
        split_fragmentation = _positive_index(gate_fragments[out_axis], name="split output fragmentation", key=key)
        if split_fragmentation % 2 != 0:
            raise ValueError(
                f"Unexpected SwiGLU factory fragmentation for {key}: {split_fragmentation} must be divisible by 2"
            )
        global_out = split_global_out
        axis_fragmentations = list(gate_fragments)
        axis_fragmentations[out_axis] = split_fragmentation // 2
    else:
        if gate_global != up_global or gate_offset != up_offset or gate_fragments != up_fragments:
            raise ValueError(f"Unexpected split-key SwiGLU metadata mismatch for {key}")
        global_out = _positive_index(gate_global[out_axis], name="split global output rows", key=key) * 2
        axis_fragmentations = list(gate_fragments)

    rank_offset = gate_offset[out_axis] // split_local_out
    fused_local_out = gate_local[0] * 2
    return SwiGLUFactoryGeometry(
        sharded_tensor=gate,
        sub_tensors=(gate, up),
        local_out=fused_local_out,
        global_out=global_out,
        out_offset=rank_offset * fused_local_out,
        axis_fragmentations=tuple(axis_fragmentations),
        same_key_splits=same_key_splits,
    )
