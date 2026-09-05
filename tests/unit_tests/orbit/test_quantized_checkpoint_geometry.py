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

"""Regression tests for strict quantized checkpoint geometry and preflight."""

from types import SimpleNamespace

import pytest
import torch
from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory

from megatron.bridge.models.conversion.param_mapping import DirectMapping
from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge
from megatron.bridge.orbit.conversion.modelopt_nvfp4 import ModelOptNVFP4DequantMixin
from megatron.bridge.orbit.low_precision import int4 as int4_low_precision
from megatron.bridge.orbit.low_precision.int4 import transform_sharded_state_dict_for_int4_dense
from megatron.bridge.orbit.low_precision.nvfp4 import (
    build_fused_nvfp4_weight_entries,
    dequantize_nvfp4,
    normalize_weight_scale_2_for_shared_fused_scale,
    quantize_to_nvfp4,
    transform_sharded_state_dict_for_nvfp4_dense,
    validate_nvfp4_weight_bundle,
)
from megatron.bridge.orbit.model_bridges.deepseek_v3_int4_bridge import DeepSeekV3INT4Bridge
from megatron.bridge.orbit.quant.int4_utils import transform_sharded_state_dict_for_int4
from megatron.bridge.orbit.quant.nvfp4_utils import transform_sharded_state_dict_for_nvfp4


pytestmark = pytest.mark.unit


def _dense_shard(key: str, *, out_features: int = 8, in_features: int = 32, rank_offsets=()):
    return ShardedTensor.from_rank_offsets(
        key,
        torch.empty((out_features, in_features), dtype=torch.bfloat16),
        *rank_offsets,
    )


def _layer_axis_dense_shard(shared_key: str, *, layer_index: int = 2, global_layers: int = 4) -> ShardedTensor:
    return ShardedTensor.from_rank_offsets(
        shared_key,
        torch.empty((8, 32), dtype=torch.bfloat16),
        (0, layer_index, global_layers),
        prepend_axis_num=1,
    )


def _layer_expert_axis_shard(
    shared_key: str,
    *,
    layer_index: int = 2,
    global_layers: int = 4,
    expert_index: int = 5,
    global_experts: int = 8,
) -> ShardedTensor:
    return ShardedTensor.from_rank_offsets(
        shared_key,
        torch.empty((8, 32), dtype=torch.bfloat16),
        (0, layer_index, global_layers),
        (1, expert_index, global_experts),
        prepend_axis_num=2,
    )


def _nvfp4_checkpoint_keys(module_path: str, *, split: bool = False) -> set[str]:
    weight_keys = {f"{module_path}.weight_w", f"{module_path}.weight_v"} if split else {f"{module_path}.weight"}
    return weight_keys | {
        f"{module_path}.weight_quantizer._scale",
        f"{module_path}.weight_quantizer._double_scale",
        f"{module_path}.weight_quantizer._amax",
    }


def _assert_strict_quantized_entries(state_dict: dict[str, object]) -> None:
    entries = [value for value in state_dict.values() if hasattr(value, "allow_shape_mismatch")]
    assert entries
    assert all(getattr(entry, "allow_shape_mismatch", None) is False for entry in entries)


@pytest.mark.parametrize("expert", [False, True], ids=["dense", "expert"])
def test_int4_quantized_schema_is_strict_for_aligned_tp_input_shards(expert: bool) -> None:
    if expert:
        key = "decoder.layers.0.mlp.experts.linear_fc2.weight0"
        transform = transform_sharded_state_dict_for_int4
        kwargs = {"group_size": 32}
    else:
        key = "decoder.layers.0.self_attention.linear_proj.weight"
        transform = transform_sharded_state_dict_for_int4_dense
        kwargs = {"group_size": 32}

    source = _dense_shard(key, in_features=32, rank_offsets=((1, 1, 2),))
    transformed = transform({key: source}, **kwargs)

    _assert_strict_quantized_entries(transformed)
    packed = transformed[f"{key}_packed"]
    assert packed.local_shape == (8, 4)
    assert packed.global_shape == (8, 8)
    assert packed.global_offset == (0, 4)
    assert transformed[f"{key}_shape"].dtype == torch.int32


@pytest.mark.parametrize("scope", ["experts", "all"])
def test_direct_int4_save_triplet_dtypes_match_load_schema(scope: str) -> None:
    if scope == "experts":
        target_key = "decoder.layers.0.mlp.experts.linear_fc2.weight0"
        source_key = "model.layers.0.mlp.experts.0.down_proj.weight"
        scale_dtype = torch.float16
        transform = transform_sharded_state_dict_for_int4
    else:
        target_key = "decoder.layers.0.self_attention.linear_proj.weight"
        source_key = "model.layers.0.self_attn.o_proj.weight"
        scale_dtype = torch.bfloat16
        transform = transform_sharded_state_dict_for_int4_dense
    hf_state = {
        f"{source_key}_packed": torch.full((8, 4), 0x11111111, dtype=torch.int32),
        f"{source_key}_scale": torch.full((8, 1), 0.25, dtype=torch.float32),
        f"{source_key}_shape": torch.tensor([8, 32], dtype=torch.int64),
    }
    mapping = DirectMapping(target_key, source_key)
    task = SimpleNamespace(mapping=mapping, megatron_module=object(), param_name=target_key)
    bridge = SimpleNamespace(build_conversion_tasks=lambda hf, model: [task])

    saved = int4_low_precision.build_int4_direct_model_state_dict(
        bridge,
        SimpleNamespace(state=hf_state),
        [object()],
        {},
        group_size=32,
        scale_dtype=scale_dtype,
    )
    load_schema = transform(
        {target_key: _dense_shard(target_key, in_features=32)},
        group_size=32,
        scale_dtype=scale_dtype,
    )

    for suffix in ("_packed", "_scale", "_shape"):
        assert saved[f"{target_key}{suffix}"].dtype == load_schema[f"{target_key}{suffix}"].dtype


@pytest.mark.parametrize("expert", [False, True], ids=["dense", "expert"])
def test_int4_quantized_schema_rejects_group_boundary_inside_tp_shard(expert: bool) -> None:
    if expert:
        key = "decoder.layers.0.mlp.experts.linear_fc2.weight0"
        transform = transform_sharded_state_dict_for_int4
    else:
        key = "decoder.layers.0.self_attention.linear_proj.weight"
        transform = transform_sharded_state_dict_for_int4_dense

    source = _dense_shard(key, in_features=48, rank_offsets=((1, 1, 2),))
    with pytest.raises(ValueError, match="group_size=32"):
        transform({key: source}, group_size=32)


@pytest.mark.parametrize("quant", ["int4", "nvfp4"])
def test_expert_quantized_schema_keeps_local_outer_keys_and_maps_global_layer_and_expert(quant: str) -> None:
    local_key = "decoder.layers.0.mlp.experts.linear_fc2.weight0"
    shared_key = "decoder.layers.mlp.experts.linear_fc2.weight"
    source = _layer_expert_axis_shard(shared_key)

    if quant == "int4":
        transformed = transform_sharded_state_dict_for_int4({local_key: source}, group_size=32)
        packed = transformed[f"{local_key}_packed"]
        scale = transformed[f"{local_key}_scale"]
        assert packed.key == "decoder.layers.2.mlp.experts.linear_fc2.weight5_packed"
        assert scale.key == "decoder.layers.2.mlp.experts.linear_fc2.weight5_scale"
    else:
        transformed = transform_sharded_state_dict_for_nvfp4({local_key: source})
        packed = transformed[local_key]
        local_scale_key = "decoder.layers.0.mlp.experts.linear_fc2.weight_quantizer._scale0"
        scale = transformed[local_scale_key]
        assert packed.key == "decoder.layers.2.mlp.experts.linear_fc2.weight5"
        assert scale.key == "decoder.layers.2.mlp.experts.linear_fc2.weight_quantizer._scale5"

    _assert_strict_quantized_entries(transformed)


@pytest.mark.parametrize("quant", ["int4", "nvfp4"])
def test_dense_quantized_prepended_layer_schema_keeps_local_outer_key_and_maps_global_key(quant: str) -> None:
    local_key = "decoder.layers.0.self_attention.linear_proj.weight"
    shared_key = "decoder.layers.self_attention.linear_proj.weight"
    source = _layer_axis_dense_shard(shared_key)

    if quant == "int4":
        transformed = transform_sharded_state_dict_for_int4_dense({local_key: source}, group_size=32)
        local_packed_key = f"{local_key}_packed"
        checkpoint_packed_key = "decoder.layers.2.self_attention.linear_proj.weight_packed"
        local_scale_key = f"{local_key}_scale"
        checkpoint_scale_key = "decoder.layers.2.self_attention.linear_proj.weight_scale"
    else:
        checkpoint_module = "decoder.layers.2.self_attention.linear_proj"
        transformed = transform_sharded_state_dict_for_nvfp4_dense(
            {local_key: source},
            _nvfp4_checkpoint_keys(checkpoint_module),
        )
        local_packed_key = local_key
        checkpoint_packed_key = f"{checkpoint_module}.weight"
        local_module = local_key.removesuffix(".weight")
        local_scale_key = f"{local_module}.weight_quantizer._scale"
        checkpoint_scale_key = f"{checkpoint_module}.weight_quantizer._scale"

    packed = transformed[local_packed_key]
    scale = transformed[local_scale_key]
    assert packed.key == checkpoint_packed_key
    assert scale.key == checkpoint_scale_key
    _assert_strict_quantized_entries(transformed)


@pytest.mark.parametrize("quant", ["int4", "nvfp4"])
def test_dense_quantized_schema_rejects_missing_fragmentation_metadata(quant: str) -> None:
    key = "decoder.layers.0.self_attention.linear_proj.weight"
    malformed = _dense_shard(key)
    object.__setattr__(malformed, "axis_fragmentations", None)

    with pytest.raises(ValueError, match="axis_fragmentations must be present"):
        if quant == "int4":
            transform_sharded_state_dict_for_int4_dense({key: malformed}, group_size=32)
        else:
            module_path = key.removesuffix(".weight")
            transform_sharded_state_dict_for_nvfp4_dense(
                {key: malformed},
                _nvfp4_checkpoint_keys(module_path),
            )


@pytest.mark.parametrize("expert", [False, True], ids=["dense", "expert"])
def test_nvfp4_quantized_schema_is_strict_for_aligned_tp_input_shards(expert: bool) -> None:
    if expert:
        key = "decoder.layers.0.mlp.experts.linear_fc2.weight0"
        transformed = transform_sharded_state_dict_for_nvfp4(
            {key: _dense_shard(key, in_features=32, rank_offsets=((1, 1, 2),))}
        )
    else:
        key = "decoder.layers.0.self_attention.linear_proj.weight"
        module_path = key.removesuffix(".weight")
        transformed = transform_sharded_state_dict_for_nvfp4_dense(
            {key: _dense_shard(key, in_features=32, rank_offsets=((1, 1, 2),))},
            _nvfp4_checkpoint_keys(module_path),
        )

    _assert_strict_quantized_entries(transformed)


@pytest.mark.parametrize("expert", [False, True], ids=["dense", "expert"])
def test_nvfp4_quantized_schema_rejects_scale_boundary_inside_tp_shard(expert: bool) -> None:
    if expert:
        key = "decoder.layers.0.mlp.experts.linear_fc2.weight0"
        source = _dense_shard(key, in_features=24, rank_offsets=((1, 1, 2),))
        call = lambda: transform_sharded_state_dict_for_nvfp4({key: source})
    else:
        key = "decoder.layers.0.self_attention.linear_proj.weight"
        module_path = key.removesuffix(".weight")
        source = _dense_shard(key, in_features=24, rank_offsets=((1, 1, 2),))
        call = lambda: transform_sharded_state_dict_for_nvfp4_dense({key: source}, _nvfp4_checkpoint_keys(module_path))

    with pytest.raises(ValueError, match="16"):
        call()


def _swiglu_factory(key: str, *, part_count: int = 2, out_features: int = 8) -> ShardedTensorFactory:
    data = torch.empty((out_features, 32), dtype=torch.bfloat16)

    def build(factory_key, tensor, replica_id, flattened_range):
        assert flattened_range is None
        gate, up = torch.chunk(tensor, 2, dim=0)
        parts = [
            ShardedTensor.from_rank_offsets(factory_key, gate, (0, 0, 2), replica_id=replica_id),
            ShardedTensor.from_rank_offsets(factory_key, up, (0, 1, 2), replica_id=replica_id),
        ]
        if part_count == 1:
            return parts[:1]
        if part_count == 3:
            parts.append(ShardedTensor.from_rank_offsets(factory_key, gate, (0, 0, 2), replica_id=replica_id))
        return parts

    return ShardedTensorFactory(
        key=key,
        data=data,
        build_fn=build,
        merge_fn=lambda shards: torch.cat(shards, dim=0),
    )


@pytest.mark.parametrize(
    ("transform_name", "key"),
    [
        ("int4_dense", "decoder.layers.0.mlp.linear_fc1.weight"),
        ("int4_expert", "decoder.layers.0.mlp.experts.linear_fc1.weight0"),
        ("nvfp4_dense", "decoder.layers.0.mlp.linear_fc1.weight"),
        ("nvfp4_expert", "decoder.layers.0.mlp.experts.linear_fc1.weight0"),
    ],
)
@pytest.mark.parametrize("part_count", [1, 3])
def test_quantized_schema_requires_exactly_two_swiglu_factory_parts(
    transform_name: str,
    key: str,
    part_count: int,
) -> None:
    factory = _swiglu_factory(key, part_count=part_count)

    with pytest.raises(ValueError, match="exactly two"):
        if transform_name == "int4_dense":
            transform_sharded_state_dict_for_int4_dense({key: factory}, group_size=32)
        elif transform_name == "int4_expert":
            transform_sharded_state_dict_for_int4({key: factory}, group_size=32)
        elif transform_name == "nvfp4_dense":
            module_path = key.removesuffix(".weight")
            transform_sharded_state_dict_for_nvfp4_dense(
                {key: factory}, _nvfp4_checkpoint_keys(module_path, split=True)
            )
        else:
            transform_sharded_state_dict_for_nvfp4({key: factory})


@pytest.mark.parametrize(
    ("transform_name", "key"),
    [
        ("int4_dense", "decoder.layers.0.mlp.linear_fc1.weight"),
        ("int4_expert", "decoder.layers.0.mlp.experts.linear_fc1.weight0"),
        ("nvfp4_dense", "decoder.layers.0.mlp.linear_fc1.weight"),
        ("nvfp4_expert", "decoder.layers.0.mlp.experts.linear_fc1.weight0"),
    ],
)
def test_quantized_schema_rejects_unequal_swiglu_factory_halves(transform_name: str, key: str) -> None:
    factory = _swiglu_factory(key, out_features=9)

    with pytest.raises(ValueError, match="factory shapes"):
        if transform_name == "int4_dense":
            transform_sharded_state_dict_for_int4_dense({key: factory}, group_size=32)
        elif transform_name == "int4_expert":
            transform_sharded_state_dict_for_int4({key: factory}, group_size=32)
        elif transform_name == "nvfp4_dense":
            module_path = key.removesuffix(".weight")
            transform_sharded_state_dict_for_nvfp4_dense(
                {key: factory},
                _nvfp4_checkpoint_keys(module_path, split=True),
            )
        else:
            transform_sharded_state_dict_for_nvfp4({key: factory})


def test_dense_nvfp4_split_keys_reject_unassociated_output_sharding() -> None:
    key = "decoder.layers.0.mlp.linear_fc1.weight"
    module_path = key.removesuffix(".weight")
    source = _dense_shard(key, out_features=8, rank_offsets=((0, 1, 2),))

    with pytest.raises(ValueError, match="ambiguous.*output sharding"):
        transform_sharded_state_dict_for_nvfp4_dense({key: source}, _nvfp4_checkpoint_keys(module_path, split=True))


def test_dense_nvfp4_checkpoint_family_rejects_partial_quantizer_schema() -> None:
    key = "decoder.layers.0.self_attention.linear_proj.weight"
    module_path = key.removesuffix(".weight")

    with pytest.raises(ValueError, match=r"Incomplete NVFP4 checkpoint family"):
        transform_sharded_state_dict_for_nvfp4_dense(
            {key: _dense_shard(key)},
            {key, f"{module_path}.weight_quantizer._scale"},
        )


def _valid_nvfp4_bundle(*, require_input_scale: bool = True) -> dict[str, torch.Tensor]:
    bundle = {
        "weight": torch.zeros((4, 8), dtype=torch.uint8),
        "weight_scale": torch.ones((4, 1), dtype=torch.float8_e4m3fn),
        "weight_scale_2": torch.tensor(0.5, dtype=torch.float32),
    }
    if require_input_scale:
        bundle["input_scale"] = torch.tensor(0.25, dtype=torch.float32)
    return bundle


def test_nvfp4_bundle_validation_accepts_canonical_direct_bundle() -> None:
    validate_nvfp4_weight_bundle(_valid_nvfp4_bundle(), key="layer.weight", require_input_scale=True)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("weight", torch.zeros((4, 8), dtype=torch.int8), "uint8"),
        ("weight", torch.zeros((4, 8, 1), dtype=torch.uint8), "rank 2"),
        ("weight_scale", torch.ones((4, 1), dtype=torch.float32), "float8_e4m3fn"),
        ("weight_scale", torch.ones((4, 2), dtype=torch.float8_e4m3fn), "packed columns"),
        ("weight_scale_2", torch.ones(1, dtype=torch.float32), "scalar"),
        ("weight_scale_2", torch.tensor(0.0, dtype=torch.float32), "positive"),
        ("input_scale", torch.tensor(float("inf"), dtype=torch.float32), "finite"),
        ("input_scale", torch.tensor(0.0, dtype=torch.float32), "positive"),
    ],
)
def test_nvfp4_bundle_validation_rejects_noncanonical_fields(field, replacement, message) -> None:
    bundle = _valid_nvfp4_bundle()
    bundle[field] = replacement

    with pytest.raises((TypeError, ValueError), match=message):
        validate_nvfp4_weight_bundle(bundle, key="layer.weight", require_input_scale=True)


def test_quantize_to_nvfp4_canonicalizes_zero_weight_scales() -> None:
    packed, scale, scale_2, shape = quantize_to_nvfp4(torch.zeros((4, 16), dtype=torch.bfloat16))

    assert shape.tolist() == [4, 16]
    assert torch.all(scale.float() > 0)
    assert scale_2.item() > 0
    validate_nvfp4_weight_bundle(
        {"weight": packed, "weight_scale": scale, "weight_scale_2": scale_2},
        key="layer.weight",
        require_input_scale=False,
    )
    restored = dequantize_nvfp4(packed, scale, scale_2, shape, dtype=torch.float32)
    assert torch.count_nonzero(restored).item() == 0


def test_quantize_to_nvfp4_cpu_does_not_query_cuda_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_cuda_query(*_args, **_kwargs):
        raise AssertionError("CPU NVFP4 quantization must not query CUDA capability")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected_cuda_query)
    weight = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]])

    packed, scale, scale_2, shape = quantize_to_nvfp4(weight)
    restored = dequantize_nvfp4(packed, scale, scale_2, shape, dtype=torch.float32)

    assert packed.device.type == "cpu"
    assert scale.device.type == "cpu"
    assert scale_2.device.type == "cpu"
    assert shape.device.type == "cpu"
    torch.testing.assert_close(restored, weight, rtol=0.0, atol=0.0)


def test_nvfp4_double_scale_normalization_returns_positive_canonical_fp8() -> None:
    bundles = {
        "gate": {**_valid_nvfp4_bundle(), "weight_scale_2": torch.tensor(1.0)},
        "up": {**_valid_nvfp4_bundle(), "weight_scale_2": torch.tensor(1e-8)},
    }

    normalized, shared = normalize_weight_scale_2_for_shared_fused_scale(
        bundles,
        megatron_weight_key="decoder.layers.0.mlp.linear_fc1.weight",
    )

    assert shared.dtype == torch.float32
    for bundle in normalized.values():
        scale = bundle["weight_scale"]
        assert scale.dtype == torch.float8_e4m3fn
        assert torch.all(scale.float().isfinite())
        assert torch.all(scale.float() > 0)


def _normalization_bundle(
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: float,
) -> dict[str, torch.Tensor]:
    return {
        "weight": packed.to(torch.uint8),
        "weight_scale": weight_scale.to(torch.float8_e4m3fn),
        "weight_scale_2": torch.tensor(weight_scale_2, dtype=torch.float32),
    }


def test_nvfp4_double_scale_normalization_accepts_exact_half_ulp_boundary() -> None:
    min_positive = 2.0**-9
    exact_boundary_scale = min_positive / (1.0 + 2.0**-4)
    packed = torch.full((1, 8), 0x11, dtype=torch.uint8)
    bundles = {
        "shared": _normalization_bundle(packed, torch.ones((1, 1)), 1.0),
        "boundary": _normalization_bundle(packed, torch.ones((1, 1)), exact_boundary_scale),
    }

    normalized, shared = normalize_weight_scale_2_for_shared_fused_scale(
        bundles,
        megatron_weight_key="decoder.layers.0.mlp.linear_fc1.weight",
    )

    assert shared.item() == 1.0
    assert normalized["boundary"]["weight_scale"].float().item() == min_positive


def test_nvfp4_double_scale_normalization_rejects_beyond_half_ulp_boundary() -> None:
    min_positive = 2.0**-9
    beyond_boundary_scale = min_positive / (1.0 + 2.0**-4) * (1.0 - 1e-3)
    packed = torch.full((1, 8), 0x11, dtype=torch.uint8)
    bundles = {
        "shared": _normalization_bundle(packed, torch.ones((1, 1)), 1.0),
        "beyond": _normalization_bundle(packed, torch.ones((1, 1)), beyond_boundary_scale),
    }

    with pytest.raises(ValueError, match="half-ULP bound"):
        normalize_weight_scale_2_for_shared_fused_scale(
            bundles,
            megatron_weight_key="decoder.layers.0.mlp.linear_fc1.weight",
        )


def test_nvfp4_double_scale_normalization_associates_activity_with_each_scale_group() -> None:
    min_positive = 2.0**-9
    negative_zero_group = torch.full((1, 8), 0x88, dtype=torch.uint8)
    nonzero_group = torch.full((1, 8), 0x11, dtype=torch.uint8)
    shared = _normalization_bundle(
        torch.cat((nonzero_group, nonzero_group), dim=1),
        torch.ones((1, 2)),
        1.0,
    )
    scales = torch.tensor([[min_positive, 1.0]], dtype=torch.float32)

    zero_underflow = _normalization_bundle(
        torch.cat((negative_zero_group, nonzero_group), dim=1),
        scales,
        0.5,
    )
    normalized, _ = normalize_weight_scale_2_for_shared_fused_scale(
        {"shared": shared, "candidate": zero_underflow},
        megatron_weight_key="decoder.layers.0.mlp.linear_fc1.weight",
    )
    assert normalized["candidate"]["weight_scale"].float().tolist() == [[min_positive, 0.5]]

    nonzero_underflow = _normalization_bundle(
        torch.cat((nonzero_group, negative_zero_group), dim=1),
        scales,
        0.5,
    )
    with pytest.raises(ValueError, match="half-ULP bound"):
        normalize_weight_scale_2_for_shared_fused_scale(
            {"shared": shared, "candidate": nonzero_underflow},
            megatron_weight_key="decoder.layers.0.mlp.linear_fc1.weight",
        )


def test_split_nvfp4_builder_rejects_wrong_merged_scale_output_shape() -> None:
    class WrongScaleMapping:
        def hf_to_megatron(self, weights, module):
            del module
            first = next(iter(weights.values()))
            return torch.ones((7, first.shape[1]), dtype=first.dtype)

    with pytest.raises(ValueError, match="packed and scale rows differ"):
        build_fused_nvfp4_weight_entries(
            WrongScaleMapping(),
            "decoder.layers.0.mlp.linear_fc1.weight",
            {"gate": _valid_nvfp4_bundle(), "up": _valid_nvfp4_bundle()},
            object(),
            split_swiglu_weight=True,
        )


class _PlainNVFP4Base:
    def maybe_modify_loaded_hf_weight(self, hf_param, hf_state_dict):
        if isinstance(hf_param, dict):
            return {role: hf_state_dict[key] for role, key in hf_param.items()}
        return hf_state_dict[hf_param]


class _NVFP4Bridge(ModelOptNVFP4DequantMixin, _PlainNVFP4Base):
    pass


def test_generic_nvfp4_preflights_mixed_complete_and_partial_sources_before_dequant(monkeypatch) -> None:
    complete = _valid_nvfp4_bundle(require_input_scale=False)
    state = {
        "q.weight": complete["weight"],
        "q.weight_scale": complete["weight_scale"],
        "q.weight_scale_2": complete["weight_scale_2"],
        "k.weight": torch.zeros((4, 8), dtype=torch.uint8),
        "k.weight_scale": torch.ones((4, 1), dtype=torch.float8_e4m3fn),
    }
    calls = []
    monkeypatch.setattr(
        "megatron.bridge.orbit.low_precision.nvfp4.dequantize_nvfp4",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match=r"k\.weight.*weight_scale_2"):
        _NVFP4Bridge().maybe_modify_loaded_hf_weight({"q": "q.weight", "k": "k.weight"}, state)
    assert calls == []


def test_generic_nvfp4_preflights_later_task_family_before_first_dequant(monkeypatch) -> None:
    complete = _valid_nvfp4_bundle(require_input_scale=False)
    state = {
        "q.weight": complete["weight"],
        "q.weight_scale": complete["weight_scale"],
        "q.weight_scale_2": complete["weight_scale_2"],
        "later.weight": torch.zeros((4, 8), dtype=torch.uint8),
        "later.weight_scale": torch.ones((4, 1), dtype=torch.float8_e4m3fn),
    }
    calls = []
    monkeypatch.setattr(
        "megatron.bridge.orbit.low_precision.nvfp4.dequantize_nvfp4",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match=r"later\.weight.*weight_scale_2"):
        _NVFP4Bridge().maybe_modify_loaded_hf_weight("q.weight", state)
    assert calls == []


def test_generic_nvfp4_preflights_later_malformed_complete_family_before_first_dequant(monkeypatch) -> None:
    complete = _valid_nvfp4_bundle(require_input_scale=False)
    state = {
        "q.weight": complete["weight"],
        "q.weight_scale": complete["weight_scale"],
        "q.weight_scale_2": complete["weight_scale_2"],
        "later.weight": torch.zeros((4, 8), dtype=torch.uint8),
        "later.weight_scale": torch.ones((4, 1), dtype=torch.float32),
        "later.weight_scale_2": torch.tensor(0.5, dtype=torch.float32),
    }
    calls = []
    monkeypatch.setattr(
        "megatron.bridge.orbit.low_precision.nvfp4.dequantize_nvfp4",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(TypeError, match=r"later\.weight.*float8_e4m3fn"):
        _NVFP4Bridge().maybe_modify_loaded_hf_weight("q.weight", state)
    assert calls == []


def test_generic_nvfp4_rejects_uint8_weight_without_scale_siblings() -> None:
    state = {"layer.weight": torch.zeros((4, 8), dtype=torch.uint8)}

    with pytest.raises(ValueError, match=r"layer\.weight.*weight_scale.*weight_scale_2"):
        _NVFP4Bridge().maybe_modify_loaded_hf_weight("layer.weight", state)


def test_direct_nvfp4_preflight_treats_orphan_input_scale_as_partial_family() -> None:
    from megatron.bridge.orbit.low_precision.nvfp4 import preflight_nvfp4_source_families

    with pytest.raises(ValueError, match=r"layer\.weight.*layer\.weight.*weight_scale.*weight_scale_2"):
        preflight_nvfp4_source_families(
            {"layer.input_scale": torch.tensor(1.0, dtype=torch.float32)},
            require_input_scale=True,
        )


def _deepseek_model(*, etp: int = 1, tp: int = 1):
    return SimpleNamespace(config=SimpleNamespace(expert_tensor_parallel_size=etp, tensor_model_parallel_size=tp))


def _bridge_with_parent_sentinel(monkeypatch, events: list[str]) -> DeepSeekV3INT4Bridge:
    bridge = DeepSeekV3INT4Bridge.__new__(DeepSeekV3INT4Bridge)

    def parent_load(self, hf_pretrained, megatron_model, allowed_mismatched_params=None):
        events.append("parent")
        return megatron_model if isinstance(megatron_model, list) else [megatron_model]

    monkeypatch.setattr(DeepSeekV3Bridge, "load_weights_hf_to_megatron", parent_load)
    return bridge


def test_deepseek_int4_rejects_etp_before_parent_mutation(monkeypatch) -> None:
    events = []
    bridge = _bridge_with_parent_sentinel(monkeypatch, events)

    with pytest.raises(ValueError, match="expert tensor parallel.*1"):
        bridge.load_weights_hf_to_megatron(SimpleNamespace(state={}), _deepseek_model(etp=2))
    assert events == []


def test_deepseek_int4_rejects_multiple_vp_chunks_before_parent_mutation(monkeypatch) -> None:
    events = []
    bridge = _bridge_with_parent_sentinel(monkeypatch, events)

    with pytest.raises(ValueError, match="single model chunk"):
        bridge.load_weights_hf_to_megatron(
            SimpleNamespace(state={}),
            [_deepseek_model(), _deepseek_model()],
        )
    assert events == []


def test_deepseek_int4_rejects_missing_expert_triplet_before_parent_mutation(monkeypatch) -> None:
    events = []
    bridge = _bridge_with_parent_sentinel(monkeypatch, events)
    task = SimpleNamespace(
        param_name="decoder.layers.0.mlp.experts.linear_fc2.weight0",
        megatron_module=object(),
        mapping=SimpleNamespace(hf_param="model.layers.0.mlp.experts.down_proj.weight", tp_size=1),
    )
    monkeypatch.setattr(bridge, "build_conversion_tasks", lambda hf, models: [task])

    with pytest.raises(ValueError, match="missing or unsupported INT4 triplet"):
        bridge.load_weights_hf_to_megatron(SimpleNamespace(state={}), _deepseek_model())
    assert events == []


def test_deepseek_int4_collects_and_validates_complete_expert_source_scale() -> None:
    bridge = DeepSeekV3INT4Bridge.__new__(DeepSeekV3INT4Bridge)
    source_key = "model.layers.0.mlp.experts.0.down_proj.weight"
    target_key = "decoder.layers.0.mlp.experts.linear_fc2.weight0"
    mapping = DirectMapping(target_key, source_key)
    task = SimpleNamespace(param_name=target_key, megatron_module=object(), mapping=mapping)
    source_scale = torch.ones((2, 1), dtype=torch.float16)
    hf_pretrained = SimpleNamespace(
        state={
            f"{source_key}_packed": torch.zeros((2, 4), dtype=torch.int32),
            f"{source_key}_scale": source_scale,
            f"{source_key}_shape": torch.tensor([2, 32], dtype=torch.int64),
        }
    )

    scales = bridge._collect_source_scales(
        hf_pretrained,
        [_deepseek_model()],
        conversion_tasks=[task],
    )

    assert set(scales) == {target_key}
    assert scales[target_key] is source_scale


def test_deepseek_int4_rejects_non_fp16_source_scale_before_parent_mutation(monkeypatch) -> None:
    events = []
    bridge = _bridge_with_parent_sentinel(monkeypatch, events)
    source_key = "model.layers.0.mlp.experts.0.down_proj.weight"
    target_key = "decoder.layers.0.mlp.experts.linear_fc2.weight0"
    mapping = DirectMapping(target_key, source_key)
    task = SimpleNamespace(param_name=target_key, megatron_module=object(), mapping=mapping)
    hf_pretrained = SimpleNamespace(
        state={
            f"{source_key}_packed": torch.zeros((2, 4), dtype=torch.int32),
            f"{source_key}_scale": torch.ones((2, 1), dtype=torch.bfloat16),
            f"{source_key}_shape": torch.tensor([2, 32], dtype=torch.int32),
        }
    )
    monkeypatch.setattr(bridge, "build_conversion_tasks", lambda hf, models: [task])
    monkeypatch.setattr(
        bridge,
        "_requantize_experts_int4",
        lambda *args, **kwargs: events.append("requantize"),
    )

    with pytest.raises(ValueError, match=r"weight_scale.*torch\.float16"):
        bridge.load_weights_hf_to_megatron(hf_pretrained, _deepseek_model())

    assert events == []


def test_deepseek_standard_int4_writer_matches_expert_load_schema() -> None:
    key = "decoder.layers.0.mlp.experts.linear_fc2.weight0"
    module = torch.nn.Linear(32, 2, bias=False, dtype=torch.bfloat16)
    module.weight.data.copy_(torch.arange(64, dtype=torch.bfloat16).reshape(2, 32) / 16)
    source_scale = torch.tensor([[0.5], [0.25]], dtype=torch.float16)
    bridge = DeepSeekV3INT4Bridge.__new__(DeepSeekV3INT4Bridge)

    bridge._quantize_one_weight(
        module,
        "weight",
        module.weight.data,
        group_size=32,
        source_scale=source_scale,
    )
    load_schema = transform_sharded_state_dict_for_int4(
        {key: _dense_shard(key, out_features=2, in_features=32)},
        group_size=32,
    )

    assert module.weight_packed.dtype == load_schema[f"{key}_packed"].dtype == torch.int32
    assert module.weight_scale.dtype == load_schema[f"{key}_scale"].dtype == torch.float16
    assert module.weight_shape.dtype == load_schema[f"{key}_shape"].dtype == torch.int32
    torch.testing.assert_close(module.weight_scale, source_scale, rtol=0, atol=0)


def test_deepseek_standard_int4_writer_fallback_matches_expert_scale_schema() -> None:
    key = "decoder.layers.0.mlp.experts.linear_fc2.weight0"
    module = torch.nn.Linear(32, 2, bias=False, dtype=torch.bfloat16)
    module.weight.data.copy_(torch.arange(64, dtype=torch.bfloat16).reshape(2, 32) / 16)
    bridge = DeepSeekV3INT4Bridge.__new__(DeepSeekV3INT4Bridge)

    bridge._quantize_one_weight(
        module,
        "weight",
        module.weight.data,
        group_size=32,
        source_scale=None,
    )
    load_schema = transform_sharded_state_dict_for_int4(
        {key: _dense_shard(key, out_features=2, in_features=32)},
        group_size=32,
    )

    assert module.weight_scale.dtype == load_schema[f"{key}_scale"].dtype == torch.float16


def test_deepseek_int4_tp2_etp1_collects_real_source_scales_before_requantize(monkeypatch) -> None:
    events = []
    bridge = _bridge_with_parent_sentinel(monkeypatch, events)
    model = _deepseek_model(etp=1, tp=2)
    source_key = "model.layers.0.mlp.experts.0.down_proj.weight"
    target_key = "decoder.layers.0.mlp.experts.linear_fc2.weight0"
    source_scale = torch.ones((2, 1), dtype=torch.float16)
    mapping = DirectMapping(target_key, source_key)
    tp_group = object()
    etp_group = object()
    mapping._tp_group = tp_group
    mapping._etp_group = etp_group
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.param_mapping.get_pg_size",
        lambda group: {tp_group: 2, etp_group: 1}[group],
    )
    assert mapping.is_expert
    assert mapping.tp_group is etp_group
    assert mapping.tp_size == 1
    task = SimpleNamespace(param_name=target_key, megatron_module=object(), mapping=mapping)
    hf_pretrained = SimpleNamespace(
        state={
            f"{source_key}_packed": torch.zeros((2, 4), dtype=torch.int32),
            f"{source_key}_scale": source_scale,
            f"{source_key}_shape": torch.tensor([2, 32], dtype=torch.int32),
        }
    )
    monkeypatch.setattr(bridge, "build_conversion_tasks", lambda hf, models: [task])

    captured_scales = []
    monkeypatch.setattr(
        bridge,
        "_requantize_experts_int4",
        lambda actual_model, source_scales=None: events.append("requantize")
        or captured_scales.append((actual_model, source_scales)),
    )

    result = bridge.load_weights_hf_to_megatron(hf_pretrained, model)

    assert result == [model]
    assert events == ["parent", "requantize"]
    assert captured_scales == [(model, {target_key: source_scale})]
