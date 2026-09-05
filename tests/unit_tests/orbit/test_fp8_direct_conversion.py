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

import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory
from safetensors.torch import save_file

from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource, StateDict
from megatron.bridge.orbit.low_precision import fp8 as fp8_low_precision
from megatron.bridge.orbit.quant.fp8_utils import register_fp8_scale_inv_buffers_after_load


pytestmark = pytest.mark.unit


def _fp8_weight(shape: tuple[int, int] = (129, 129), *, dtype: torch.dtype = torch.float8_e4m3fn):
    return torch.ones(shape, dtype=torch.float32).to(dtype)


def _task(target_key: str, source_key: str):
    mapping = fp8_low_precision.DirectMapping(target_key, source_key)
    return SimpleNamespace(mapping=mapping, megatron_module=object(), param_name=target_key)


def _gated_task(target_key: str, gate_key: str, up_key: str):
    mapping = fp8_low_precision.GatedMLPMapping(target_key, gate_key, up_key)
    module = SimpleNamespace(config=SimpleNamespace(gated_linear_unit=True))
    return SimpleNamespace(mapping=mapping, megatron_module=module, param_name=target_key)


def _swiglu_factory(
    runtime_key: str,
    physical_key: str,
    *,
    expert_offset: tuple[int, int, int] | None = None,
):
    offsets = () if expert_offset is None else (expert_offset,)
    template_data = torch.empty((256, 128), dtype=torch.bfloat16)
    sharded_tensor = ShardedTensor.from_rank_offsets(
        physical_key,
        template_data,
        *offsets,
        prepend_axis_num=len(offsets),
    )
    return apply_swiglu_sharded_factory(sharded_tensor, offsets, singleton_local_shards=False)


def _nested_module(module_path: str) -> torch.nn.Module:
    root = torch.nn.Module()
    current = root
    for segment in module_path.split("."):
        child = torch.nn.Module()
        current.add_module(segment, child)
        current = child
    return root


def _bridge(_conversion_tasks):
    def load(hf_param, state):
        if isinstance(hf_param, dict):
            return {role: state[key] for role, key in hf_param.items()}
        return state[hf_param]

    return SimpleNamespace(
        build_conversion_tasks=lambda hf, model: pytest.fail("precomputed conversion tasks must be reused"),
        maybe_modify_loaded_hf_weight=load,
    )


def _capture_modelopt_quant_config(
    monkeypatch: pytest.MonkeyPatch,
    preset: dict[str, object],
    module_names: set[str],
) -> dict[str, object]:
    modelopt_module = ModuleType("modelopt")
    modelopt_torch_module = ModuleType("modelopt.torch")
    quantization_module = ModuleType("modelopt.torch.quantization")
    quantization_module.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG = deepcopy(preset)
    captured: dict[str, object] = {}

    def quantize(module, quant_cfg, forward_loop):
        captured["module"] = module
        captured["quant_cfg"] = quant_cfg
        captured["forward_loop"] = forward_loop

    quantization_module.quantize = quantize
    modelopt_module.torch = modelopt_torch_module
    modelopt_torch_module.quantization = quantization_module
    monkeypatch.setitem(sys.modules, "modelopt", modelopt_module)
    monkeypatch.setitem(sys.modules, "modelopt.torch", modelopt_torch_module)
    monkeypatch.setitem(sys.modules, "modelopt.torch.quantization", quantization_module)

    model = object()
    fp8_low_precision.apply_modelopt_fp8_to_meta_model(model, module_names=module_names)
    assert captured["module"] is model
    return captured["quant_cfg"]


@pytest.mark.parametrize("scale_suffix", ["_scale_inv", "_scale"])
@pytest.mark.parametrize("bad_value", [0.0, -1.0, float("nan"), float("inf")])
def test_fp8_source_family_rejects_nonpositive_or_nonfinite_scale(
    scale_suffix: str,
    bad_value: float,
) -> None:
    source_key = "model.layers.0.self_attn.o_proj.weight"
    task = _task("decoder.layers.0.self_attention.linear_proj.weight", source_key)
    state = {
        source_key: _fp8_weight(),
        f"{source_key}{scale_suffix}": torch.tensor(bad_value, dtype=torch.float32),
    }

    with pytest.raises(ValueError, match=r"scale.*finite and positive"):
        fp8_low_precision.collect_fp8_target_module_names([task], state)


def test_fp8_source_family_rejects_nonfloating_scale() -> None:
    source_key = "model.layers.0.self_attn.o_proj.weight"
    task = _task("decoder.layers.0.self_attention.linear_proj.weight", source_key)
    state = {
        source_key: _fp8_weight(),
        f"{source_key}_scale_inv": torch.ones((2, 2), dtype=torch.int32),
    }

    with pytest.raises(TypeError, match=r"scale.*floating"):
        fp8_low_precision.collect_fp8_target_module_names([task], state)


@pytest.mark.parametrize("scale", [torch.ones((2, 1)), torch.ones((1, 2, 2))])
def test_fp8_source_family_rejects_wrong_scale_grid(scale: torch.Tensor) -> None:
    source_key = "model.layers.0.self_attn.o_proj.weight"
    task = _task("decoder.layers.0.self_attention.linear_proj.weight", source_key)
    state = {source_key: _fp8_weight(), f"{source_key}_scale_inv": scale}

    with pytest.raises(ValueError, match=r"expected scalar or FP8 scale grid \(2, 2\)"):
        fp8_low_precision.collect_fp8_target_module_names([task], state)


def test_fp8_source_family_requires_exactly_one_scale_sibling() -> None:
    source_key = "model.layers.0.self_attn.o_proj.weight"
    task = _task("decoder.layers.0.self_attention.linear_proj.weight", source_key)

    with pytest.raises(ValueError, match=r"exactly one.*weight_scale_inv.*weight_scale"):
        fp8_low_precision.collect_fp8_target_module_names([task], {source_key: _fp8_weight()})

    with pytest.raises(ValueError, match=r"exactly one.*weight_scale_inv.*weight_scale"):
        fp8_low_precision.collect_fp8_target_module_names(
            [task],
            {
                source_key: _fp8_weight(),
                f"{source_key}_scale_inv": torch.ones((2, 2)),
                f"{source_key}_scale": torch.ones((2, 2)),
            },
        )


def test_fp8_source_family_rejects_scale_on_bf16_weight() -> None:
    source_key = "model.layers.0.self_attn.o_proj.weight"
    task = _task("decoder.layers.0.self_attention.linear_proj.weight", source_key)
    state = {
        source_key: torch.ones((129, 129), dtype=torch.bfloat16),
        f"{source_key}_scale_inv": torch.ones((2, 2)),
    }

    with pytest.raises(ValueError, match=r"scale metadata.*non-FP8 weight"):
        fp8_low_precision.collect_fp8_target_module_names([task], state)


def test_fp8_source_family_rejects_unsupported_float8_weight_dtype() -> None:
    source_key = "model.layers.0.self_attn.o_proj.weight"
    task = _task("decoder.layers.0.self_attention.linear_proj.weight", source_key)
    state = {
        source_key: _fp8_weight(dtype=torch.float8_e5m2),
        f"{source_key}_scale_inv": torch.ones((2, 2)),
    }

    with pytest.raises(TypeError, match=r"only torch.float8_e4m3fn"):
        fp8_low_precision.collect_fp8_target_module_names([task], state)


@pytest.mark.parametrize(
    "target_key",
    ["embedding.word_embeddings.weight", "output_layer.weight"],
)
def test_fp8_source_family_rejects_non_linear_target(target_key: str) -> None:
    source_key = "model.embed_tokens.weight"
    task = _task(target_key, source_key)
    state = {
        source_key: _fp8_weight(),
        f"{source_key}_scale_inv": torch.ones((2, 2)),
    }

    with pytest.raises(ValueError, match=r"unsupported target parameter"):
        fp8_low_precision.collect_fp8_target_module_names([task], state)


def test_fp8_source_family_rejects_unimplemented_weight_permutation() -> None:
    source_key = "model.layers.0.proj.weight"
    target_key = "decoder.layers.0.self_attention.linear_proj.weight"
    mapping = fp8_low_precision.AutoMapping(target_key, source_key, permute_dims=(1, 0))
    task = SimpleNamespace(mapping=mapping, megatron_module=object(), param_name=target_key)
    state = {
        source_key: _fp8_weight((128, 256)),
        f"{source_key}_scale_inv": torch.ones((1, 2)),
    }

    with pytest.raises(RuntimeError, match=r"permuted FP8 mapping"):
        fp8_low_precision.collect_fp8_target_module_names([task], state)


def test_fp8_source_family_rejects_unknown_mapping_semantics() -> None:
    source_key = "model.layers.0.proj.weight"
    target_key = "decoder.layers.0.self_attention.linear_proj.weight"
    task = SimpleNamespace(
        mapping=SimpleNamespace(hf_param=source_key),
        megatron_module=object(),
        param_name=target_key,
    )
    state = {
        source_key: _fp8_weight(),
        f"{source_key}_scale_inv": torch.ones((2, 2)),
    }

    with pytest.raises(RuntimeError, match=r"unsupported mapping type"):
        fp8_low_precision.collect_fp8_target_module_names([task], state)


def test_fp8_source_family_rejects_mixed_fused_weight_storage() -> None:
    target_key = "decoder.layers.0.mlp.linear_fc1.weight"
    gate_key = "model.layers.0.mlp.gate_proj.weight"
    up_key = "model.layers.0.mlp.up_proj.weight"
    mapping = fp8_low_precision.GatedMLPMapping(target_key, gate_key, up_key)
    task = SimpleNamespace(mapping=mapping, megatron_module=object(), param_name=target_key)
    state = {
        gate_key: _fp8_weight((128, 128)),
        f"{gate_key}_scale_inv": torch.ones((1, 1)),
        up_key: torch.ones((128, 128), dtype=torch.bfloat16),
    }

    with pytest.raises(ValueError, match=r"mixes E4M3FN and non-FP8 source weights"):
        fp8_low_precision.collect_fp8_target_module_names([task], state)


def test_fp8_source_family_rejects_unaligned_gated_concat_boundary() -> None:
    target_key = "decoder.layers.0.mlp.linear_fc1.weight"
    gate_key = "model.layers.0.mlp.gate_proj.weight"
    up_key = "model.layers.0.mlp.up_proj.weight"
    task = _gated_task(target_key, gate_key, up_key)
    state = {
        gate_key: _fp8_weight((129, 128)),
        f"{gate_key}_scale_inv": torch.ones((2, 1)),
        up_key: _fp8_weight((129, 128)),
        f"{up_key}_scale_inv": torch.ones((2, 1)),
    }

    with pytest.raises(ValueError, match=r"gated-MLP.*boundary.*128-element"):
        fp8_low_precision.preflight_fp8_conversion_tasks([task], state)


def test_fp8_source_family_rejects_mismatched_gated_weight_shapes() -> None:
    target_key = "decoder.layers.0.mlp.linear_fc1.weight"
    gate_key = "model.layers.0.mlp.gate_proj.weight"
    up_key = "model.layers.0.mlp.up_proj.weight"
    task = _gated_task(target_key, gate_key, up_key)
    state = {
        gate_key: _fp8_weight((128, 128)),
        f"{gate_key}_scale_inv": torch.ones((1, 1)),
        up_key: _fp8_weight((256, 128)),
        f"{up_key}_scale_inv": torch.ones((2, 1)),
    }

    with pytest.raises(ValueError, match=r"gate and up.*matching shapes"):
        fp8_low_precision.preflight_fp8_conversion_tasks([task], state)


def test_fp8_source_family_rejects_qkv_head_boundary_inside_scale_block() -> None:
    target_key = "decoder.layers.0.self_attention.linear_qkv.weight"
    q_key = "model.layers.0.self_attn.q_proj.weight"
    k_key = "model.layers.0.self_attn.k_proj.weight"
    v_key = "model.layers.0.self_attn.v_proj.weight"
    mapping = fp8_low_precision.QKVMapping(target_key, q_key, k_key, v_key)
    module = SimpleNamespace(
        config=SimpleNamespace(
            num_attention_heads=2,
            num_query_groups=1,
            kv_channels=64,
            hidden_size=128,
            attention_output_gate=False,
        )
    )
    task = SimpleNamespace(mapping=mapping, megatron_module=module, param_name=target_key)
    state = {
        q_key: _fp8_weight((128, 128)),
        f"{q_key}_scale_inv": torch.ones((1, 1)),
        k_key: _fp8_weight((64, 128)),
        f"{k_key}_scale_inv": torch.ones((1, 1)),
        v_key: _fp8_weight((64, 128)),
        f"{v_key}_scale_inv": torch.ones((1, 1)),
    }

    with pytest.raises(ValueError, match=r"QKV.*head.*boundary.*128-element"):
        fp8_low_precision.preflight_fp8_conversion_tasks([task], state)


def test_fp8_source_family_accepts_block_aligned_qkv_geometry() -> None:
    target_key = "decoder.layers.0.self_attention.linear_qkv.weight"
    q_key = "model.layers.0.self_attn.q_proj.weight"
    k_key = "model.layers.0.self_attn.k_proj.weight"
    v_key = "model.layers.0.self_attn.v_proj.weight"
    mapping = fp8_low_precision.QKVMapping(target_key, q_key, k_key, v_key)
    module = SimpleNamespace(
        config=SimpleNamespace(
            num_attention_heads=2,
            num_query_groups=1,
            kv_channels=128,
            hidden_size=128,
            attention_output_gate=False,
        )
    )
    task = SimpleNamespace(mapping=mapping, megatron_module=module, param_name=target_key)
    state = {
        q_key: _fp8_weight((256, 128)),
        f"{q_key}_scale_inv": torch.tensor([[1.0], [2.0]]),
        k_key: _fp8_weight((128, 128)),
        f"{k_key}_scale_inv": torch.tensor([[3.0]]),
        v_key: _fp8_weight((128, 128)),
        f"{v_key}_scale_inv": torch.tensor([[4.0]]),
    }

    plan = fp8_low_precision.preflight_fp8_conversion_tasks([task], state)

    assert plan.fp8_task_ids == frozenset({id(task)})
    torch.testing.assert_close(
        fp8_low_precision.build_merged_scale_inv_for_task(
            task,
            state,
            source_shapes={key: plan.source_shape(key) for key in (q_key, k_key, v_key)},
        ),
        torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )


def test_fp8_target_collection_excludes_bf16_eligible_linear() -> None:
    fp8_source = "model.layers.0.self_attn.o_proj.weight"
    bf16_source = "model.layers.0.mlp.down_proj.weight"
    fp8_target = "decoder.layers.0.self_attention.linear_proj.weight"
    bf16_target = "decoder.layers.0.mlp.linear_fc2.weight"
    tasks = [_task(fp8_target, fp8_source), _task(bf16_target, bf16_source)]
    state = {
        fp8_source: _fp8_weight(),
        f"{fp8_source}_scale_inv": torch.full((2, 2), 2.0),
        bf16_source: torch.ones((129, 129), dtype=torch.bfloat16),
    }

    assert fp8_low_precision.collect_fp8_target_module_names(tasks, state) == {
        "decoder.layers.0.self_attention.linear_proj"
    }


def test_fp8_preflight_uses_safetensors_metadata_and_builder_reuses_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_key = "model.layers.0.self_attn.o_proj.weight"
    scale_key = f"{source_key}_scale_inv"
    target_key = "decoder.layers.0.self_attention.linear_proj.weight"
    task = _task(target_key, source_key)
    save_file(
        {
            source_key: _fp8_weight(),
            scale_key: torch.full((2, 2), 2.0),
        },
        tmp_path / "model.safetensors",
    )

    class CountingSource(SafeTensorsStateSource):
        def __init__(self, path: Path):
            super().__init__(path)
            self.loaded_keys: list[str] = []

        def load_tensors(self, keys: list[str]) -> dict[str, torch.Tensor]:
            self.loaded_keys.extend(keys)
            return super().load_tensors(keys)

    source = CountingSource(tmp_path)
    state = StateDict(source)
    plan = fp8_low_precision.preflight_fp8_conversion_tasks([task], state)

    assert plan.module_names == frozenset({"decoder.layers.0.self_attention.linear_proj"})
    assert source.loaded_keys == [scale_key]

    def unexpected_preflight(*args, **kwargs):
        pytest.fail("the builder must reuse its prevalidated FP8 plan")

    monkeypatch.setattr(fp8_low_precision, "preflight_fp8_conversion_tasks", unexpected_preflight)
    result = fp8_low_precision.build_fp8_direct_model_state_dict(
        _bridge([task]),
        SimpleNamespace(state=state),
        [object()],
        {},
        conversion_tasks=[task],
        fp8_plan=plan,
    )

    assert f"{target_key}_w" in result
    assert source.loaded_keys.count(source_key) == 1
    assert source.loaded_keys.count(scale_key) == 2


def test_fp8_builder_rejects_plan_for_different_tasks() -> None:
    source_key = "model.layers.0.self_attn.o_proj.weight"
    target_key = "decoder.layers.0.self_attention.linear_proj.weight"
    task = _task(target_key, source_key)
    state = {
        source_key: _fp8_weight(),
        f"{source_key}_scale_inv": torch.full((2, 2), 2.0),
    }
    plan = fp8_low_precision.preflight_fp8_conversion_tasks([task], state)
    replacement_task = _task(target_key, source_key)

    with pytest.raises(ValueError, match=r"preflight plan.*conversion tasks"):
        fp8_low_precision.build_fp8_direct_model_state_dict(
            _bridge([replacement_task]),
            SimpleNamespace(state=state),
            [object()],
            {},
            conversion_tasks=[replacement_task],
            fp8_plan=plan,
        )


@pytest.mark.parametrize(
    "incomplete_task",
    [
        None,
        SimpleNamespace(
            mapping=SimpleNamespace(hf_param="unused"),
            megatron_module=None,
            param_name="decoder.layers.1.mlp.linear_fc2.weight",
        ),
    ],
    ids=["missing-mapping", "nonlocal-placeholder"],
)
def test_fp8_builder_rejects_incomplete_conversion_tasks(incomplete_task) -> None:
    source_key = "model.layers.0.self_attn.o_proj.weight"
    target_key = "decoder.layers.0.self_attention.linear_proj.weight"
    valid_task = _task(target_key, source_key)
    state = {
        source_key: _fp8_weight((128, 128)),
        f"{source_key}_scale_inv": torch.ones((1, 1)),
    }

    with pytest.raises(RuntimeError, match=r"incomplete.*task.*index 1"):
        fp8_low_precision.build_fp8_direct_model_state_dict(
            _bridge([valid_task, incomplete_task]),
            SimpleNamespace(state=state),
            [object()],
            {},
            conversion_tasks=[valid_task, incomplete_task],
        )


@pytest.mark.parametrize(
    ("scale_suffix", "source_scale", "expected_scale_inv"),
    [
        ("_scale_inv", 2.0, 2.0),
        ("_scale", 0.5, 2.0),
    ],
)
def test_fp8_builder_expands_scalar_scale_and_preserves_bf16_linear(
    scale_suffix: str,
    source_scale: float,
    expected_scale_inv: float,
) -> None:
    fp8_source = "model.layers.0.self_attn.o_proj.weight"
    bf16_source = "model.layers.0.mlp.down_proj.weight"
    fp8_target = "decoder.layers.0.self_attention.linear_proj.weight"
    bf16_target = "decoder.layers.0.mlp.linear_fc2.weight"
    tasks = [_task(fp8_target, fp8_source), _task(bf16_target, bf16_source)]
    state = {
        fp8_source: _fp8_weight(),
        f"{fp8_source}{scale_suffix}": torch.tensor(source_scale),
        bf16_source: torch.ones((129, 129), dtype=torch.bfloat16),
    }

    result = fp8_low_precision.build_fp8_direct_model_state_dict(
        _bridge(tasks),
        SimpleNamespace(state=state),
        [object()],
        {},
        conversion_tasks=tasks,
    )

    assert f"{fp8_target}_w" in result
    assert fp8_target not in result
    assert bf16_target in result
    assert f"{bf16_target}_w" not in result
    scale_entry = result[f"{fp8_target}_scale_inv"]
    assert scale_entry.dtype == torch.float32
    assert scale_entry.local_shape == (2, 2)
    torch.testing.assert_close(scale_entry.data, torch.full((2, 2), expected_scale_inv))


@pytest.mark.parametrize(
    ("target_key", "physical_key", "expert_offset"),
    [
        (
            "decoder.layers.0.mlp.linear_fc1.weight",
            "decoder.layers.0.mlp.linear_fc1.weight",
            None,
        ),
        (
            "decoder.layers.0.mlp.shared_experts.linear_fc1.weight",
            "decoder.layers.0.mlp.shared_experts.linear_fc1.weight",
            None,
        ),
        (
            "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight",
            "decoder.layers.0.mlp.experts.linear_fc1.weight",
            (0, 0, 2),
        ),
        (
            "decoder.layers.0.mlp.experts.linear_fc1.weight0",
            "decoder.layers.0.mlp.experts.linear_fc1.weight",
            (0, 0, 2),
        ),
    ],
)
def test_fp8_builder_derives_all_swiglu_split_layouts_from_mcore_factory(
    target_key: str,
    physical_key: str,
    expert_offset: tuple[int, int, int] | None,
) -> None:
    gate_key = "model.layers.0.mlp.gate_proj.weight"
    up_key = "model.layers.0.mlp.up_proj.weight"
    task = _gated_task(target_key, gate_key, up_key)
    state = {
        gate_key: torch.full((128, 128), 1.0).to(torch.float8_e4m3fn),
        f"{gate_key}_scale_inv": torch.full((1, 1), 2.0),
        up_key: torch.full((128, 128), 3.0).to(torch.float8_e4m3fn),
        f"{up_key}_scale_inv": torch.full((1, 1), 4.0),
    }
    factory = _swiglu_factory(
        target_key,
        physical_key,
        expert_offset=expert_offset,
    )

    built_template = factory.build()
    assert [entry.key for entry in built_template] == [physical_key, physical_key]
    result = fp8_low_precision.build_fp8_direct_model_state_dict(
        _bridge([task]),
        SimpleNamespace(state=state),
        [object()],
        {target_key: factory},
        conversion_tasks=[task],
    )

    weight_w_key = f"{target_key}_w"
    weight_v_key = f"{target_key}_v"
    assert result[weight_w_key].key == weight_w_key
    assert result[weight_v_key].key == weight_v_key
    for key in (weight_w_key, weight_v_key):
        assert result[key].local_shape == (128, 128)
        assert result[key].global_shape == (128, 128)
        assert result[key].global_offset == (0, 0)
        assert result[key].axis_fragmentations == (1, 1)
        assert result[key].prepend_axis_num == 0
    torch.testing.assert_close(result[weight_w_key].data, state[gate_key])
    torch.testing.assert_close(result[weight_v_key].data, state[up_key])

    scale_key = f"{target_key}_scale_inv"
    assert result[scale_key].local_shape == (2, 1)
    loaded_state = {
        weight_w_key: result[weight_w_key].data,
        weight_v_key: result[weight_v_key].data,
        scale_key: result[scale_key].data,
    }
    module_path = target_key.rsplit(".", 1)[0]
    model = _nested_module(module_path)
    leaf = model
    for segment in module_path.split("."):
        leaf = getattr(leaf, segment)
    weight_name = target_key.rsplit(".", 1)[1]
    leaf.register_parameter(weight_name, torch.nn.Parameter(torch.empty((256, 128))))

    assert register_fp8_scale_inv_buffers_after_load(model, loaded_state) == 1
    torch.testing.assert_close(loaded_state[target_key], torch.cat([state[gate_key], state[up_key]], dim=0))


@pytest.mark.parametrize("expert_index", [0, 1])
def test_fp8_builder_strips_grouped_fc2_expert_axis(expert_index: int) -> None:
    target_key = f"decoder.layers.0.mlp.experts.linear_fc2.weight{expert_index}"
    physical_key = "decoder.layers.0.mlp.experts.linear_fc2.weight"
    source_key = f"model.layers.0.mlp.experts.{expert_index}.down_proj.weight"
    task = _task(target_key, source_key)
    state = {
        source_key: _fp8_weight((128, 128)),
        f"{source_key}_scale_inv": torch.ones((1, 1)),
    }
    template = ShardedTensor.from_rank_offsets(
        physical_key,
        torch.empty((128, 128), dtype=torch.bfloat16),
        (0, expert_index, 2),
        prepend_axis_num=1,
    )

    result = fp8_low_precision.build_fp8_direct_model_state_dict(
        _bridge([task]),
        SimpleNamespace(state=state),
        [object()],
        {target_key: template},
        conversion_tasks=[task],
    )

    weight_entry = result[target_key]
    assert weight_entry.key == target_key
    assert weight_entry.local_shape == (128, 128)
    assert weight_entry.global_shape == (128, 128)
    assert weight_entry.global_offset == (0, 0)
    assert weight_entry.axis_fragmentations == (1, 1)
    assert weight_entry.prepend_axis_num == 0


def test_fp8_builder_strips_dense_layer_axis_for_direct_layout() -> None:
    target_key = "decoder.layers.0.self_attention.linear_proj.weight"
    physical_key = "decoder.layers.self_attention.linear_proj.weight"
    source_key = "model.layers.0.self_attn.o_proj.weight"
    task = _task(target_key, source_key)
    state = {
        source_key: _fp8_weight((128, 128)),
        f"{source_key}_scale_inv": torch.ones((1, 1)),
    }
    template = ShardedTensor.from_rank_offsets(
        physical_key,
        torch.empty((128, 128), dtype=torch.bfloat16),
        (0, 0, 2),
        prepend_axis_num=1,
    )
    scale_key = f"{target_key}_scale_inv"
    scale_template = ShardedTensor.from_rank_offsets(
        "decoder.layers.self_attention.linear_proj.weight_scale_inv",
        torch.empty((1, 1), dtype=torch.float32),
        (0, 0, 2),
        prepend_axis_num=1,
    )

    result = fp8_low_precision.build_fp8_direct_model_state_dict(
        _bridge([task]),
        SimpleNamespace(state=state),
        [object()],
        {target_key: template, scale_key: scale_template},
        conversion_tasks=[task],
    )

    weight_entry = result[f"{target_key}_w"]
    assert weight_entry.key == f"{target_key}_w"
    assert weight_entry.global_shape == (128, 128)
    assert weight_entry.global_offset == (0, 0)
    assert weight_entry.axis_fragmentations == (1, 1)
    assert weight_entry.prepend_axis_num == 0
    assert result[scale_key].key == scale_key
    assert result[scale_key].global_shape == (1, 1)
    assert result[scale_key].global_offset == (0, 0)
    assert result[scale_key].prepend_axis_num == 0


def test_fp8_builder_preserves_bf16_template_physical_key_and_layer_axis() -> None:
    target_key = "decoder.layers.0.mlp.linear_fc2.weight"
    physical_key = "decoder.layers.mlp.linear_fc2.weight"
    source_key = "model.layers.0.mlp.down_proj.weight"
    task = _task(target_key, source_key)
    state = {source_key: torch.ones((128, 128), dtype=torch.bfloat16)}
    template = ShardedTensor.from_rank_offsets(
        physical_key,
        torch.empty((128, 128), dtype=torch.bfloat16),
        (0, 0, 2),
        prepend_axis_num=1,
    )

    result = fp8_low_precision.build_fp8_direct_model_state_dict(
        _bridge([task]),
        SimpleNamespace(
            state={
                **state,
                "model.layers.1.self_attn.o_proj.weight": _fp8_weight((128, 128)),
                "model.layers.1.self_attn.o_proj.weight_scale_inv": torch.ones((1, 1)),
            }
        ),
        [object()],
        {target_key: template},
        conversion_tasks=[
            task,
            _task(
                "decoder.layers.1.self_attention.linear_proj.weight",
                "model.layers.1.self_attn.o_proj.weight",
            ),
        ],
    )

    weight_entry = result[target_key]
    assert weight_entry.key == physical_key
    assert weight_entry.global_shape == (2, 128, 128)
    assert weight_entry.global_offset == (0, 0, 0)
    assert weight_entry.prepend_axis_num == 1


def test_fp8_builder_preserves_bf16_swiglu_factory() -> None:
    target_key = "decoder.layers.0.mlp.linear_fc1.weight"
    physical_key = "decoder.layers.mlp.linear_fc1.weight"
    gate_key = "model.layers.0.mlp.gate_proj.weight"
    up_key = "model.layers.0.mlp.up_proj.weight"
    task = _gated_task(target_key, gate_key, up_key)
    fp8_task = _task(
        "decoder.layers.1.self_attention.linear_proj.weight",
        "model.layers.1.self_attn.o_proj.weight",
    )
    state = {
        gate_key: torch.ones((128, 128), dtype=torch.bfloat16),
        up_key: torch.ones((128, 128), dtype=torch.bfloat16),
        fp8_task.mapping.hf_param: _fp8_weight((128, 128)),
        f"{fp8_task.mapping.hf_param}_scale_inv": torch.ones((1, 1)),
    }
    sharded_offsets = ((0, 0, 2),)
    template = ShardedTensor.from_rank_offsets(
        physical_key,
        torch.empty((256, 128), dtype=torch.bfloat16),
        *sharded_offsets,
        prepend_axis_num=1,
    )
    factory = apply_swiglu_sharded_factory(
        template,
        sharded_offsets,
        singleton_local_shards=False,
    )

    result = fp8_low_precision.build_fp8_direct_model_state_dict(
        _bridge([task, fp8_task]),
        SimpleNamespace(state=state),
        [object()],
        {target_key: factory},
        conversion_tasks=[task, fp8_task],
    )

    saved_factory = result[target_key]
    assert saved_factory.key == physical_key
    assert [entry.key for entry in saved_factory.build()] == [physical_key, physical_key]


def test_fp8_builder_uses_complete_explicit_swiglu_templates() -> None:
    target_key = "decoder.layers.0.mlp.shared_experts.linear_fc1.weight"
    gate_key = "model.layers.0.mlp.gate_proj.weight"
    up_key = "model.layers.0.mlp.up_proj.weight"
    task = _gated_task(target_key, gate_key, up_key)
    state = {
        gate_key: _fp8_weight((128, 128)),
        f"{gate_key}_scale_inv": torch.ones((1, 1)),
        up_key: _fp8_weight((128, 128)),
        f"{up_key}_scale_inv": torch.ones((1, 1)),
    }
    weight_w_key = f"{target_key}_w"
    weight_v_key = f"{target_key}_v"
    model_template = {
        weight_w_key: ShardedTensor.from_rank_offsets(
            weight_w_key,
            torch.empty((128, 128), dtype=torch.bfloat16),
        ),
        weight_v_key: ShardedTensor.from_rank_offsets(
            weight_v_key,
            torch.empty((128, 128), dtype=torch.bfloat16),
        ),
    }

    result = fp8_low_precision.build_fp8_direct_model_state_dict(
        _bridge([task]),
        SimpleNamespace(state=state),
        [object()],
        model_template,
        conversion_tasks=[task],
    )

    assert result[weight_w_key].key == weight_w_key
    assert result[weight_v_key].key == weight_v_key


@pytest.mark.parametrize("present_suffix", ["_w", "_v"])
def test_fp8_builder_rejects_partial_explicit_swiglu_template(present_suffix: str) -> None:
    target_key = "decoder.layers.0.mlp.shared_experts.linear_fc1.weight"
    gate_key = "model.layers.0.mlp.gate_proj.weight"
    up_key = "model.layers.0.mlp.up_proj.weight"
    task = _gated_task(target_key, gate_key, up_key)
    state = {
        gate_key: _fp8_weight((128, 128)),
        f"{gate_key}_scale_inv": torch.ones((1, 1)),
        up_key: _fp8_weight((128, 128)),
        f"{up_key}_scale_inv": torch.ones((1, 1)),
    }
    present_key = f"{target_key}{present_suffix}"
    model_template = {
        present_key: ShardedTensor.from_rank_offsets(
            present_key,
            torch.empty((128, 128), dtype=torch.bfloat16),
        )
    }

    with pytest.raises(ValueError, match=r"incomplete.*SwiGLU.*_w.*_v"):
        fp8_low_precision.build_fp8_direct_model_state_dict(
            _bridge([task]),
            SimpleNamespace(state=state),
            [object()],
            model_template,
            conversion_tasks=[task],
        )


def test_fp8_builder_rejects_linear_fc1_without_split_layout_evidence() -> None:
    target_key = "decoder.layers.0.mlp.shared_experts.linear_fc1.weight"
    gate_key = "model.layers.0.mlp.gate_proj.weight"
    up_key = "model.layers.0.mlp.up_proj.weight"
    task = _gated_task(target_key, gate_key, up_key)
    state = {
        gate_key: _fp8_weight((128, 128)),
        f"{gate_key}_scale_inv": torch.ones((1, 1)),
        up_key: _fp8_weight((128, 128)),
        f"{up_key}_scale_inv": torch.ones((1, 1)),
    }

    with pytest.raises(ValueError, match=r"cannot determine.*SwiGLU.*template"):
        fp8_low_precision.build_fp8_direct_model_state_dict(
            _bridge([task]),
            SimpleNamespace(state=state),
            [object()],
            {},
            conversion_tasks=[task],
        )


def test_fp8_builder_allows_nongated_linear_fc1_without_split_template() -> None:
    target_key = "decoder.layers.0.mlp.linear_fc1.weight"
    source_key = "model.layers.0.mlp.fc1.weight"
    task = _task(target_key, source_key)
    state = {
        source_key: _fp8_weight((128, 128)),
        f"{source_key}_scale_inv": torch.ones((1, 1)),
    }

    result = fp8_low_precision.build_fp8_direct_model_state_dict(
        _bridge([task]),
        SimpleNamespace(state=state),
        [object()],
        {},
        conversion_tasks=[task],
    )

    assert set(result) == {f"{target_key}_w", f"{target_key}_scale_inv"}


def test_fp8_builder_rejects_swiglu_factory_with_wrong_fused_shape() -> None:
    target_key = "decoder.layers.0.mlp.linear_fc1.weight"
    gate_key = "model.layers.0.mlp.gate_proj.weight"
    up_key = "model.layers.0.mlp.up_proj.weight"
    task = _gated_task(target_key, gate_key, up_key)
    state = {
        gate_key: _fp8_weight((128, 128)),
        f"{gate_key}_scale_inv": torch.ones((1, 1)),
        up_key: _fp8_weight((128, 128)),
        f"{up_key}_scale_inv": torch.ones((1, 1)),
    }
    wrong_factory = apply_swiglu_sharded_factory(
        ShardedTensor.from_rank_offsets(
            target_key,
            torch.empty((512, 128), dtype=torch.bfloat16),
        ),
        (),
        singleton_local_shards=False,
    )

    with pytest.raises(ValueError, match=r"template.*shape.*converted weight"):
        fp8_low_precision.build_fp8_direct_model_state_dict(
            _bridge([task]),
            SimpleNamespace(state=state),
            [object()],
            {target_key: wrong_factory},
            conversion_tasks=[task],
        )


def test_fp8_builder_rejects_scale_template_shape_mismatch() -> None:
    target_key = "decoder.layers.0.self_attention.linear_proj.weight"
    source_key = "model.layers.0.self_attn.o_proj.weight"
    task = _task(target_key, source_key)
    state = {
        source_key: _fp8_weight((256, 128)),
        f"{source_key}_scale_inv": torch.ones((2, 1)),
    }
    scale_key = f"{target_key}_scale_inv"
    model_template = {
        scale_key: ShardedTensor.from_rank_offsets(
            scale_key,
            torch.empty((1, 1), dtype=torch.float32),
        )
    }

    with pytest.raises(ValueError, match=r"scale_inv.*shape.*expects"):
        fp8_low_precision.build_fp8_direct_model_state_dict(
            _bridge([task]),
            SimpleNamespace(state=state),
            [object()],
            model_template,
            conversion_tasks=[task],
        )


def test_selective_fp8_modelopt_config_preserves_ordered_list_exclusions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enabled_cfg = {
        "num_bits": (4, 3),
        "block_sizes": {-1: 128, -2: 128},
    }
    preset = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": enabled_cfg},
            {"quantizer_name": "*input_quantizer", "enable": False},
            {"quantizer_name": "*lm_head*", "enable": False},
            {"quantizer_name": "*router*", "enable": False},
        ],
        "algorithm": "max",
    }
    quant_cfg = _capture_modelopt_quant_config(
        monkeypatch,
        preset,
        {
            "decoder.layers.1.mlp.linear_fc2",
            "decoder.layers.0.self_attention.linear_proj",
        },
    )

    assert quant_cfg == {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "enable": False},
            {
                "quantizer_name": "decoder.layers.0.self_attention.linear_proj.weight_quantizer",
                "cfg": enabled_cfg,
            },
            {
                "quantizer_name": "decoder.layers.1.mlp.linear_fc2.weight_quantizer",
                "cfg": enabled_cfg,
            },
            {"quantizer_name": "*input_quantizer", "enable": False},
            {"quantizer_name": "*lm_head*", "enable": False},
            {"quantizer_name": "*router*", "enable": False},
        ],
        "algorithm": "max",
    }
    assert preset["quant_cfg"][1]["cfg"] is enabled_cfg


def test_selective_fp8_modelopt_config_preserves_legacy_dict_order_and_exclusions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enabled_cfg = {"num_bits": (4, 3), "axis": None}
    preset = {
        "quant_cfg": {
            "*": {"enable": False},
            "*weight_quantizer": enabled_cfg,
            "*input_quantizer": {"enable": False},
            "*lm_head*": {"enable": False},
        },
        "algorithm": "max",
    }
    quant_cfg = _capture_modelopt_quant_config(
        monkeypatch,
        preset,
        {"decoder.layers.0.self_attention.linear_proj"},
    )

    assert quant_cfg == {
        "quant_cfg": {
            "*": {"enable": False},
            "*weight_quantizer": {"enable": False},
            "decoder.layers.0.self_attention.linear_proj.weight_quantizer": enabled_cfg,
            "*input_quantizer": {"enable": False},
            "*lm_head*": {"enable": False},
        },
        "algorithm": "max",
    }
    assert list(quant_cfg["quant_cfg"]) == [
        "*",
        "*weight_quantizer",
        "decoder.layers.0.self_attention.linear_proj.weight_quantizer",
        "*input_quantizer",
        "*lm_head*",
    ]
