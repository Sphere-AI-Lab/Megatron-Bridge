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

import json
import warnings
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from megatron.bridge.orbit.conversion import oft_export


@dataclass
class _OFTConfig:
    r: int = 0
    block_size: int = 16
    coft: bool = False
    eps: float = 6e-5
    block_share: bool = False
    module_dropout: float = 0.1
    target_modules: tuple[str, ...] = ("linear_qkv",)
    layers_to_transform: tuple[int, ...] = (1, 3)


@pytest.mark.unit
def test_export_format_string_matches_serialized_value() -> None:
    """The Python 3.10-compatible enum must retain ``StrEnum`` semantics."""
    assert str(oft_export.OFTExportFormat.SGLANG) == "sglang"
    assert oft_export.OFTExportFormat.HF_PEFT == "hf_peft"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("projection", "hf_leaf"),
    [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")],
)
def test_dsv4_grouped_names_are_consumer_specific(projection: str, hf_leaf: str) -> None:
    assert (
        oft_export._dsv4_grouped_oft_name(
            layer="3",
            expert_idx=7,
            projection=projection,
            export_format=oft_export.OFTExportFormat.SGLANG,
        )
        == f"layers.3.ffn.experts.7.{projection}.oft_R.weight"
    )
    assert (
        oft_export._dsv4_grouped_oft_name(
            layer="3",
            expert_idx=7,
            projection=projection,
            export_format=oft_export.OFTExportFormat.HF_PEFT,
        )
        == f"model.layers.3.mlp.experts.7.{hf_leaf}.oft_R.weight"
    )
    assert (
        oft_export._dsv4_grouped_oft_name(
            layer="3",
            expert_idx=7,
            projection=projection,
            export_format="hf_peft",  # type: ignore[arg-type]
        )
        == f"model.layers.3.mlp.experts.7.{hf_leaf}.oft_R.weight"
    )


@pytest.mark.unit
@pytest.mark.parametrize(("n_elements", "block_size"), [(1, 2), (6, 4), (120, 16)])
def test_infer_oft_block_size(n_elements: int, block_size: int) -> None:
    assert oft_export._infer_oft_block_size_from_n_elements(n_elements) == block_size


@pytest.mark.unit
def test_infer_oft_block_size_rejects_non_triangular_length() -> None:
    with pytest.raises(ValueError, match="Cannot infer OFT block size"):
        oft_export._infer_oft_block_size_from_n_elements(5)


@pytest.mark.unit
def test_globalize_dsv4_expert_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oft_export.parallel_state, "get_expert_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(oft_export.parallel_state, "get_expert_model_parallel_rank", lambda: 2)

    result = oft_export._globalize_dsv4_native_expert_oft_base_prefix(
        "decoder.layers.3.mlp.experts.1.w2",
        num_moe_experts=16,
    )

    assert result == "decoder.layers.3.mlp.experts.9.w2"


@pytest.mark.unit
def test_globalize_dsv4_expert_prefix_rejects_nondivisible_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        oft_export.parallel_state,
        "get_expert_model_parallel_world_size",
        lambda: 3,
    )

    with pytest.raises(ValueError, match="must be divisible"):
        oft_export._globalize_dsv4_native_expert_oft_base_prefix(
            "decoder.layers.3.mlp.experts.1.w2",
            num_moe_experts=8,
            ep_rank=0,
        )


@pytest.mark.unit
def test_build_oft_adapter_config_uses_only_hf_schema_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oft_export, "_installed_peft_version", lambda: "0.19.1")
    result = oft_export.build_oft_adapter_config_dict(
        _OFTConfig(),
        target_modules=["model.layers.1.mlp.gate_proj", "model.layers.3.self_attn.q_proj"],
        base_model_name_or_path="radixark/base-model",
    )

    assert set(result) == {
        "auto_mapping",
        "base_model_name_or_path",
        "bias",
        "block_share",
        "coft",
        "eps",
        "exclude_modules",
        "fan_in_fan_out",
        "inference_mode",
        "init_weights",
        "layers_pattern",
        "layers_to_transform",
        "module_dropout",
        "modules_to_save",
        "num_cayley_neumann_terms",
        "oft_block_size",
        "peft_type",
        "peft_version",
        "r",
        "revision",
        "target_modules",
        "task_type",
        "use_cayley_neumann",
    }
    assert result == {
        "auto_mapping": None,
        "base_model_name_or_path": "radixark/base-model",
        "bias": "none",
        "block_share": False,
        "coft": False,
        "eps": 6e-5,
        "exclude_modules": None,
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "module_dropout": 0.1,
        "modules_to_save": None,
        "num_cayley_neumann_terms": 5,
        "oft_block_size": 16,
        "peft_type": "OFT",
        "peft_version": "0.19.1",
        "r": 0,
        "revision": None,
        "target_modules": ["model.layers.1.mlp.gate_proj", "model.layers.3.self_attn.q_proj"],
        "task_type": "CAUSAL_LM",
        "use_cayley_neumann": True,
    }


@pytest.mark.unit
@pytest.mark.parametrize("config_name", ["OFT", "CanonicalOFT", "VLMOFT"])
def test_real_bridge_oft_configs_do_not_leak_runtime_fields(config_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from megatron.bridge.orbit.oft.canonical_oft import CanonicalOFT
    from megatron.bridge.orbit.oft.oft import OFT, VLMOFT

    config = {"OFT": OFT, "CanonicalOFT": CanonicalOFT, "VLMOFT": VLMOFT}[config_name]()
    # Exercise mutable matcher state that dataclasses.asdict used to leak.
    config._pattern_to_alias["runtime-only"] = "cache"
    monkeypatch.setattr(oft_export, "_installed_peft_version", lambda: "0.19.1")

    serialized = oft_export.build_oft_adapter_config_dict(
        config,
        target_modules=["model.layers.0.self_attn.q_proj"],
    )

    assert "params_to_save" not in serialized
    assert "canonical_mapping" not in serialized
    assert "_pattern_to_alias" not in serialized
    assert "freeze_vision_model" not in serialized


@pytest.mark.unit
def test_infer_oft_target_modules_ignores_non_adapter_weights() -> None:
    names = [
        "base_model.model.model.layers.0.self_attn.q_proj.oft_R.weight",
        "base_model.model.model.layers.0.mlp.gate_proj.oft_R.weight",
        "base_model.model.model.layers.1.self_attn.q_proj.oft_R.weight",
        "base_model.model.model.embed_tokens.oft_embedding_R.weight",
        "model.embed_tokens.weight",
    ]

    assert oft_export.infer_oft_target_modules(names) == [
        "model.embed_tokens",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.self_attn.q_proj",
        "model.layers.1.self_attn.q_proj",
    ]


@pytest.mark.unit
def test_make_oft_param_name_distinguishes_embedding_from_linear() -> None:
    mixin = oft_export.OrbitOFTExportMixin()

    assert mixin._make_oft_param_name("model.embed_tokens.weight", is_embedding=True) == (
        "model.embed_tokens.oft_embedding_R.weight"
    )
    assert mixin._make_oft_param_name("model.layers.0.self_attn.q_proj.weight") == (
        "model.layers.0.self_attn.q_proj.oft_R.weight"
    )
    for export_format in oft_export.OFTExportFormat:
        with pytest.raises(ValueError, match="requires a module weight mapping"):
            mixin._make_oft_param_name(
                "model.layers.0.mlp.experts.gate_up_proj",
                export_format=export_format,
            )


@pytest.mark.unit
def test_task_cpu_materialization_releases_device_aliases() -> None:
    shared = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    outputs = oft_export._materialize_oft_task_outputs(
        [("first", shared), ("second", shared)],
        cpu=True,
    )

    assert [name for name, _ in outputs] == ["first", "second"]
    assert all(tensor.device.type == "cpu" and tensor.is_contiguous() for _, tensor in outputs)
    assert outputs[0][1].untyped_storage().data_ptr() != outputs[1][1].untyped_storage().data_ptr()


@pytest.mark.unit
def test_embedding_task_plan_uses_peft_embedding_key() -> None:
    mixin = oft_export.OrbitOFTExportMixin()
    mixin._get_base_hf_param_names_for_adapter = (  # type: ignore[attr-defined]
        lambda registry, prefix, _, suffix: ["model.embed_tokens.weight"]
    )
    task = oft_export.OFTAdapterConversionTask(
        global_base_prefix="embedding.word_embeddings",
        local_base_prefix="embedding.word_embeddings",
        is_expert=False,
        input_is_parallel=False,
        block_size=16,
        r=2,
        block_share=False,
        pp_rank=0,
        vp_stage=0,
        is_embedding=True,
    )

    assert mixin._planned_hf_names_for_oft_task(
        task,
        mapping_registry=object(),
        num_moe_experts=0,
        export_format=oft_export.OFTExportFormat.HF_PEFT,
    ) == ["model.embed_tokens.oft_embedding_R.weight"]


@pytest.mark.unit
@pytest.mark.parametrize(("shape", "grouped_count"), [((2, 6), None), ((3, 2, 6), 3)])
def test_discovery_records_explicit_expert_axis_only_for_3d_rotation(
    shape: tuple[int, ...],
    grouped_count: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mixin = oft_export.OrbitOFTExportMixin()
    adapter = SimpleNamespace(
        is_expert=True,
        input_is_parallel=False,
        block_size=4,
        r=2,
        block_share=False,
    )
    mixin._unwrap_name = lambda name: name  # type: ignore[attr-defined]
    mixin._get_oft_adapter_wrap_module = lambda *args, **kwargs: adapter  # type: ignore[method-assign]
    monkeypatch.setattr(
        oft_export,
        "get_module_and_param_from_name",
        lambda *args, **kwargs: (object(), object()),
    )
    global_name = "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.adapter_gate.oft_r"

    info = mixin._local_oft_adapter_info_for_parameter(
        megatron_model=[object()],
        model_config=object(),
        local_param_name=global_name,
        param=torch.zeros(shape),
        vp_stage=0,
        pp_rank=0,
        local_to_global=lambda *args: global_name,
    )

    assert info is not None
    assert info[12] == grouped_count
    assert info[13:] == ("torch.float32", "cpu")


@pytest.mark.unit
def test_canonical_sequential_expert_plan_emits_only_its_local_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mixin = oft_export.OrbitOFTExportMixin()
    calls: list[str] = []

    def mapped_names(registry, prefix, _, suffix):
        calls.append(suffix)
        return ["model.layers.0.mlp.experts.0.gate_proj.weight"]

    mixin._get_base_hf_param_names_for_adapter = mapped_names  # type: ignore[attr-defined]
    monkeypatch.setattr(
        oft_export.parallel_state,
        "get_expert_model_parallel_world_size",
        lambda: 1,
    )
    task = oft_export.OFTAdapterConversionTask(
        global_base_prefix="decoder.layers.0.mlp.experts.local_experts.0.linear_fc1",
        local_base_prefix="decoder.layers.0.mlp.experts.local_experts.0.linear_fc1",
        is_expert=True,
        input_is_parallel=False,
        block_size=16,
        r=2,
        block_share=False,
        pp_rank=0,
        vp_stage=0,
        slice_name="gate",
    )

    assert mixin._planned_hf_names_for_oft_task(
        task,
        mapping_registry=object(),
        num_moe_experts=8,
        export_format=oft_export.OFTExportFormat.HF_PEFT,
    ) == ["model.layers.0.mlp.experts.0.gate_proj.oft_R.weight"]
    assert calls == [".weight"]


@pytest.mark.unit
def test_canonical_grouped_expert_plan_fans_out_explicit_expert_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mixin = oft_export.OrbitOFTExportMixin()
    calls: list[str] = []

    def mapped_names(registry, prefix, _, suffix):
        calls.append(suffix)
        expert_idx = int(suffix.removeprefix(".weight"))
        return [
            f"model.layers.0.mlp.experts.{expert_idx}.gate_proj.weight",
            f"model.layers.0.mlp.experts.{expert_idx}.up_proj.weight",
        ]

    mixin._get_base_hf_param_names_for_adapter = mapped_names  # type: ignore[attr-defined]
    monkeypatch.setattr(
        oft_export.parallel_state,
        "get_expert_model_parallel_world_size",
        lambda: 2,
    )
    task = oft_export.OFTAdapterConversionTask(
        global_base_prefix="decoder.layers.0.mlp.linear_fc1",
        local_base_prefix="decoder.layers.0.mlp.linear_fc1",
        is_expert=True,
        input_is_parallel=False,
        block_size=16,
        r=2,
        block_share=False,
        pp_rank=0,
        vp_stage=0,
        slice_name="gate",
        grouped_expert_count=2,
    )

    assert mixin._planned_hf_names_for_oft_task(
        task,
        mapping_registry=object(),
        num_moe_experts=4,
        export_format=oft_export.OFTExportFormat.HF_PEFT,
    ) == [
        "model.layers.0.mlp.experts.0.gate_proj.oft_R.weight",
        "model.layers.0.mlp.experts.1.gate_proj.oft_R.weight",
        "model.layers.0.mlp.experts.2.gate_proj.oft_R.weight",
        "model.layers.0.mlp.experts.3.gate_proj.oft_R.weight",
    ]
    assert calls == [".weight0", ".weight1", ".weight2", ".weight3"]


@pytest.mark.unit
def test_grouped_expert_plan_rejects_wrong_local_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    mixin = oft_export.OrbitOFTExportMixin()
    mixin._get_base_hf_param_names_for_adapter = (  # type: ignore[attr-defined]
        lambda *args: pytest.fail("invalid expert geometry must fail before mapping")
    )
    monkeypatch.setattr(
        oft_export.parallel_state,
        "get_expert_model_parallel_world_size",
        lambda: 2,
    )
    task = oft_export.OFTAdapterConversionTask(
        global_base_prefix="decoder.layers.0.mlp.linear_fc1",
        local_base_prefix="decoder.layers.0.mlp.linear_fc1",
        is_expert=True,
        input_is_parallel=False,
        block_size=16,
        r=2,
        block_share=False,
        pp_rank=0,
        vp_stage=0,
        slice_name="gate",
        grouped_expert_count=1,
    )

    with pytest.raises(ValueError, match="1 local experts, expected 2"):
        mixin._planned_hf_names_for_oft_task(
            task,
            mapping_registry=object(),
            num_moe_experts=4,
            export_format=oft_export.OFTExportFormat.HF_PEFT,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("prefix", "expert_idx"),
    [
        ("decoder.layers.0.mlp.experts.local_experts.0.linear_fc1", None),
        ("decoder.layers.0.mlp.linear_fc1", 0),
    ],
)
def test_rank_local_expert_plan_fails_closed_under_ep(
    prefix: str,
    expert_idx: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mixin = oft_export.OrbitOFTExportMixin()
    mixin._get_base_hf_param_names_for_adapter = (  # type: ignore[attr-defined]
        lambda *args: pytest.fail("unsafe rank-local layout must fail before mapping")
    )
    monkeypatch.setattr(
        oft_export.parallel_state,
        "get_expert_model_parallel_world_size",
        lambda: 2,
    )
    task = oft_export.OFTAdapterConversionTask(
        global_base_prefix=prefix,
        local_base_prefix=prefix,
        is_expert=True,
        input_is_parallel=False,
        block_size=16,
        r=2,
        block_share=False,
        pp_rank=0,
        vp_stage=0,
        slice_name="gate",
        expert_idx=expert_idx,
    )

    with pytest.raises(ValueError, match="rank-local expert OFT layout"):
        mixin._planned_hf_names_for_oft_task(
            task,
            mapping_registry=object(),
            num_moe_experts=4,
            export_format=oft_export.OFTExportFormat.HF_PEFT,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("layout", "expected_names"),
    [
        (
            "dense",
            [
                "model.layers.0.self_attn.q_proj.oft_R.weight",
                "model.layers.0.self_attn.k_proj.oft_R.weight",
                "model.layers.0.self_attn.v_proj.oft_R.weight",
            ],
        ),
        ("canonical_dense", ["model.layers.0.self_attn.q_proj.oft_R.weight"]),
        ("embedding", ["model.embed_tokens.oft_embedding_R.weight"]),
        (
            "canonical_grouped",
            [
                "model.layers.0.mlp.experts.0.gate_proj.oft_R.weight",
                "model.layers.0.mlp.experts.1.gate_proj.oft_R.weight",
            ],
        ),
        (
            "legacy_grouped",
            [
                "model.layers.0.mlp.experts.0.gate_proj.oft_R.weight",
                "model.layers.0.mlp.experts.0.up_proj.oft_R.weight",
                "model.layers.0.mlp.experts.1.gate_proj.oft_R.weight",
                "model.layers.0.mlp.experts.1.up_proj.oft_R.weight",
            ],
        ),
        ("native_dsv4", ["model.layers.0.mlp.experts.0.gate_proj.oft_R.weight"]),
        (
            "grouped_dsv4",
            [
                "model.layers.0.mlp.experts.0.gate_proj.oft_R.weight",
                "model.layers.0.mlp.experts.1.gate_proj.oft_R.weight",
            ],
        ),
    ],
)
def test_stream_matches_preflight_plan_for_supported_layouts(
    layout: str,
    expected_names: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pp_size = 2 if layout == "dense" else 1

    class _PipelineGroup:
        @staticmethod
        def size() -> int:
            return pp_size

    num_experts = 2 if "grouped" in layout else 1
    tensor = torch.zeros((num_experts, 2, 6) if layout in {"canonical_grouped", "grouped_dsv4"} else (2, 6))
    prefix = {
        "dense": "decoder.layers.0.self_attention.linear_qkv",
        "canonical_dense": "decoder.layers.0.self_attention.linear_qkv",
        "embedding": "embedding.word_embeddings",
        "canonical_grouped": "decoder.layers.0.mlp.linear_fc1",
        "legacy_grouped": "decoder.layers.0.mlp.linear_fc1",
        "native_dsv4": "decoder.layers.0.mlp.experts.0.w1",
        "grouped_dsv4": "decoder.layers.0.mlp.w1",
    }[layout]
    task = oft_export.OFTAdapterConversionTask(
        global_base_prefix=prefix,
        local_base_prefix=prefix,
        is_expert=layout in {"canonical_grouped", "legacy_grouped", "native_dsv4", "grouped_dsv4"},
        input_is_parallel=False,
        block_size=4,
        r=2,
        block_share=False,
        pp_rank=1 if layout == "dense" else 0,
        vp_stage=0,
        slice_name="q" if layout == "canonical_dense" else "gate" if layout == "canonical_grouped" else None,
        is_embedding=layout == "embedding",
        grouped_expert_count=num_experts if layout in {"canonical_grouped", "grouped_dsv4"} else None,
        tensor_dtype="torch.float32",
        device_type="cpu",
    )
    adapter = SimpleNamespace(oft_r=tensor)
    mixin = oft_export.OrbitOFTExportMixin()
    mixin.mapping_registry = lambda: object()  # type: ignore[attr-defined]
    mixin._megatron_global_oft_adapters_info_all_pp_ranks = lambda model: [task]  # type: ignore[method-assign]
    mixin._get_oft_adapter_wrap_module = lambda *args, **kwargs: adapter  # type: ignore[method-assign]
    mixin._with_progress_tracking = lambda tasks, *args, **kwargs: tasks  # type: ignore[attr-defined]
    mixin._gather_expert_adapter_weight = lambda weight: None  # type: ignore[attr-defined]
    mixin._select_expert_adapter_weight = (  # type: ignore[attr-defined]
        lambda weight, gathered, expert_idx, total: weight[expert_idx] if weight.ndim > 2 else weight
    )

    def mapped_names(registry, global_prefix, _, suffix):
        if layout in {"dense", "canonical_dense"}:
            return [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
            ]
        if layout == "embedding":
            return ["model.embed_tokens.weight"]
        if layout in {"canonical_grouped", "legacy_grouped"}:
            expert_idx = int(suffix.removeprefix(".weight"))
            return [
                f"model.layers.0.mlp.experts.{expert_idx}.gate_proj.weight",
                f"model.layers.0.mlp.experts.{expert_idx}.up_proj.weight",
            ]
        if layout == "native_dsv4":
            return ["model.layers.0.mlp.experts.0.gate_proj.weight"]
        raise AssertionError(f"grouped DSV4 naming bypasses the mapping registry: {global_prefix}")

    mixin._get_base_hf_param_names_for_adapter = mapped_names  # type: ignore[attr-defined]
    monkeypatch.setattr(oft_export, "unwrap_model", lambda models: models)
    monkeypatch.setattr(oft_export, "get_module_and_param_from_name", lambda *args, **kwargs: (object(), tensor))
    monkeypatch.setattr(oft_export.parallel_state, "get_pipeline_model_parallel_group", _PipelineGroup)
    monkeypatch.setattr(oft_export.parallel_state, "get_pipeline_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(oft_export.parallel_state, "get_expert_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(oft_export.parallel_state, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(oft_export.parallel_state, "get_expert_tensor_parallel_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: list(range(pp_size)))
    monkeypatch.setattr(torch.distributed, "broadcast", lambda tensor, src, group: None)
    model = SimpleNamespace(config=SimpleNamespace(num_moe_experts=num_experts))

    exported = list(
        mixin.stream_oft_adapter_weights_megatron_to_hf(
            model,
            cpu=True,
            show_progress=False,
            export_format="hf_peft",  # type: ignore[arg-type]
        )
    )

    assert [item.param_name for item in exported] == expected_names
    assert all(item.weight.device.type == "cpu" for item in exported)


@pytest.mark.unit
def test_serialized_geometry_rejects_silently_adjusted_block_size() -> None:
    state = {"base_model.model.model.layers.0.mlp.down_proj.oft_R.weight": torch.zeros(2, 15)}

    with pytest.raises(ValueError, match="encodes block_size=6, not configured block_size=5"):
        oft_export._validate_serialized_oft_state_geometry(state, _OFTConfig(block_size=5))


@pytest.mark.unit
def test_serialized_geometry_rejects_tp_gathered_r_mismatch() -> None:
    state = {"base_model.model.model.layers.0.mlp.down_proj.oft_R.weight": torch.zeros(4, 6)}

    with pytest.raises(ValueError, match="contains 4 blocks, not the configured PEFT count 2"):
        oft_export._validate_serialized_oft_state_geometry(state, _OFTConfig(r=2, block_size=0))


@pytest.mark.unit
def test_serialized_geometry_accepts_block_shared_r_mode() -> None:
    state = {"base_model.model.model.layers.0.mlp.down_proj.oft_R.weight": torch.zeros(1, 6)}

    oft_export._validate_serialized_oft_state_geometry(
        state,
        _OFTConfig(r=2, block_size=0, block_share=True),
    )


@pytest.mark.unit
def test_serialized_geometry_accepts_nonshared_coft() -> None:
    state = {"base_model.model.model.layers.0.mlp.down_proj.oft_R.weight": torch.zeros(2, 120)}

    oft_export._validate_serialized_oft_state_geometry(state, _OFTConfig(coft=True))


@pytest.mark.unit
def test_installed_peft_version_rejects_pre_parameterization_release(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oft_export.importlib_metadata, "version", lambda package: "0.17.1")

    with pytest.raises(RuntimeError, match="requires peft>=0.18.0"):
        oft_export._installed_peft_version()


@pytest.mark.unit
def test_legacy_grouped_export_distinguishes_serving_from_hf_peft() -> None:
    names = [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
    ]

    serving_names = oft_export._filter_legacy_grouped_oft_weight_names(
        names,
        export_format=oft_export.OFTExportFormat.SGLANG,
    )
    peft_names = oft_export._filter_legacy_grouped_oft_weight_names(
        names,
        export_format=oft_export.OFTExportFormat.HF_PEFT,
    )

    assert serving_names == [names[0]]
    assert peft_names == names


@pytest.mark.unit
def test_legacy_grouped_hf_export_matches_fused_serving_output() -> None:
    """Duplicating the shared rotation must preserve fused gate/up behavior."""
    names = [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
    ]
    serving_names = oft_export._filter_legacy_grouped_oft_weight_names(
        names,
        export_format=oft_export.OFTExportFormat.SGLANG,
    )
    hf_names = oft_export._filter_legacy_grouped_oft_weight_names(
        names,
        export_format=oft_export.OFTExportFormat.HF_PEFT,
    )

    oft_r = torch.tensor([[0.25], [-0.4]], dtype=torch.float32)
    rotation = oft_export.OrbitOFTExportMixin._compute_oft_rotation_matrix(
        oft_r,
        block_size=2,
        in_features=4,
        block_share=False,
    )
    gate_weight = torch.tensor([[1.0, 2.0, -1.0, 0.5], [0.0, -2.0, 3.0, 1.0]])
    up_weight = torch.tensor([[4.0, -1.0, 0.5, 2.0], [-3.0, 1.0, 2.0, -0.5]])
    inputs = torch.tensor([[0.5, -1.0, 2.0, 3.0], [-2.0, 0.25, 1.0, -1.5]])

    serving_uses_shared_rotation = names[0] in serving_names and names[1] not in serving_names
    serving_rotation = rotation if serving_uses_shared_rotation else torch.eye(4)
    fused_weight = torch.cat((gate_weight, up_weight), dim=0) @ serving_rotation
    serving_output = torch.nn.functional.linear(inputs, fused_weight)

    exported_names = set(hf_names)
    hf_gate_rotation = rotation if names[0] in exported_names else torch.eye(4)
    hf_up_rotation = rotation if names[1] in exported_names else torch.eye(4)
    hf_output = torch.cat(
        (
            torch.nn.functional.linear(inputs, gate_weight @ hf_gate_rotation),
            torch.nn.functional.linear(inputs, up_weight @ hf_up_rotation),
        ),
        dim=-1,
    )

    torch.testing.assert_close(hf_output, serving_output)


@pytest.mark.unit
def test_merge_oft_adapter_weights_preserves_input_rotation_output() -> None:
    """The direct weight-merge helper must fold the row-vector rotation."""
    base_weight = torch.tensor(
        [[1.0, 2.0, -1.0, 0.5], [0.0, -2.0, 3.0, 1.0]],
        dtype=torch.float32,
    )
    oft_r = torch.tensor([[0.25], [-0.4]], dtype=torch.float32)
    inputs = torch.tensor([[0.5, -1.0, 2.0, 3.0], [-2.0, 0.25, 1.0, -1.5]])
    rotation = oft_export.OrbitOFTExportMixin._compute_oft_rotation_matrix(
        oft_r,
        block_size=2,
        in_features=4,
        block_share=False,
    )
    expected = torch.nn.functional.linear(inputs @ rotation, base_weight)
    converted = {"model.layers.0.self_attn.q_proj.weight": base_weight.clone()}
    mixin = oft_export.OrbitOFTExportMixin()

    merged = mixin._merge_oft_adapter_weights(
        [],
        converted,
        oft_r,
        block_size=2,
        block_share=False,
    )

    torch.testing.assert_close(torch.nn.functional.linear(inputs, next(iter(merged.values()))), expected)


@pytest.mark.unit
def test_save_hf_oft_adapter_writes_loadable_peft_directory(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = [
        ("model.layers.0.self_attn.q_proj.oft_R.weight", torch.arange(240, dtype=torch.float32).reshape(2, 120)),
        ("model.layers.0.mlp.gate_proj.oft_R.weight", torch.ones(2, 120)),
    ]
    export_calls = []

    def fake_export(*args, **kwargs):
        export_calls.append(kwargs)
        return iter(weights)

    monkeypatch.setattr(oft_export, "export_oft_adapter_weights", fake_export)
    monkeypatch.setattr(oft_export, "_installed_peft_version", lambda: "0.19.1")
    auto_bridge = SimpleNamespace(hf_pretrained=SimpleNamespace(model_name_or_path="radixark/base-model"))
    save_dir = tmp_path / "adapter"

    oft_export.save_hf_oft_adapter(
        auto_bridge,
        object(),
        save_dir,
        _OFTConfig(),
        show_progress=False,
    )

    config = json.loads((save_dir / "adapter_config.json").read_text())
    saved = load_file(save_dir / "adapter_model.safetensors")
    assert config["base_model_name_or_path"] == "radixark/base-model"
    assert config["target_modules"] == [
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.self_attn.q_proj",
    ]
    assert set(saved) == {
        "base_model.model.model.layers.0.mlp.gate_proj.oft_R.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.oft_R.weight",
    }
    assert torch.equal(saved["base_model.model.model.layers.0.self_attn.q_proj.oft_R.weight"], weights[0][1])
    assert export_calls == [
        {
            "cpu": True,
            "show_progress": False,
            "export_format": oft_export.OFTExportFormat.HF_PEFT,
        }
    ]


@pytest.mark.unit
def test_save_hf_oft_adapter_dealiases_shared_cpu_rotation(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    shared = torch.arange(240, dtype=torch.float32).reshape(2, 120)
    weights = [
        ("model.layers.0.self_attn.q_proj.oft_R.weight", shared),
        ("model.layers.0.self_attn.k_proj.oft_R.weight", shared),
    ]
    monkeypatch.setattr(oft_export, "export_oft_adapter_weights", lambda *args, **kwargs: iter(weights))
    monkeypatch.setattr(oft_export, "_installed_peft_version", lambda: "0.19.1")
    save_dir = tmp_path / "adapter"

    oft_export.save_hf_oft_adapter(
        SimpleNamespace(),
        object(),
        save_dir,
        _OFTConfig(),
        show_progress=False,
    )

    saved = load_file(save_dir / "adapter_model.safetensors")
    torch.testing.assert_close(saved["base_model.model.model.layers.0.self_attn.q_proj.oft_R.weight"], shared)
    torch.testing.assert_close(saved["base_model.model.model.layers.0.self_attn.k_proj.oft_R.weight"], shared)


@pytest.mark.unit
def test_saved_adapter_roundtrips_through_peft_from_pretrained(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The on-disk key must become ``oft_R.default.weight`` inside PEFT."""
    from peft import PeftModel
    from transformers import LlamaConfig, LlamaForCausalLM

    expected = torch.arange(240, dtype=torch.float32).reshape(2, 120)
    weights = [("model.layers.0.self_attn.q_proj.oft_R.weight", expected)]
    monkeypatch.setattr(oft_export, "export_oft_adapter_weights", lambda *args, **kwargs: iter(weights))
    monkeypatch.setattr(oft_export, "_installed_peft_version", lambda: "0.19.1")

    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=64,
    )
    base_model = LlamaForCausalLM(config)
    auto_bridge = SimpleNamespace(hf_pretrained=SimpleNamespace(model_name_or_path="local/tiny-llama"))
    save_dir = tmp_path / "adapter"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        oft_export.save_hf_oft_adapter(
            auto_bridge,
            object(),
            save_dir,
            _OFTConfig(target_modules=("linear_qkv",), layers_to_transform=(0,)),
            show_progress=False,
        )

        loaded = PeftModel.from_pretrained(base_model, save_dir)
    actual = loaded.base_model.model.model.layers[0].self_attn.q_proj.oft_R["default"].weight
    torch.testing.assert_close(actual, expected)


@pytest.mark.unit
def test_saved_embedding_adapter_roundtrips_under_peft_name(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from peft import PeftModel
    from transformers import LlamaConfig, LlamaForCausalLM

    expected = torch.arange(240, dtype=torch.float32).reshape(2, 120)
    weights = [("model.embed_tokens.oft_embedding_R.weight", expected)]
    monkeypatch.setattr(oft_export, "export_oft_adapter_weights", lambda *args, **kwargs: iter(weights))
    monkeypatch.setattr(oft_export, "_installed_peft_version", lambda: "0.19.1")

    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=64,
    )
    base_model = LlamaForCausalLM(config)
    auto_bridge = SimpleNamespace(hf_pretrained=SimpleNamespace(model_name_or_path="local/tiny-llama"))
    save_dir = tmp_path / "embedding-adapter"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        oft_export.save_hf_oft_adapter(
            auto_bridge,
            object(),
            save_dir,
            _OFTConfig(target_modules=("word_embeddings",), layers_to_transform=()),
            show_progress=False,
        )
        loaded = PeftModel.from_pretrained(base_model, save_dir)

    actual = loaded.base_model.model.model.embed_tokens.oft_embedding_R["default"].weight
    torch.testing.assert_close(actual, expected)


@pytest.mark.unit
def test_save_hf_oft_adapter_rejects_empty_model(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oft_export, "export_oft_adapter_weights", lambda *args, **kwargs: iter(()))

    with pytest.raises(RuntimeError, match="No adapter weights were found"):
        oft_export.save_hf_oft_adapter(
            SimpleNamespace(), object(), tmp_path / "adapter", _OFTConfig(), show_progress=False
        )


@pytest.mark.unit
def test_save_hf_oft_adapter_rejects_duplicate_output_key(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    duplicate = "model.layers.0.self_attn.q_proj.oft_R.weight"
    monkeypatch.setattr(
        oft_export,
        "export_oft_adapter_weights",
        lambda *args, **kwargs: iter(((duplicate, torch.zeros(2, 3)), (duplicate, torch.ones(2, 3)))),
    )

    with pytest.raises(RuntimeError, match="duplicate OFT adapter output key"):
        oft_export.save_hf_oft_adapter(
            SimpleNamespace(), object(), tmp_path / "adapter", _OFTConfig(), show_progress=False
        )


@pytest.mark.unit
def test_save_hf_oft_adapter_refuses_existing_destination(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    save_dir = tmp_path / "adapter"
    save_dir.mkdir()
    marker = save_dir / "keep.txt"
    marker.write_text("original")
    monkeypatch.setattr(
        oft_export,
        "export_oft_adapter_weights",
        lambda *args, **kwargs: pytest.fail("destination must be rejected before model export"),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        oft_export.save_hf_oft_adapter(SimpleNamespace(), object(), save_dir, _OFTConfig(), show_progress=False)

    assert marker.read_text() == "original"


@pytest.mark.unit
def test_save_hf_oft_adapter_failure_does_not_publish_partial_directory(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import safetensors.torch

    save_dir = tmp_path / "adapter"
    weights = [("model.layers.0.self_attn.q_proj.oft_R.weight", torch.zeros(2, 120))]
    monkeypatch.setattr(oft_export, "export_oft_adapter_weights", lambda *args, **kwargs: iter(weights))
    monkeypatch.setattr(oft_export, "_installed_peft_version", lambda: "0.19.1")
    monkeypatch.setattr(safetensors.torch, "save_file", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        oft_export.save_hf_oft_adapter(SimpleNamespace(), object(), save_dir, _OFTConfig(), show_progress=False)

    assert not save_dir.exists()
    assert list(tmp_path.glob(".adapter.oft-staging-*")) == []
