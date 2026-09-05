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

"""CanonicalOFT target-name translation and fused-linear wrapper selection.

Both launchers hardcoded the legacy shared-R ``OFT`` class, which left
``CanonicalOFT`` unreachable and silently gave Q/K/V (and gate/up) a single
shared rotation on Megatron's fused projections. These tests cover the two
halves of that fix: the legacy -> split target translation the launchers apply,
and the wrapper each split target actually selects.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.bridge.orbit.oft.canonical_oft import (
    CanonicalOFT,
    OFTLinearSplitFC1UpGate,
    OFTLinearSplitQKV,
    canonical_target_modules,
)
from megatron.bridge.orbit.oft.oft_layers import OFTLinear, OFTVocabParallelEmbedding


# --------------------------------------------------------------------------
# legacy -> canonical target translation
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("legacy", "expected"),
    [
        # The fused leaves expand to their split siblings.
        (["linear_qkv"], ["linear_q", "linear_k", "linear_v"]),
        (["linear_fc1"], ["linear_fc1_gate", "linear_fc1_up"]),
        # Non-fused targets pass through untouched, order preserved.
        (
            ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
            [
                "linear_q",
                "linear_k",
                "linear_v",
                "linear_proj",
                "linear_fc1_gate",
                "linear_fc1_up",
                "linear_fc2",
            ],
        ),
        # Kimi's MLA projections have no fused form at all.
        (
            ["linear_q_down_proj", "linear_kv_up_proj", "linear_fc2"],
            ["linear_q_down_proj", "linear_kv_up_proj", "linear_fc2"],
        ),
        # Wildcard prefixes survive the expansion.
        (
            ["*.layers.0.*.linear_qkv"],
            ["*.layers.0.*.linear_q", "*.layers.0.*.linear_k", "*.layers.0.*.linear_v"],
        ),
        # Already-split names are left alone rather than double-expanded.
        (["linear_q", "linear_fc1_gate"], ["linear_q", "linear_fc1_gate"]),
        # Overlapping inputs collapse to one entry each.
        (["linear_qkv", "linear_q"], ["linear_q", "linear_k", "linear_v"]),
    ],
)
def test_canonical_target_modules_translates_fused_leaves(legacy: list[str], expected: list[str]) -> None:
    assert canonical_target_modules(legacy) == expected


@pytest.mark.unit
def test_translated_targets_are_accepted_by_canonical_oft() -> None:
    """CanonicalOFT.__post_init__ rejects fused leaves; translation must satisfy it."""
    peft = CanonicalOFT(target_modules=canonical_target_modules(["linear_qkv", "linear_proj", "linear_fc1"]))

    # canonical_mapping is what match() consults: fused leaf -> split suffixes.
    assert peft.canonical_mapping["linear_qkv"] == {"linear_q", "linear_k", "linear_v"}
    assert peft.canonical_mapping["linear_fc1"] == {"linear_fc1_gate", "linear_fc1_up"}
    assert peft.canonical_mapping["linear_proj"] == {"linear_proj"}


@pytest.mark.unit
def test_canonical_oft_rebuilds_mapping_after_target_mutation() -> None:
    peft = CanonicalOFT(target_modules=["linear_q"])
    peft.target_modules = ["*.layers.1.*.linear_fc1_up"]

    peft._init_target_match_state()

    assert set(peft.canonical_mapping) == {
        "*.layers.1.*.linear_fc1",
        "*.layers.1.*.linear_fc1_up",
    }
    assert peft.canonical_mapping["*.layers.1.*.linear_fc1"] == {"linear_fc1_up"}
    assert "linear_qkv" not in peft.canonical_mapping


@pytest.mark.unit
@pytest.mark.parametrize(
    ("module_name", "expected_pattern"),
    [
        ("linear_qkv", "linear_qkv"),
        ("linear_q", "linear_q"),
    ],
)
def test_canonical_oft_records_split_alias_for_fused_and_unfused_layouts(
    module_name: str,
    expected_pattern: str,
) -> None:
    peft = CanonicalOFT(target_modules=["linear_q"])
    peft._reset_target_match_state()

    match = peft.match(nn.Linear(4, 4), name=module_name, prefix="decoder.layers.0.self_attention")

    assert match == (expected_pattern, f"decoder.layers.0.self_attention.{module_name}")
    assert peft._alias_matches["linear_q"] == {f"decoder.layers.0.self_attention.{module_name}"}


@pytest.mark.unit
@pytest.mark.parametrize("target", ["linear_qkv", "linear_fc1"])
def test_canonical_oft_still_rejects_untranslated_fused_targets(target: str) -> None:
    """The guard remains an explicit runtime error even under optimized Python."""
    with pytest.raises(ValueError, match=rf"does not accept target '{target}'"):
        CanonicalOFT(target_modules=[target])


@pytest.mark.unit
@pytest.mark.parametrize("implementation_name", ["oft", "canonical_oft"])
@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"r": -1, "block_size": 0}, "r must be non-negative"),
        ({"r": 0, "block_size": -1}, "block_size must be non-negative"),
        ({"r": 0, "block_size": 1}, "block_size must be 0 or at least 2"),
        ({"r": 0, "block_size": 0}, "exactly one of r or block_size must be positive"),
        ({"r": 2, "block_size": 4}, "exactly one of r or block_size must be positive"),
        ({"coft": True, "eps": 0.0}, "eps must be finite and positive"),
        ({"coft": True, "eps": float("nan")}, "eps must be finite and positive"),
        ({"module_dropout": -0.1}, "module_dropout must be finite"),
        ({"module_dropout": float("inf")}, "module_dropout must be finite"),
    ],
)
def test_public_oft_config_rejects_invalid_hyperparameters_without_matching_a_module(
    implementation_name: str,
    kwargs: dict[str, object],
    error: str,
) -> None:
    if implementation_name == "oft":
        from megatron.bridge.orbit.oft.oft import OFT as peft_type
    else:
        peft_type = CanonicalOFT

    with pytest.raises(ValueError, match=error):
        peft_type(target_modules=["does_not_match"], **kwargs)


# --------------------------------------------------------------------------
# wrapper selection on fused Megatron linears
# --------------------------------------------------------------------------


class _FakeMegatronLinear(nn.Module):
    """Stands in for a Megatron parallel linear: has a ``config``, is not nn.Linear."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.config = SimpleNamespace(
            num_attention_heads=8,
            num_query_groups=8,
            kv_channels=8,
            sequence_parallel=False,
            params_dtype=torch.float32,
            gated_linear_unit=True,
        )


class _StubRotation(nn.Module):
    """Tiny trainable rotation stand-in for wrapper wiring/state-key checks."""

    def __init__(self) -> None:
        super().__init__()
        self.oft_r = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


@pytest.fixture
def stub_adapter_plumbing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the two runtime dependencies transform() has on a live Megatron model.

    ``get_oft_adapter_attributes_from_linear`` inspects real TP state, and
    ``OFTRotationModule`` allocates the rotation bank. Neither is under test here
    -- the assertion is purely which wrapper class comes back.
    """
    import megatron.bridge.orbit.oft.canonical_oft as canonical_oft

    monkeypatch.setattr(
        canonical_oft,
        "get_oft_adapter_attributes_from_linear",
        lambda module, is_expert=False: SimpleNamespace(
            input_is_parallel=False,
            in_features=module.in_features,
            out_features=module.out_features,
        ),
    )
    monkeypatch.setattr(canonical_oft, "OFTRotationModule", lambda **kwargs: _StubRotation())


@pytest.mark.unit
def test_fused_qkv_gets_three_independent_rotations(stub_adapter_plumbing: None) -> None:
    """The whole point of the fix: linear_qkv must not end up with one shared R."""
    peft = CanonicalOFT(target_modules=canonical_target_modules(["linear_qkv"]))

    wrapper = peft.transform(
        _FakeMegatronLinear(512, 1536),
        name="linear_qkv",
        prefix="decoder.layers.0.self_attention",
    )

    assert isinstance(wrapper, OFTLinearSplitQKV)
    assert wrapper.adapter_q is not wrapper.adapter_k
    assert wrapper.adapter_k is not wrapper.adapter_v


@pytest.mark.unit
def test_canonical_oft_rejects_output_gated_qkv_before_wrapping(stub_adapter_plumbing: None) -> None:
    """The HF q_proj combines query and output-gate rows, so split export is not equivalent."""
    peft = CanonicalOFT(target_modules=["linear_q"])
    module = _FakeMegatronLinear(512, 2048)
    module.config.attention_output_gate = True

    with pytest.raises(ValueError, match="attention_output_gate=True"):
        peft.transform(
            module,
            name="linear_qkv",
            prefix="decoder.layers.0.self_attention",
        )


@pytest.mark.unit
def test_fused_fc1_gets_separate_gate_and_up_rotations(stub_adapter_plumbing: None) -> None:
    peft = CanonicalOFT(target_modules=canonical_target_modules(["linear_fc1"]))

    wrapper = peft.transform(
        _FakeMegatronLinear(512, 2048),
        name="linear_fc1",
        prefix="decoder.layers.0.mlp",
    )

    assert isinstance(wrapper, OFTLinearSplitFC1UpGate)
    assert wrapper.adapter_gate is not wrapper.adapter_up


@pytest.mark.unit
@pytest.mark.parametrize(
    ("targets", "module_name", "prefix", "out_features", "expected_adapters"),
    [
        (["linear_q"], "linear_qkv", "decoder.layers.0.self_attention", 1536, ("q",)),
        (["linear_k", "linear_v"], "linear_qkv", "decoder.layers.0.self_attention", 1536, ("k", "v")),
        (["linear_fc1_gate"], "linear_fc1", "decoder.layers.0.mlp", 2048, ("gate",)),
        (["linear_fc1_up"], "linear_fc1", "decoder.layers.0.mlp", 2048, ("up",)),
    ],
)
def test_canonical_oft_instantiates_only_requested_fused_adapters(
    stub_adapter_plumbing: None,
    targets: list[str],
    module_name: str,
    prefix: str,
    out_features: int,
    expected_adapters: tuple[str, ...],
) -> None:
    peft = CanonicalOFT(target_modules=targets)

    wrapper = peft.transform(
        _FakeMegatronLinear(512, out_features),
        name=module_name,
        prefix=prefix,
    )

    assert wrapper._adapter_names == expected_adapters
    adapter_state_keys = {key for key in wrapper.state_dict() if key.startswith("adapter_")}
    assert adapter_state_keys == {f"adapter_{name}.oft_r" for name in expected_adapters}
    assert {name for name, _ in wrapper.named_parameters() if name.startswith("adapter_")} == {
        f"adapter_{name}.oft_r" for name in expected_adapters
    }


@pytest.mark.unit
def test_canonical_oft_target_mutation_controls_the_instantiated_subset(
    stub_adapter_plumbing: None,
) -> None:
    peft = CanonicalOFT(target_modules=["linear_q", "linear_k", "linear_v"])
    peft.target_modules = ["linear_k"]
    peft._init_target_match_state()

    wrapper = peft.transform(
        _FakeMegatronLinear(512, 1536),
        name="linear_qkv",
        prefix="decoder.layers.0.self_attention",
    )

    assert isinstance(wrapper, OFTLinearSplitQKV)
    assert wrapper._adapter_names == ("k",)
    assert {key for key in wrapper.state_dict() if key.startswith("adapter_")} == {"adapter_k.oft_r"}


@pytest.mark.unit
def test_canonical_oft_ignores_non_type_optional_te_placeholder(
    monkeypatch: pytest.MonkeyPatch,
    stub_adapter_plumbing: None,
) -> None:
    import megatron.bridge.orbit.oft.canonical_oft as canonical_oft

    monkeypatch.setattr(canonical_oft, "HAVE_TE_LN_COL_LINEAR", True)
    monkeypatch.setattr(canonical_oft, "TELayerNormColumnParallelLinear", object())

    wrapper = CanonicalOFT(target_modules=["linear_q"]).transform(
        _FakeMegatronLinear(512, 1536),
        name="linear_qkv",
        prefix="decoder.layers.0.self_attention",
    )

    assert isinstance(wrapper, OFTLinearSplitQKV)


@pytest.mark.unit
@pytest.mark.parametrize(
    "prefix",
    ["decoder.layers.0.mlp", "decoder.layers.0.mlp.experts"],
    ids=["dense", "grouped-experts"],
)
@pytest.mark.parametrize("out_features", [2048, 2047], ids=["even-width", "odd-width"])
def test_non_gated_fc1_rejects_split_targets_before_wrapping(
    stub_adapter_plumbing: None,
    prefix: str,
    out_features: int,
) -> None:
    peft = CanonicalOFT(target_modules=canonical_target_modules(["linear_fc1"]))
    module = _FakeMegatronLinear(512, out_features)
    module.config.gated_linear_unit = False

    with pytest.raises(ValueError, match="gated_linear_unit=True"):
        peft.transform(module, name="linear_fc1", prefix=prefix)


@pytest.mark.unit
def test_unfused_targets_still_get_a_single_rotation(stub_adapter_plumbing: None) -> None:
    """linear_proj / linear_fc2 are not fused, so plain OFTLinear stays correct."""
    peft = CanonicalOFT(target_modules=["linear_proj"])

    wrapper = peft.transform(
        _FakeMegatronLinear(512, 512),
        name="linear_proj",
        prefix="decoder.layers.0.self_attention",
    )

    assert isinstance(wrapper, OFTLinear)
    assert not isinstance(wrapper, (OFTLinearSplitQKV, OFTLinearSplitFC1UpGate))


@pytest.mark.unit
@pytest.mark.parametrize("implementation_name", ["oft", "canonical_oft"])
@pytest.mark.parametrize("block_share", [False, True], ids=["nonshared", "shared"])
def test_row_parallel_r_mode_localizes_the_global_block_count(
    monkeypatch: pytest.MonkeyPatch,
    implementation_name: str,
    block_share: bool,
) -> None:
    if implementation_name == "oft":
        import megatron.bridge.orbit.oft.oft as implementation

        peft_type = implementation.OFT
    else:
        import megatron.bridge.orbit.oft.canonical_oft as implementation

        peft_type = implementation.CanonicalOFT

    monkeypatch.setattr(
        implementation,
        "get_oft_adapter_attributes_from_linear",
        lambda module, is_expert=False: SimpleNamespace(
            input_is_parallel=True,
            in_features=module.in_features,
            out_features=module.out_features,
        ),
    )
    monkeypatch.setattr(implementation.parallel_state, "get_tensor_model_parallel_world_size", lambda: 2)
    captured: list[dict[str, object]] = []

    def fake_rotation(**kwargs: object) -> nn.Module:
        captured.append(kwargs)
        return nn.Identity()

    monkeypatch.setattr(implementation, "OFTRotationModule", fake_rotation)

    peft_type(target_modules=["linear_proj"], r=4, block_size=0, block_share=block_share).transform(
        _FakeMegatronLinear(16, 16),
        name="linear_proj",
        prefix="decoder.layers.0.self_attention",
    )

    assert len(captured) == 1
    assert captured[0]["in_features"] == 8
    assert captured[0]["r"] == 2
    assert captured[0]["block_size"] == 0


@pytest.mark.unit
@pytest.mark.parametrize("implementation_name", ["oft", "canonical_oft"])
def test_row_parallel_r_mode_rejects_blocks_that_cross_tp_shards(
    monkeypatch: pytest.MonkeyPatch,
    implementation_name: str,
) -> None:
    if implementation_name == "oft":
        import megatron.bridge.orbit.oft.oft as implementation

        peft_type = implementation.OFT
    else:
        import megatron.bridge.orbit.oft.canonical_oft as implementation

        peft_type = implementation.CanonicalOFT

    monkeypatch.setattr(
        implementation,
        "get_oft_adapter_attributes_from_linear",
        lambda module, is_expert=False: SimpleNamespace(
            input_is_parallel=True,
            in_features=module.in_features,
            out_features=module.out_features,
        ),
    )
    monkeypatch.setattr(implementation.parallel_state, "get_tensor_model_parallel_world_size", lambda: 2)

    with pytest.raises(ValueError, match=r"r \(3\) must be divisible by tensor-parallel size \(2\)"):
        peft_type(target_modules=["linear_proj"], r=3, block_size=0).transform(
            _FakeMegatronLinear(12, 12),
            name="linear_proj",
            prefix="decoder.layers.0.self_attention",
        )


@pytest.mark.unit
@pytest.mark.parametrize("target", ["word_embeddings", "*.word_embeddings"])
def test_canonical_embedding_target_classification_uses_the_matched_module_leaf(
    target: str,
    stub_adapter_plumbing: None,
) -> None:
    peft = CanonicalOFT(target_modules=[target])
    embedding = nn.Embedding(32, 16)

    wrapper = peft.transform(
        embedding,
        name="word_embeddings",
        prefix="embedding",
    )

    assert isinstance(wrapper, OFTVocabParallelEmbedding)
    assert wrapper.to_wrap is embedding


@pytest.mark.unit
def test_canonical_embedding_target_reports_missing_geometry_as_value_error() -> None:
    peft = CanonicalOFT(target_modules=["word_embeddings"])

    with pytest.raises(ValueError, match="Cannot infer embedding_dim"):
        peft.transform(nn.Module(), name="word_embeddings", prefix="embedding")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "target", "module"),
    [
        ("word_embeddings", "word_embeddings", nn.Embedding(32, 16)),
        ("word_embeddings", "*.word_embeddings", nn.Embedding(32, 16)),
        ("output_layer", "output_layer", nn.Linear(16, 32, bias=False)),
        ("output_layer", "*.output_layer", nn.Linear(16, 32, bias=False)),
    ],
)
def test_legacy_oft_rejects_unsupported_all_mode_targets_by_module_leaf(
    name: str,
    target: str,
    module: nn.Module,
) -> None:
    from megatron.bridge.orbit.oft.oft import OFT

    with pytest.raises(NotImplementedError, match=rf"does not support OFT on '{name}'") as exc_info:
        OFT(target_modules=[target]).transform(module, name=name, prefix="model")

    assert "Explicitly select --oft-type canonical_oft" in str(exc_info.value)
    assert "the default" not in str(exc_info.value)


@pytest.mark.unit
def test_legacy_oft_gives_the_fused_projection_one_shared_rotation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Contrast case, documenting exactly what --oft-type oft still selects."""
    import megatron.bridge.orbit.oft.oft as legacy_oft
    from megatron.bridge.orbit.oft.oft import OFT

    monkeypatch.setattr(
        legacy_oft,
        "get_oft_adapter_attributes_from_linear",
        lambda module, is_expert=False: SimpleNamespace(
            input_is_parallel=False,
            in_features=module.in_features,
            out_features=module.out_features,
        ),
    )
    monkeypatch.setattr(legacy_oft, "OFTRotationModule", lambda **kwargs: nn.Identity())

    wrapper = OFT(target_modules=["linear_qkv"]).transform(
        _FakeMegatronLinear(512, 1536),
        name="linear_qkv",
        prefix="decoder.layers.0.self_attention",
    )

    # One OFTLinear, one rotation -- Q, K and V all see the same R.
    assert isinstance(wrapper, OFTLinear)
    assert not isinstance(wrapper, OFTLinearSplitQKV)
