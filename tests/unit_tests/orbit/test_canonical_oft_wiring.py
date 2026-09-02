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
from megatron.bridge.orbit.oft.oft_layers import OFTLinear


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
    """CanonicalOFT.__post_init__ asserts against fused leaves; translation must satisfy it."""
    peft = CanonicalOFT(target_modules=canonical_target_modules(["linear_qkv", "linear_proj", "linear_fc1"]))

    # canonical_mapping is what match() consults: fused leaf -> split suffixes.
    assert peft.canonical_mapping["linear_qkv"] == {"linear_q", "linear_k", "linear_v"}
    assert peft.canonical_mapping["linear_fc1"] == {"linear_fc1_gate", "linear_fc1_up"}
    assert peft.canonical_mapping["linear_proj"] == {"linear_proj"}


@pytest.mark.unit
def test_canonical_oft_still_rejects_untranslated_fused_targets() -> None:
    """The guard that makes translation necessary is intact."""
    with pytest.raises(AssertionError, match="does not accept target 'linear_qkv'"):
        CanonicalOFT(target_modules=["linear_qkv"])


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
        )


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
    monkeypatch.setattr(canonical_oft, "OFTRotationModule", lambda **kwargs: nn.Identity())


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
