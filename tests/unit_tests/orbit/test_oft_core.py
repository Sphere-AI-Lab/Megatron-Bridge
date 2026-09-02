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

import pytest
import torch
import torch.nn as nn
from megatron.core import parallel_state
from megatron.core.transformer import utils as transformer_utils

from megatron.bridge.orbit.oft.canonical_oft import GroupedOFTRotation, _split_wrapper_sharded_state_dict
from megatron.bridge.orbit.oft.oft import _SplitLNOFTLinear
from megatron.bridge.orbit.oft.oft_layers import MultiplicativeDropoutLayer, OFTRotationModule
from megatron.bridge.orbit.oft.param_names import is_peft_adapter_param_name, is_trainable_base_param_name


@pytest.fixture(autouse=True)
def mock_tensor_parallel_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_group", lambda: object())
    monkeypatch.setattr(parallel_state, "get_expert_tensor_parallel_group", lambda: object())


@pytest.mark.unit
def test_oft_rotation_starts_as_identity() -> None:
    adapter = OFTRotationModule(in_features=8, block_size=4, input_is_parallel=True, dtype=torch.float32)
    x = torch.randn(3, 8)

    assert torch.equal(adapter(x), x)
    assert torch.equal(adapter.get_delta_weight(), torch.eye(8))


@pytest.mark.unit
def test_exact_cayley_rotation_is_orthogonal_and_differentiable() -> None:
    adapter = OFTRotationModule(in_features=4, block_size=4, input_is_parallel=True, dtype=torch.float64)
    q = torch.tensor([[0.1, -0.2, 0.05, 0.15, -0.1, 0.2]], dtype=torch.float64, requires_grad=True)

    rotation = adapter._cayley_batch(q, block_size=4, use_cayley_neumann=False)
    identity = torch.eye(4, dtype=torch.float64).unsqueeze(0)

    torch.testing.assert_close(rotation @ rotation.transpose(-1, -2), identity, atol=1e-12, rtol=1e-12)
    rotation.square().sum().backward()
    assert q.grad is not None
    assert torch.isfinite(q.grad).all()


@pytest.mark.unit
def test_oft_rotation_adjusts_invalid_block_size_to_nearest_divisor() -> None:
    adapter = OFTRotationModule(in_features=12, block_size=5, input_is_parallel=True)

    assert adapter.block_size == 6
    assert adapter.r == 2
    assert adapter.oft_r.shape == (2, 15)


@pytest.mark.unit
def test_multiplicative_dropout_replaces_every_block_with_identity() -> None:
    dropout = MultiplicativeDropoutLayer(p=1.0).train()
    rotations = torch.randn(3, 4, 4)

    result = dropout(rotations)

    torch.testing.assert_close(result, torch.eye(4).repeat(3, 1, 1))


@pytest.mark.unit
@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
def test_split_fused_norm_honors_zero_centered_gamma(normalization: str) -> None:
    split = _SplitLNOFTLinear.__new__(_SplitLNOFTLinear)
    nn.Module.__init__(split)
    split._normalization = normalization
    split._hidden_size = 4
    split._eps = 1e-5
    split._orig_module = nn.Module()
    split._orig_module.layer_norm_weight = nn.Parameter(torch.zeros(4))
    split._orig_module.layer_norm_bias = nn.Parameter(torch.zeros(4))
    split._orig_module.zero_centered_gamma = True
    x = torch.tensor([[1.0, 2.0, 4.0, 8.0]])

    actual = split._apply_norm(x)
    if normalization == "RMSNorm":
        expected = torch.nn.functional.rms_norm(x, (4,), torch.ones(4), split._eps)
    else:
        expected = torch.nn.functional.layer_norm(x, (4,), torch.ones(4), torch.zeros(4), split._eps)

    torch.testing.assert_close(actual, expected)


@pytest.mark.unit
def test_grouped_oft_eager_fallback_applies_coft_projection() -> None:
    rotation = GroupedOFTRotation(
        num_local_experts=2,
        in_features=8,
        block_size=4,
        coft=True,
        eps=_COFT_EPS,
        input_is_parallel=True,
    ).eval()
    with torch.no_grad():
        rotation.oft_r.fill_(1.0)

    rotation(torch.randn(3, 8), expert_idx=1)

    projected = rotation.oft_r[1]
    norms = _generator_norms(rotation._template, projected)
    torch.testing.assert_close(norms, torch.full_like(norms, _COFT_EPS), rtol=1e-5, atol=0.0)


@pytest.mark.unit
def test_grouped_oft_eager_fallback_applies_module_dropout() -> None:
    rotation = GroupedOFTRotation(
        num_local_experts=2,
        in_features=8,
        block_size=4,
        module_dropout=1.0,
        input_is_parallel=True,
    ).train()
    with torch.no_grad():
        rotation.oft_r.fill_(0.2)
    x = torch.randn(3, 8)

    torch.testing.assert_close(rotation(x, expert_idx=0), x)


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    [
        "decoder.layers.0.linear_qkv.adapter.oft_r",
        "decoder.layers.0.linear_qkv.adapter_q.oft_r",
        "decoder.layers.0.linear_fc1.adapter_gate.oft_r",
        "decoder.layers.0.linear_fc1.lora_a",
    ],
)
def test_parameter_name_predicates_recognize_all_adapter_shapes(name: str) -> None:
    assert is_peft_adapter_param_name(name)
    assert not is_trainable_base_param_name(name)


@pytest.mark.unit
def test_parameter_name_predicates_reject_wrapped_and_adapter_base_parameters() -> None:
    assert not is_trainable_base_param_name("decoder.layers.0.linear_qkv.to_wrap.weight")
    assert is_trainable_base_param_name("decoder.layers.0.linear_qkv.weight")


@pytest.mark.unit
def test_split_wrapper_sharded_state_dict_delegates_each_child(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_sharded_state_dict_default(child, prefix, sharded_offsets, metadata, tp_group):
        calls.append((child, prefix, sharded_offsets, metadata, tp_group))
        return {f"{prefix}weight": child.weight}

    monkeypatch.setattr(transformer_utils, "sharded_state_dict_default", fake_sharded_state_dict_default)
    wrapper = nn.Module()
    wrapper.to_wrap = nn.Linear(4, 4, bias=False)
    wrapper.to_wrap.tp_group = "base-tp"
    wrapper.adapter_q = nn.Linear(4, 4, bias=False)
    wrapper.adapter_q.tp_group = "adapter-tp"

    result = _split_wrapper_sharded_state_dict(
        wrapper,
        prefix="decoder.",
        sharded_offsets=((0, 0, 1),),
        metadata={"dp_cp_group": "dp"},
    )

    assert set(result) == {"decoder.to_wrap.weight", "decoder.adapter_q.weight"}
    assert [(prefix, tp_group) for _, prefix, _, _, tp_group in calls] == [
        ("decoder.to_wrap.", "base-tp"),
        ("decoder.adapter_q.", "adapter-tp"),
    ]


# ---------------------------------------------------------------------------
# Constrained OFT (--coft) epsilon-ball projection
#
# The projection is centred on zero, on the skew-symmetric generator, matching
# the reference PEFT implementation. It previously measured ||Q_skew - I||,
# which for a skew-symmetric Q is sqrt(||Q||^2 + block_size) and therefore never
# below sqrt(block_size); the resulting shrink factor was a near-constant
# eps/sqrt(block_size) that ignored the parameter and had no fixed point above
# zero, so repeated forwards annihilated the adapter.
# ---------------------------------------------------------------------------

_COFT_EPS = 1e-1


def _coft_adapter() -> OFTRotationModule:
    adapter = OFTRotationModule(
        in_features=8,
        block_size=4,
        coft=True,
        eps=_COFT_EPS,
        input_is_parallel=True,
        dtype=torch.float64,
    )
    adapter.eval()
    return adapter


def _generator_norms(adapter: OFTRotationModule, weight: torch.Tensor) -> torch.Tensor:
    """Frobenius norm of each block's skew-symmetric generator."""
    skew = adapter._pytorch_skew_symmetric(weight, adapter.block_size)
    return torch.norm(skew, p="fro", dim=(-2, -1))


@pytest.mark.unit
def test_coft_projection_leaves_blocks_inside_the_ball_untouched() -> None:
    """Below the boundary the projection must be a no-op."""
    adapter = _coft_adapter()
    weight = torch.full_like(adapter.oft_r, 1e-3)
    assert float(_generator_norms(adapter, weight).max()) < _COFT_EPS

    projected = adapter._project_batch(weight, eps=_COFT_EPS)

    # rtol accommodates the 1e-8 guard in the denominator, not a real change.
    torch.testing.assert_close(projected, weight, rtol=1e-4, atol=0.0)


@pytest.mark.unit
def test_coft_projection_pulls_blocks_outside_the_ball_onto_the_boundary() -> None:
    """Above the boundary the generator norm must land exactly on eps."""
    adapter = _coft_adapter()
    weight = torch.full_like(adapter.oft_r, 1.0)
    assert float(_generator_norms(adapter, weight).min()) > _COFT_EPS

    norms = _generator_norms(adapter, adapter._project_batch(weight, eps=_COFT_EPS))

    torch.testing.assert_close(norms, torch.full_like(norms, _COFT_EPS), rtol=1e-6, atol=0.0)


@pytest.mark.unit
def test_coft_projection_is_idempotent_on_the_boundary() -> None:
    """Projecting twice must equal projecting once -- the boundary is a fixed point.

    Under the identity-centred version the second call shrank again by another
    ~1e-5, which is what drove the parameter to zero.
    """
    adapter = _coft_adapter()
    weight = torch.full_like(adapter.oft_r, 1.0)

    once = adapter._project_batch(weight, eps=_COFT_EPS)
    twice = adapter._project_batch(once, eps=_COFT_EPS)

    torch.testing.assert_close(twice, once, rtol=1e-4, atol=0.0)


@pytest.mark.unit
def test_coft_rotation_does_not_collapse_the_parameter_over_repeated_forwards() -> None:
    """End-to-end regression: _compute_rotation writes the projection back into
    oft_r in place on every forward, so a non-idempotent projection compounds."""
    adapter = _coft_adapter()
    with torch.no_grad():
        adapter.oft_r.fill_(1e-3)
    before = adapter.oft_r.detach().clone()

    for _ in range(8):
        adapter._compute_rotation()

    # Identity-centred projection shrank by ~0.05 per call here, reaching 3.9e-14
    # by the eighth. (At the shipped block_size=32 / eps=6e-5 the factor is 1e-5.)
    torch.testing.assert_close(adapter.oft_r, before, rtol=1e-4, atol=0.0)


@pytest.mark.unit
def test_coft_scale_tracks_the_parameter_not_the_block_size() -> None:
    """A parameter 10x further outside the ball must be shrunk 10x harder.

    Uses a tight eps and small parameters -- the regime real training sits in,
    where ``sqrt(block_size)`` completely swamps ``||Q||``. The identity-centred
    version returned 5.00e-05 for both (ratio 1.0001), i.e. a shrink factor that
    ignored the parameter entirely; centred on zero the ratio is exactly 10.
    """
    adapter = _coft_adapter()
    eps = 1e-4
    near = torch.full_like(adapter.oft_r, 1e-3)
    far = torch.full_like(adapter.oft_r, 1e-2)
    assert float(_generator_norms(adapter, near).min()) > eps

    ratio_near = (adapter._project_batch(near, eps=eps) / near).mean()
    ratio_far = (adapter._project_batch(far, eps=eps) / far).mean()

    torch.testing.assert_close(ratio_near / ratio_far, torch.tensor(10.0, dtype=ratio_near.dtype), rtol=1e-4, atol=0.0)


@pytest.mark.unit
@pytest.mark.parametrize("input_is_parallel", [True, False], ids=["row_parallel", "column_parallel"])
def test_coft_forward_runs_on_both_tensor_parallel_layouts(input_is_parallel: bool) -> None:
    """Regression: --coft used to raise in forward on column-parallel targets.

    linear_qkv and linear_fc1 are column-parallel, where oft_r is replicated and
    handed to copy_to_tensor_model_parallel_region. The projection used to be
    applied to that wrapper's *output*, which is an autograd view of the
    parameter; the in-place write never reached the parameter and the following
    _cayley_batch raised "Output 0 of _CopyToModelParallelRegion is a view and
    its base ... has been modified inplace". So --coft could not run at all on
    the two most important targets. Projecting the leaf parameter first works on
    both layouts.

    Forward only: the column-parallel backward all-reduces oft_r's gradient
    through mcore's _reduce(), which needs a real process group. That plumbing is
    unrelated to this fix; gradient flow is covered for the row-parallel layout
    below, and end to end by the distributed tests.
    """
    adapter = OFTRotationModule(
        in_features=8,
        block_size=4,
        coft=True,
        eps=_COFT_EPS,
        input_is_parallel=input_is_parallel,
        dtype=torch.float64,
    )
    adapter.eval()
    with torch.no_grad():
        adapter.oft_r.fill_(1.0)  # outside the ball, so the projection must act

    out = adapter(torch.randn(3, 8, dtype=torch.float64))

    assert torch.isfinite(out).all()
    # The constraint landed on the Parameter itself, on both layouts.
    norms = _generator_norms(adapter, adapter.oft_r.detach())
    torch.testing.assert_close(norms, torch.full_like(norms, _COFT_EPS), rtol=1e-6, atol=0.0)


@pytest.mark.unit
def test_coft_row_parallel_backward_still_reaches_oft_r() -> None:
    """The in-place projection must not sever the autograd graph."""
    adapter = OFTRotationModule(
        in_features=8, block_size=4, coft=True, eps=_COFT_EPS, input_is_parallel=True, dtype=torch.float64
    )
    adapter.eval()
    with torch.no_grad():
        adapter.oft_r.fill_(1.0)

    adapter(torch.randn(3, 8, dtype=torch.float64, requires_grad=True)).sum().backward()

    assert adapter.oft_r.grad is not None
    assert torch.isfinite(adapter.oft_r.grad).all()
