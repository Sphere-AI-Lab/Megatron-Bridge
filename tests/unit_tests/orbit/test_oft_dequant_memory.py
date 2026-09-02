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

"""Dequantized base weights must not be retained by the autograd graph.

The OFT-rotated input requires grad, so each ``F.linear`` saves its WEIGHT for
backward (``grad_input = grad_output @ W``) -- and that weight is the temporary
BF16 dequantized copy, not the persistent low-bit buffer. Every quantized
forward therefore wraps its GEMMs in ``saved_tensors_hooks`` that substitute a
low-bit handle at save time and re-dequantize during backward. These tests pin
that guarantee per format with a deterministic probe: a ``weakref`` to the
dequantized tensor must be DEAD once forward returns (the graph holds only the
handle), while outputs and gradients stay bit-identical to an unhooked
reference.

``_forward_nvfp4_grouped_buffers`` -- the Qwen3-MoE NVFP4 direct-checkpoint
path -- was the one forward missing its hooks: with no recomputation on that
recipe, the retained copies accumulated across every layer (~4x the whole
4-bit payload). The INT4/FP8 tests are regression guards on paths that were
already hooked.
"""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from megatron.core import parallel_state

import megatron.bridge.orbit.low_precision.nvfp4 as nvfp4_module


# Captured at import, before any monkeypatch: fixture reference weights must not
# pollute the spy's record of what the forward dequantized.
_REAL_DEQUANTIZE_NVFP4 = nvfp4_module.dequantize_nvfp4
from megatron.bridge.orbit.oft.oft_layers import OFTLinear, OFTRotationModule


@pytest.fixture(autouse=True)
def _stub_parallel_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructors resolve a TP group eagerly; none of these tests shard."""
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_group", lambda: object())
    monkeypatch.setattr(parallel_state, "get_expert_tensor_parallel_group", lambda: object())


_IN, _OUT, _TOKENS = 32, 16, (6, 10)


def _grouped_nvfp4_module(swiglu_halves: bool) -> tuple[torch.nn.Module, dict[int, torch.Tensor]]:
    """A fake TEGroupedLinear carrying real NVFP4 buffers for two experts."""
    from megatron.bridge.orbit.low_precision.nvfp4 import quantize_to_nvfp4

    module = torch.nn.Module()
    module.num_gemms = 2
    module.config = SimpleNamespace(sequence_parallel=False)
    reference_weights = {}
    for idx in range(module.num_gemms):
        torch.manual_seed(100 + idx)
        weight = torch.randn(_OUT, _IN) * 0.1
        packed, scale, double_scale, shape = quantize_to_nvfp4(weight)
        setattr(module, f"weight{idx}", torch.nn.Parameter(torch.zeros(1), requires_grad=False))
        setattr(module, f"weight_scale{idx}", scale)
        setattr(module, f"weight_double_scale{idx}", double_scale)
        if swiglu_halves:
            half = _OUT // 2
            setattr(module, f"weight{idx}_w_packed", packed[:half].clone())
            setattr(module, f"weight{idx}_v_packed", packed[half:].clone())
        else:
            setattr(module, f"weight{idx}_packed", packed)
        # What the forward will actually multiply by (quantization is lossy, so
        # the reference is the DEQUANTIZED weight, not the original).
        reference_weights[idx] = _REAL_DEQUANTIZE_NVFP4(packed, scale, double_scale, (_OUT, _IN), dtype=torch.float32)
    return module, reference_weights


def _wrap(module: torch.nn.Module) -> OFTLinear:
    adapter = OFTRotationModule(in_features=_IN, block_size=4, input_is_parallel=True, dtype=torch.float32)
    return OFTLinear(module, adapter)


@pytest.fixture
def dequant_spy(monkeypatch: pytest.MonkeyPatch):
    """Record a weakref to every tensor dequantize_nvfp4 produces."""
    refs: list[weakref.ref] = []
    real = nvfp4_module.dequantize_nvfp4

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        refs.append(weakref.ref(out))
        return out

    monkeypatch.setattr(nvfp4_module, "dequantize_nvfp4", spy)
    return refs


@pytest.mark.unit
@pytest.mark.parametrize("swiglu_halves", [False, True], ids=["fc2_packed", "fc1_swiglu_halves"])
def test_nvfp4_grouped_buffers_releases_dequantized_weights(dequant_spy, swiglu_halves: bool) -> None:
    """After forward, the graph must hold low-bit handles, not BF16 copies."""
    module, reference_weights = _grouped_nvfp4_module(swiglu_halves)
    wrapper = _wrap(module)
    tokens = torch.tensor(_TOKENS)
    x = torch.randn(sum(_TOKENS), _IN, requires_grad=True)

    out, _ = wrapper.forward(x, tokens)
    gc.collect()

    dead = [ref() is None for ref in dequant_spy]
    assert len(dequant_spy) == 2, f"expected one dequantize per active expert, saw {len(dequant_spy)}"
    assert all(dead), (
        f"{dead.count(False)} of {len(dead)} dequantized weights still alive after forward: "
        "the autograd graph is retaining BF16 copies instead of low-bit handles"
    )

    # Numerics: identical to an unhooked per-expert reference on the same input
    # (zero-initialized oft_r rotates by identity, so the rotation is a no-op
    # value-wise while still making the input require grad).
    x_ref = x.detach().clone().requires_grad_(True)
    chunks = torch.split(x_ref, list(_TOKENS), dim=0)
    ref_out = torch.cat([torch.nn.functional.linear(chunks[idx], reference_weights[idx]) for idx in range(2)], dim=0)
    torch.testing.assert_close(out, ref_out)

    # Backward: unpack must reconstruct the exact weights (grad_input = dy @ W),
    # and gradient must reach the adapter through the re-dequantized GEMM.
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    ref_out.backward(grad_out)
    torch.testing.assert_close(x.grad, x_ref.grad)
    assert wrapper.adapter.oft_r.grad is not None
    assert torch.isfinite(wrapper.adapter.oft_r.grad).all()


@pytest.mark.unit
def test_nvfp4_grouped_buffers_backward_redequantizes(dequant_spy) -> None:
    """The trade the hooks make: one extra dequantize per expert, in backward."""
    module, _ = _grouped_nvfp4_module(swiglu_halves=False)
    wrapper = _wrap(module)
    x = torch.randn(sum(_TOKENS), _IN, requires_grad=True)

    out, _ = wrapper.forward(x, torch.tensor(_TOKENS))
    forward_count = len(dequant_spy)
    out.sum().backward()

    assert forward_count == 2  # one per active expert in forward
    assert len(dequant_spy) == 4  # one more per expert during backward


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="INT4 dequantize is a triton kernel")
def test_int4_grouped_chunked_releases_dequantized_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard: the INT4 grouped-chunked path was already hooked.

    Targets the grouped layout (what Kimi/Moonlight INT4 actually runs), whose
    python backend does plain per-chunk F.linear -- the same retention contract
    as the NVFP4 buffers path. The dense eager INT4 path is deliberately not
    probed this way: it injects the dequantized tensor into to_wrap.weight and
    calls the module, so the module itself keeps one copy alive by design.
    """
    import megatron.bridge.orbit.low_precision.int4 as int4_module
    from megatron.bridge.orbit.low_precision.int4 import quantize_to_int4

    refs: list[weakref.ref] = []
    real = int4_module.dequantize_int4

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        refs.append(weakref.ref(out))
        return out

    monkeypatch.setattr(int4_module, "dequantize_int4", spy)

    module = torch.nn.Module()
    module.num_gemms = 2
    module.config = SimpleNamespace(sequence_parallel=False)
    for idx in range(2):
        torch.manual_seed(200 + idx)
        weight = (torch.randn(_OUT, _IN, device="cuda") * 0.1).to(torch.bfloat16)
        packed, scale, shape = quantize_to_int4(weight)
        setattr(module, f"weight{idx}", torch.nn.Parameter(torch.zeros(1), requires_grad=False))
        setattr(module, f"weight{idx}_packed", packed)
        setattr(module, f"weight{idx}_scale", scale)
        setattr(module, f"weight{idx}_shape", shape)

    adapter = OFTRotationModule(
        in_features=_IN, block_size=4, input_is_parallel=True, dtype=torch.bfloat16, device="cuda"
    )
    wrapper = OFTLinear(module, adapter)
    wrapper._int4_grouped_chunk_backend = "python"
    wrapper._int4_active_expert_chunk_size = 2

    x = torch.randn(sum(_TOKENS), _IN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out, _ = wrapper.forward(x, torch.tensor(_TOKENS))
    gc.collect()

    assert refs and all(ref() is None for ref in refs)
    out.sum().backward()
    assert wrapper.adapter.oft_r.grad is not None
