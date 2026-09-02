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

"""Canonical split OFT on quantized fused bases.

The split wrappers used to route any quantized single-weight base through a
shared-R fallback: adapter_q was the only rotation ever called, adapter_k/v
(and adapter_up) were allocated but received no gradient, and the export
carried them as zeros. Now the base weight is dequantized once per forward
(the exact cost the fallback already paid) and the REAL split math runs on the
BF16 copy, with saved_tensors_hooks keeping only the low-bit rebuild handle.

The load-bearing assertions per format:
- equivalence: quantized wrapper == dense wrapper on the dequantized weight,
  outputs and every gradient, for random distinct rotations;
- discriminator: adapter_k / adapter_v / adapter_up receive real gradient
  (under the old fallback these were provably grad-less);
- retention: the dequantized copy is dead after forward.
"""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from megatron.core import parallel_state

import megatron.bridge.orbit.low_precision.nvfp4 as nvfp4_module
import megatron.bridge.orbit.quant.fp8_utils as fp8_module
from megatron.bridge.orbit.oft.canonical_oft import (
    OFTLinearSplitFC1UpGate,
    OFTLinearSplitQKV,
    _dequantize_single_weight_base,
)


_IN = 32
_QKV_OUT = 64  # (4 q heads + 2 k + 2 v) * kv_channels 8
_FC1_OUT = 32  # gate 16 + up 16
_REAL_DEQUANT_NVFP4 = nvfp4_module.dequantize_nvfp4

_PROVIDER = SimpleNamespace(num_attention_heads=4, num_query_groups=2, kv_channels=8, sequence_parallel=False)


@pytest.fixture(autouse=True)
def _stub_parallel_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_group", lambda: object())
    monkeypatch.setattr(parallel_state, "get_expert_tensor_parallel_group", lambda: object())


def _base_module(weight: torch.Tensor) -> torch.nn.Module:
    module = torch.nn.Module()
    module.in_features = weight.shape[1]
    module.out_features = weight.shape[0]
    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    module.config = SimpleNamespace(sequence_parallel=False)
    return module


def _quantized_module(fmt: str, out_features: int) -> tuple[torch.nn.Module, torch.Tensor]:
    """A fused single-weight base in the given format + its dequantized value."""
    torch.manual_seed(7)
    device = "cuda" if fmt == "int4" else "cpu"
    weight = torch.randn(out_features, _IN, device=device) * 0.1

    if fmt == "fp8_direct":
        # Materialized direct-FP8: floating 2D weight + scalar weight_scale_inv.
        # With scale 1.0 the dequantized value is the payload itself.
        module = _base_module(weight.clone())
        module.weight_scale_inv = torch.ones(1)
        w_ref = fp8_module.dequant_fp8(module.weight, module.weight_scale_inv, out_dtype=torch.float32)
    elif fmt == "nvfp4_modelopt":
        from megatron.bridge.orbit.low_precision.nvfp4 import quantize_to_nvfp4

        packed, scale, double_scale, _shape = quantize_to_nvfp4(weight)
        module = torch.nn.Module()
        module.in_features, module.out_features = _IN, out_features
        module.weight = torch.nn.Parameter(packed, requires_grad=False)
        module.weight_quantizer = SimpleNamespace(_scale=scale, _double_scale=double_scale)
        module.config = SimpleNamespace(sequence_parallel=False)
        w_ref = _REAL_DEQUANT_NVFP4(packed, scale, double_scale, (out_features, _IN), dtype=torch.float32)
    elif fmt == "int4":
        from megatron.bridge.orbit.low_precision.int4 import dequantize_int4, quantize_to_int4

        packed, scale, shape = quantize_to_int4(weight.to(torch.bfloat16))
        module = torch.nn.Module()
        module.in_features, module.out_features = _IN, out_features
        module.weight = torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=False)
        module.weight_packed = packed
        module.weight_scale = scale
        module.weight_shape = shape
        module.config = SimpleNamespace(sequence_parallel=False)
        w_ref = dequantize_int4(packed, scale, shape, device=device).to(torch.float32)
    else:  # pragma: no cover
        raise AssertionError(fmt)
    return module, w_ref


_FORMATS = [
    pytest.param("fp8_direct", id="fp8_direct"),
    pytest.param("nvfp4_modelopt", id="nvfp4_modelopt"),
    pytest.param(
        "int4",
        id="int4",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="INT4 dequant is triton"),
    ),
]


def _make_qkv(module) -> OFTLinearSplitQKV:
    return OFTLinearSplitQKV(module, in_features=_IN, provider=_PROVIDER, block_size=4, input_is_parallel=True)


def _make_fc1(module) -> OFTLinearSplitFC1UpGate:
    return OFTLinearSplitFC1UpGate(module, in_features=_IN, block_size=4, input_is_parallel=True)


def _randomize_adapters(wrapper: torch.nn.Module, seed: int, dtype: torch.dtype) -> None:
    names = [n for n in ("adapter_q", "adapter_k", "adapter_v", "adapter_gate", "adapter_up") if hasattr(wrapper, n)]
    with torch.no_grad():
        for offset, name in enumerate(names):
            torch.manual_seed(seed + offset)
            adapter = getattr(wrapper, name)
            adapter.oft_r.copy_(torch.randn_like(adapter.oft_r) * 0.02)


def _copy_adapters(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    for name in ("adapter_q", "adapter_k", "adapter_v", "adapter_gate", "adapter_up"):
        if hasattr(src, name):
            with torch.no_grad():
                getattr(dst, name).oft_r.copy_(getattr(src, name).oft_r)


@pytest.mark.unit
@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("make", [_make_qkv, _make_fc1], ids=["qkv", "fc1"])
def test_quantized_split_matches_dense_split_on_dequantized_weight(fmt: str, make) -> None:
    """Outputs and ALL gradients must equal the dense wrapper run on w_ref."""
    out_features = _QKV_OUT if make is _make_qkv else _FC1_OUT
    module, w_ref = _quantized_module(fmt, out_features)
    device = w_ref.device

    quant_wrapper = make(module)
    dense_wrapper = make(_base_module(w_ref.clone()))
    for wrapper in (quant_wrapper, dense_wrapper):
        wrapper.to(device)
    _randomize_adapters(quant_wrapper, seed=11, dtype=torch.float32)
    _copy_adapters(quant_wrapper, dense_wrapper)

    x_q = torch.randn(6, _IN, device=device, requires_grad=True)
    x_d = x_q.detach().clone().requires_grad_(True)

    out_q, _ = quant_wrapper(x_q)
    out_d, _ = dense_wrapper(x_d)
    torch.testing.assert_close(out_q, out_d)

    grad = torch.randn_like(out_q)
    out_q.backward(grad)
    out_d.backward(grad)
    torch.testing.assert_close(x_q.grad, x_d.grad)
    for name in ("adapter_q", "adapter_k", "adapter_v", "adapter_gate", "adapter_up"):
        if hasattr(quant_wrapper, name):
            g_q = getattr(quant_wrapper, name).oft_r.grad
            g_d = getattr(dense_wrapper, name).oft_r.grad
            assert g_q is not None and g_d is not None
            torch.testing.assert_close(g_q, g_d)


@pytest.mark.unit
@pytest.mark.parametrize("fmt", _FORMATS)
def test_every_rotation_gets_gradient_on_quantized_base(fmt: str) -> None:
    """THE discriminator vs the retired fallback: adapter_k/adapter_v trained.

    Under the shared-R fallback only adapter_q was ever called; adapter_k and
    adapter_v stayed grad-less forever (verified on-cluster before the change).
    """
    module, w_ref = _quantized_module(fmt, _QKV_OUT)
    wrapper = _make_qkv(module).to(w_ref.device)

    x = torch.randn(5, _IN, device=w_ref.device, requires_grad=True)
    out, _ = wrapper(x)
    out.sum().backward()

    for name in ("adapter_q", "adapter_k", "adapter_v"):
        grad = getattr(wrapper, name).oft_r.grad
        assert grad is not None, f"{name} received no gradient — shared-R fallback semantics"
        assert torch.isfinite(grad).all()


@pytest.mark.unit
@pytest.mark.parametrize("fmt", ["fp8_direct", "nvfp4_modelopt"])
def test_dequantized_weight_released_after_forward(fmt: str, monkeypatch: pytest.MonkeyPatch) -> None:
    refs: list[weakref.ref] = []
    if fmt == "fp8_direct":
        target_module, attr = fp8_module, "dequant_fp8"
    else:
        target_module, attr = nvfp4_module, "dequantize_nvfp4"
    real = getattr(target_module, attr)

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        refs.append(weakref.ref(out))
        return out

    monkeypatch.setattr(target_module, attr, spy)

    module, _ = _quantized_module(fmt, _QKV_OUT)
    wrapper = _make_qkv(module)
    x = torch.randn(4, _IN, requires_grad=True)
    out, _ = wrapper(x)
    gc.collect()

    assert refs, "expected the forward to dequantize"
    assert all(ref() is None for ref in refs), "dequantized fused weight retained by the graph"
    out.sum().backward()
    assert wrapper.adapter_q.oft_r.grad is not None


@pytest.mark.unit
def test_disabled_adapter_on_quantized_base_is_plain_base_linear() -> None:
    module, w_ref = _quantized_module("fp8_direct", _QKV_OUT)
    wrapper = _make_qkv(module)
    wrapper.disable_adapter_layers()

    x = torch.randn(3, _IN)
    out, bias = wrapper(x)

    assert bias is None
    torch.testing.assert_close(out, torch.nn.functional.linear(x, w_ref))


@pytest.mark.unit
def test_helper_returns_none_for_plain_bf16_base() -> None:
    module = _base_module(torch.randn(_FC1_OUT, _IN))
    assert _dequantize_single_weight_base(module, torch.float32) is None
