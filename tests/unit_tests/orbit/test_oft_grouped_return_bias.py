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

"""Returned-bias contracts for quantized grouped FC2 OFT fallbacks."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import megatron.bridge.orbit.low_precision.nvfp4 as nvfp4_module
import megatron.bridge.orbit.oft.oft_layers as oft_layers
import megatron.bridge.orbit.quant.fp8_utils as fp8_module
from megatron.bridge.orbit.oft.oft_layers import OFTLinear


_EXPERTS = 2
_IN_FEATURES = 16
_OUT_FEATURES = 3
_TOKENS = (1, 2)


def _zero_weight(*args, **kwargs) -> torch.Tensor:
    shape = args[3] if len(args) > 3 else kwargs.get("shape")
    dtype = kwargs.get("dtype", torch.float32)
    device = kwargs.get("device", "cpu")
    return torch.zeros(tuple(int(value) for value in shape), dtype=dtype, device=device)


def _grouped_quantized_fc2(kind: str) -> nn.Module:
    module = nn.Module()
    module.num_gemms = _EXPERTS
    module.in_features = _IN_FEATURES
    module.out_features = _OUT_FEATURES
    module.use_bias = True
    module.te_return_bias = True
    module.config = SimpleNamespace(sequence_parallel=False)

    for idx in range(_EXPERTS):
        bias = torch.linspace(0.25 + idx, 0.75 + idx, _OUT_FEATURES)
        setattr(module, f"bias{idx}", nn.Parameter(bias, requires_grad=False))
        if kind == "fp8_direct":
            setattr(
                module,
                f"weight{idx}",
                nn.Parameter(torch.zeros(_OUT_FEATURES, _IN_FEATURES), requires_grad=False),
            )
            setattr(module, f"weight{idx}_scale_inv", torch.ones(1))
        elif kind == "nvfp4_modelopt":
            setattr(
                module,
                f"weight{idx}",
                nn.Parameter(torch.zeros(_OUT_FEATURES, _IN_FEATURES // 2, dtype=torch.uint8), requires_grad=False),
            )
        elif kind == "nvfp4_buffers":
            setattr(module, f"weight{idx}", nn.Parameter(torch.zeros(1), requires_grad=False))
            setattr(
                module,
                f"weight{idx}_packed",
                torch.zeros(_OUT_FEATURES, _IN_FEATURES // 2, dtype=torch.uint8),
            )
            setattr(module, f"weight_scale{idx}", torch.ones(_OUT_FEATURES, 1))
            setattr(module, f"weight_double_scale{idx}", torch.ones(1))
        else:  # pragma: no cover
            raise AssertionError(kind)

    if kind == "nvfp4_modelopt":
        module.weight_quantizer = SimpleNamespace(
            _scale=torch.ones(_EXPERTS, _OUT_FEATURES, 1),
            _double_scale=torch.ones(_EXPERTS),
        )
    return module


@pytest.fixture(autouse=True)
def _stub_dequantizers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nvfp4_module, "dequantize_nvfp4", _zero_weight)
    monkeypatch.setattr(
        fp8_module,
        "dequant_fp8",
        lambda weight, scale_inv, out_dtype: torch.zeros_like(weight, dtype=out_dtype),
    )
    monkeypatch.setattr(oft_layers, "should_attempt_qwen3_native_fp8_gemm", lambda: False)


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["fp8_direct", "nvfp4_modelopt", "nvfp4_buffers"])
def test_quantized_grouped_fc2_returns_bias_for_probability_scaling(kind: str) -> None:
    module = _grouped_quantized_fc2(kind)
    wrapper = OFTLinear(module, nn.Identity())
    x = torch.zeros(sum(_TOKENS), _IN_FEATURES)
    tokens = torch.tensor(_TOKENS)

    output, returned_bias = wrapper(x, tokens)

    torch.testing.assert_close(output, torch.zeros(sum(_TOKENS), _OUT_FEATURES))
    assert isinstance(returned_bias, list)
    assert len(returned_bias) == _EXPERTS
    for idx, bias in enumerate(returned_bias):
        assert bias is getattr(module, f"bias{idx}")

    # Mirror TEGroupedMLP._apply_bias: the routed probability must scale the
    # returned expert bias. Injecting bias inside the GEMM would produce an
    # unscaled +bias term here instead.
    probabilities = torch.tensor([0.25, 0.5, 0.75])
    expected_chunks = []
    offset = 0
    for idx, token_count in enumerate(_TOKENS):
        probability = probabilities[offset : offset + token_count].unsqueeze(-1)
        expected_chunks.append(getattr(module, f"bias{idx}") * probability)
        offset += token_count
    actual = torch.cat(
        [
            chunk + bias * probability.unsqueeze(-1)
            for chunk, bias, probability in zip(
                torch.split(output, _TOKENS),
                returned_bias,
                torch.split(probabilities, _TOKENS),
            )
        ],
        dim=0,
    )
    torch.testing.assert_close(actual, torch.cat(expected_chunks, dim=0))


@pytest.mark.unit
def test_grouped_return_bias_rejects_incomplete_expert_bias_family() -> None:
    module = _grouped_quantized_fc2("fp8_direct")
    module.register_parameter("bias1", None)
    wrapper = OFTLinear(module, nn.Identity())

    with pytest.raises(RuntimeError, match="missing returned bias.*bias1"):
        wrapper(torch.zeros(sum(_TOKENS), _IN_FEATURES), torch.tensor(_TOKENS))
