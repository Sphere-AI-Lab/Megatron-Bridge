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

"""Unit tests for _selective_nvfp4_quant_cfg across both ModelOpt schemas."""

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.orbit.low_precision.nvfp4 import (
    _selective_nvfp4_quant_cfg,
    build_fused_nvfp4_weight_entries,
)


pytestmark = pytest.mark.unit


NVFP4 = {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}


class TestDictSchema:
    def test_globals_disabled_and_modules_enabled(self):
        base = {
            "*weight_quantizer": dict(NVFP4),
            "*input_quantizer": dict(NVFP4),
            "*output_quantizer": {"enable": False},
        }
        out = _selective_nvfp4_quant_cfg(base, ["decoder.layers.0.mlp.fc1", "decoder.layers.1.mlp.fc1"])
        assert out["*weight_quantizer"] == {"enable": False}
        assert out["*input_quantizer"] == {"enable": False}
        assert out["decoder.layers.0.mlp.fc1.weight_quantizer"] == NVFP4
        assert out["decoder.layers.1.mlp.fc1.input_quantizer"] == NVFP4
        assert out["*output_quantizer"] == {"enable": False}
        assert base["*weight_quantizer"] == NVFP4  # input untouched


class TestListSchema:
    BASE = [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*weight_quantizer", "cfg": dict(NVFP4)},
        {"quantizer_name": "*input_quantizer", "cfg": dict(NVFP4)},
        {"quantizer_name": "*lm_head*", "enable": False},
    ]

    def test_globals_replaced_in_place_by_module_entries(self):
        out = _selective_nvfp4_quant_cfg(self.BASE, ["m.b", "m.a"])
        names = [e["quantizer_name"] for e in out]
        # disable-all stays first, exclusions stay last, module enables sit
        # where the global enables sat, sorted.
        assert names == [
            "*",
            "m.a.weight_quantizer",
            "m.a.input_quantizer",
            "m.b.weight_quantizer",
            "m.b.input_quantizer",
            "*lm_head*",
        ]
        assert out[1]["cfg"] == NVFP4
        assert all(e.get("quantizer_name") not in ("*weight_quantizer", "*input_quantizer") for e in out)
        assert [e["quantizer_name"] for e in self.BASE] == ["*", "*weight_quantizer", "*input_quantizer", "*lm_head*"]

    def test_no_global_enables_appends_at_end(self):
        base = [{"quantizer_name": "*", "enable": False}]
        out = _selective_nvfp4_quant_cfg(base, ["m"])
        assert [e["quantizer_name"] for e in out] == ["*", "m.weight_quantizer", "m.input_quantizer"]
        assert out[1]["cfg"] == {"enable": True}

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported ModelOpt quant_cfg type"):
            _selective_nvfp4_quant_cfg("not-a-config", ["m"])


@pytest.mark.unit
def test_fused_qkv_normalizes_distinct_weight_double_scales() -> None:
    """Independent Q/K/V quantizers may not share ``weight_scale_2``."""
    config = SimpleNamespace(
        num_attention_heads=2,
        num_query_groups=1,
        kv_channels=8,
        hidden_size=16,
        attention_output_gate=False,
    )
    mapping = SimpleNamespace(_get_config=lambda module: config)

    def bundle(rows: int, double_scale: float) -> dict[str, torch.Tensor]:
        return {
            "weight": torch.zeros((rows, 8), dtype=torch.uint8),
            "weight_scale": torch.ones((rows, 1), dtype=torch.float8_e4m3fn),
            "weight_scale_2": torch.tensor(double_scale),
            "input_scale": torch.tensor(1.0),
        }

    entries, state = build_fused_nvfp4_weight_entries(
        mapping,
        "decoder.layers.0.self_attention.linear_qkv.weight",
        {
            "q": bundle(16, 1.0),
            "k": bundle(8, 2.0),
            "v": bundle(8, 4.0),
        },
        object(),
    )

    shared = state["weight_scale_2"]
    effective_scale = entries["decoder.layers.0.self_attention.linear_qkv.weight_quantizer._scale"].float() * shared
    expected = torch.tensor([[1.0]] * 16 + [[2.0]] * 8 + [[4.0]] * 8)
    torch.testing.assert_close(effective_scale, expected)


@pytest.mark.unit
def test_fused_qkv_normalizes_near_double_scales_back_to_canonical_fp8() -> None:
    """Near double scales stay schema-valid with only FP8-rounding error."""
    config = SimpleNamespace(
        num_attention_heads=2,
        num_query_groups=1,
        kv_channels=8,
        hidden_size=16,
        attention_output_gate=False,
    )
    mapping = SimpleNamespace(_get_config=lambda module: config)

    def bundle(rows: int, double_scale: float) -> dict[str, torch.Tensor]:
        return {
            "weight": torch.zeros((rows, 8), dtype=torch.uint8),
            "weight_scale": torch.ones((rows, 1), dtype=torch.float8_e4m3fn),
            "weight_scale_2": torch.tensor(double_scale),
            "input_scale": torch.tensor(1.0),
        }

    entries, state = build_fused_nvfp4_weight_entries(
        mapping,
        "decoder.layers.0.self_attention.linear_qkv.weight",
        {
            "q": bundle(16, 1.0),
            "k": bundle(8, 1.000005),
            "v": bundle(8, 1.0),
        },
        object(),
    )

    shared = state["weight_scale_2"]
    block_scale = entries["decoder.layers.0.self_attention.linear_qkv.weight_quantizer._scale"]
    effective_scale = block_scale.float() * shared
    expected = torch.tensor([[1.0]] * 16 + [[1.000005]] * 8 + [[1.0]] * 8)
    assert block_scale.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(effective_scale, expected, atol=1e-5, rtol=0)


@pytest.mark.unit
def test_fused_qkv_rejects_shared_scale_normalization_that_exceeds_fp8_rounding_bound() -> None:
    """Fixed nonzero packed values must not be published with corrupted effective scales."""
    config = SimpleNamespace(
        num_attention_heads=2,
        num_query_groups=1,
        kv_channels=8,
        hidden_size=16,
        attention_output_gate=False,
    )
    mapping = SimpleNamespace(_get_config=lambda module: config)

    def bundle(rows: int, double_scale: float) -> dict[str, torch.Tensor]:
        return {
            # 0x11 decodes to two nonzero E2M1 values, so scale corruption is observable.
            "weight": torch.full((rows, 8), 0x11, dtype=torch.uint8),
            "weight_scale": torch.ones((rows, 1), dtype=torch.float8_e4m3fn),
            "weight_scale_2": torch.tensor(double_scale),
            "input_scale": torch.tensor(1.0),
        }

    with pytest.raises(ValueError, match=r"cannot preserve effective block scales.*6\.25%"):
        build_fused_nvfp4_weight_entries(
            mapping,
            "decoder.layers.0.self_attention.linear_qkv.weight",
            {
                "q": bundle(16, 1.0),
                "k": bundle(8, 1e-8),
                "v": bundle(8, 1.0),
            },
            object(),
        )
