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

import pytest

from megatron.bridge.orbit.low_precision.nvfp4 import _selective_nvfp4_quant_cfg


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
