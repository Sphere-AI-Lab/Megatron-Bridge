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

"""Unit tests for scale-reusing INT4 re-quantization."""

import pytest
import torch

from megatron.bridge.models.kimi_vl.utils import dequantize_int4, quantize_to_int4
from megatron.bridge.orbit.low_precision.int4 import requantize_int4_with_scales


pytestmark = pytest.mark.unit


def _triplet_with_full_range(out_f=16, in_f=256, group=32):
    """Build a triplet whose groups use the full signed range, including -8."""
    torch.manual_seed(0)
    q = torch.randint(-7, 8, (out_f, in_f), dtype=torch.float32)
    q[:, ::group] = -8  # every group's first element is the valid extreme
    scale = (torch.rand(out_f, in_f // group) * 0.05 + 1e-3).to(torch.bfloat16)
    w_q = (q + 8).to(torch.uint8).view(out_f, in_f // 8, 8).to(torch.int32)
    packed = torch.zeros(out_f, in_f // 8, dtype=torch.int32)
    for i in range(8):
        packed |= (w_q[:, :, i] & 0xF) << (i * 4)
    shape = torch.tensor([out_f, in_f], dtype=torch.int32)
    return packed, scale, shape, q


class TestScaleReuseRoundTrip:
    def test_bitwise_roundtrip_with_minus_eight_groups(self):
        packed, scale, shape, _ = _triplet_with_full_range()
        deq = dequantize_int4(packed, scale, shape)
        p2, s2, sh2 = requantize_int4_with_scales(deq, scale)
        assert torch.equal(p2, packed)
        assert torch.equal(s2, scale)
        assert torch.equal(sh2.to(shape.dtype), shape)

    def test_recomputed_scales_do_regrid_minus_eight_groups(self):
        """The bug this fix addresses: amax/7 recomputation cannot emit -8."""
        packed, scale, shape, _ = _triplet_with_full_range()
        deq = dequantize_int4(packed, scale, shape)
        p2, s2, _ = quantize_to_int4(deq, group_size=32, scale_dtype=scale.dtype)
        assert not torch.equal(p2, packed)
        assert not torch.equal(s2, scale)

    def test_shape_validation(self):
        w = torch.randn(8, 64, dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="does not tile"):
            requantize_int4_with_scales(w, torch.ones(8, 3, dtype=torch.bfloat16))
        with pytest.raises(ValueError, match="does not tile"):
            requantize_int4_with_scales(w, torch.ones(4, 2, dtype=torch.bfloat16))

    def test_pack_width_validation_is_not_an_optimized_away_assert(self):
        weight = torch.randn(2, 10, dtype=torch.bfloat16)
        scale = torch.ones(2, 2, dtype=torch.bfloat16)

        with pytest.raises(ValueError, match="in_features must be divisible by 8, got 10"):
            requantize_int4_with_scales(weight, scale)

    def test_group128_layout(self):
        packed, scale, shape, _ = _triplet_with_full_range(out_f=8, in_f=512, group=128)
        deq = dequantize_int4(packed, scale, shape)
        p2, s2, _ = requantize_int4_with_scales(deq, scale)
        assert torch.equal(p2, packed)
        assert torch.equal(s2, scale)
