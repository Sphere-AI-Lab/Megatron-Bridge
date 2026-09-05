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

import megatron.core.extensions.transformer_engine as te_ext
import megatron.core.tensor_parallel.layers as tp_layers
import pytest
import torch

from megatron.bridge.orbit.low_precision.common import patch_meta_init_for_te_modules


pytestmark = pytest.mark.unit


def test_meta_init_patch_only_swallows_not_implemented_for_meta_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unsupported_init(weight: torch.Tensor, *args, **kwargs):
        raise NotImplementedError(f"unsupported on {weight.device.type}")

    monkeypatch.setattr(tp_layers, "_initialize_affine_weight_cpu", unsupported_init)
    monkeypatch.setattr(te_ext, "_initialize_affine_weight_cpu", unsupported_init)

    patch_meta_init_for_te_modules()
    patched_init = tp_layers._initialize_affine_weight_cpu

    assert te_ext._initialize_affine_weight_cpu is patched_init
    assert patched_init(torch.empty((2, 2), device="meta")) is None
    with torch.device("meta"):
        assert patched_init(torch.empty((2, 2), device="cpu")) is None
    with pytest.raises(NotImplementedError, match="unsupported on cpu"):
        patched_init(torch.empty((2, 2)))
