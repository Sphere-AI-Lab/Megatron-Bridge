# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
OFT-compatible Transformer Engine layer wrappers.

This package contains modified copies of TE's internal autograd functions
with OFT rotation hooks inserted between LayerNorm and GEMM.

Architecture:
    TE's _LayerNormLinear autograd Function:
        1. apply_normalization(x, ln_weight, ...) → ln_out
        2. all-gather ln_out (if SP)
        3. general_gemm(weight, ln_out) → out

    Our _OFTLayerNormLinear autograd Function:
        1. apply_normalization(x, ln_weight, ...) → ln_out
        1.5. adapter_fn(ln_out) → ln_out  ← OFT rotation inserted here
        2. all-gather ln_out (if SP)
        3. general_gemm(weight, ln_out) → out

    This preserves FP8, Userbuffers, SP all-gather, etc.
    The backward is reused from TE's original (OFT rotation grads
    flow through PyTorch autograd automatically).
"""

from megatron.bridge.orbit.oft.te_oft.te_oft_layernorm_linear import TEOFTLayerNormColumnParallelLinear


__all__ = ["TEOFTLayerNormColumnParallelLinear"]
