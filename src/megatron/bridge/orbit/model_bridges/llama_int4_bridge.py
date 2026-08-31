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

"""INT4-preserving bridge for dense Llama models.

Thin composition of the architecture-independent
:class:`~megatron.bridge.orbit.conversion.compressed_tensors_int4.CompressedTensorsINT4DequantMixin`
with the upstream Llama bridge. Unlike the DeepSeek path, every linear
(attention Q/K/V/O, gated MLP gate/up/down, lm_head if quantized) may be
INT4-quantized at the HF level, not just experts; layernorms and embeddings
remain BF16. Merging happens post-dequant: separate HF Q/K/V triplets become
one Megatron ``linear_qkv``, gate+up become one ``linear_fc1``. Those merged
BF16 tensors are re-quantized by ``build_int4_direct_model_state_dict``
before the dist checkpoint is written.

Verified against:
    nm-testing/Meta-Llama-3-8B-Instruct-W4A16-compressed-tensors-test
    (compressed-tensors, format=pack-quantized, symmetric, group_size=128,
    no actorder / weight_g_idx)
"""

from megatron.bridge.models.llama.llama_bridge import LlamaBridge
from megatron.bridge.orbit.conversion.compressed_tensors_int4 import CompressedTensorsINT4DequantMixin


class LlamaINT4Bridge(CompressedTensorsINT4DequantMixin, LlamaBridge):
    """Llama bridge that preserves INT4 through direct-write conversion."""
