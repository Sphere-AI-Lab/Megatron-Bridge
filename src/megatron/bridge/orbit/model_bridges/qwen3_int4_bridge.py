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

"""INT4-preserving bridges for Qwen3 models.

Thin compositions of the architecture-independent
:class:`~megatron.bridge.orbit.conversion.compressed_tensors_int4.CompressedTensorsINT4Mixin`
with the upstream Qwen3 bridges. The MoE variant also carries the orbit
provider extensions (fp32 router dtype, ``moe_layer_freq`` derivation).
"""

from megatron.bridge.models.qwen.qwen3_bridge import Qwen3Bridge
from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge
from megatron.bridge.orbit.conversion.compressed_tensors_int4 import CompressedTensorsINT4Mixin
from megatron.bridge.orbit.model_bridges.qwen3_moe_provider_ext import Qwen3MoEOrbitProviderMixin


class Qwen3INT4Bridge(CompressedTensorsINT4Mixin, Qwen3Bridge):
    """Dense Qwen3 bridge that preserves compressed-tensors INT4 direct writes."""


class Qwen3MoEINT4Bridge(CompressedTensorsINT4Mixin, Qwen3MoEOrbitProviderMixin, Qwen3MoEBridge):
    """Qwen3 MoE bridge that preserves compressed-tensors INT4 direct writes."""
