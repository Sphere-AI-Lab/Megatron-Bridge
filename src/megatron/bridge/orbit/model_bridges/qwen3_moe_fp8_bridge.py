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

"""FP8-preserving bridge for Qwen3 MoE.

Thin composition of the architecture-independent
:class:`~megatron.bridge.orbit.conversion.fp8_preserve.FP8PreserveMixin`
with the upstream Qwen3 MoE bridge plus the orbit provider extensions
(fp32 router dtype, ``moe_layer_freq`` derivation). FP8-quantised HF
checkpoints (``quant_method: fp8``, ``weight_block_size: [128, 128]``) load
without ever allocating BF16 base weights; ``weight_scale_inv`` tensors are
merged/split alongside and registered as module buffers.

Example:
    from megatron.bridge.orbit.model_bridges.qwen3_moe_fp8_bridge import Qwen3MoEFP8Bridge

    bridge = Qwen3MoEFP8Bridge()
"""

from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge
from megatron.bridge.orbit.conversion.fp8_preserve import FP8PreserveMixin
from megatron.bridge.orbit.model_bridges.qwen3_moe_provider_ext import Qwen3MoEOrbitProviderMixin


class Qwen3MoEFP8Bridge(FP8PreserveMixin, Qwen3MoEOrbitProviderMixin, Qwen3MoEBridge):
    """Qwen3 MoE bridge that keeps FP8 weights in FP8 throughout conversion."""
