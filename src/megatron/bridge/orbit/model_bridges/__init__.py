# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Orbit model bridges (quantized-checkpoint variants).

These bridges are selected explicitly by the orbit conversion/finetune scripts
(instantiated directly rather than resolved through AutoBridge dispatch); this
package re-exports them all from one place.
"""

from megatron.bridge.orbit.model_bridges.deepseek_v3_int4_bridge import DeepSeekV3INT4Bridge  # noqa: F401
from megatron.bridge.orbit.model_bridges.kimi_k25_vl_nvfp4_bridge import KimiK25VLNVFP4Bridge  # noqa: F401
from megatron.bridge.orbit.model_bridges.llama_int4_bridge import LlamaINT4Bridge  # noqa: F401
from megatron.bridge.orbit.model_bridges.qwen3_int4_bridge import Qwen3INT4Bridge, Qwen3MoEINT4Bridge  # noqa: F401
from megatron.bridge.orbit.model_bridges.qwen3_moe_fp8_bridge import Qwen3MoEFP8Bridge  # noqa: F401


__all__ = [
    "DeepSeekV3INT4Bridge",
    "KimiK25VLNVFP4Bridge",
    "LlamaINT4Bridge",
    "Qwen3INT4Bridge",
    "Qwen3MoEINT4Bridge",
    "Qwen3MoEFP8Bridge",
]
