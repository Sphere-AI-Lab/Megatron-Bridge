#!/usr/bin/env python3
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
Convert a HuggingFace INT4 checkpoint to Megatron format, preserving INT4.

Supports:
    - DeepSeek-V3 / Moonlight (DeepseekV3ForCausalLM) with INT4 expert weights
    - Kimi-K2.5 (KimiK25ForConditionalGeneration) with INT4 expert weights

Expert weights are dequantised to BF16 for QKV merge / TP split, then
re-quantised to INT4 before saving.  Non-expert weights stay in BF16.

Usage:
    python scripts/orbit/conversion/convert_int4_checkpoint.py \\
        --hf-model-path /path/to/Moonlight-16B-A3B-INT4 \\
        --megatron-path ./checkpoints/Moonlight-16B-A3B-INT4

    python scripts/orbit/conversion/convert_int4_checkpoint.py \\
        --hf-model-path /path/to/Kimi-K2.5 \\
        --megatron-path ./checkpoints/Kimi-K2.5-INT4
"""

import argparse
import sys
from functools import partial

from megatron.bridge import AutoBridge
from megatron.bridge.orbit.model_bridges.deepseek_v3_int4_bridge import DeepSeekV3INT4Bridge

# Register Kimi-K2.5 VL bridge (only if the model uses that architecture).
try:
    from megatron.bridge.models.kimi_vl.kimi_k25_vl_bridge import KimiK25VLBridge  # noqa: F401
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Convert HF INT4 checkpoint to Megatron format (preserving INT4)",
    )
    parser.add_argument("--hf-model-path", required=True, help="Path to HF INT4 model directory")
    parser.add_argument("--megatron-path", required=True, help="Output Megatron checkpoint directory")
    args = parser.parse_args()

    print(f"Converting: {args.hf_model_path} -> {args.megatron_path}")

    auto_bridge = AutoBridge.from_hf_pretrained(args.hf_model_path, trust_remote_code=True)

    # Use AutoBridge for model config, but swap in INT4 bridge for weight loading.
    # This works for any architecture (DeepSeek-V3, Kimi-K2.5, etc.)
    provider = auto_bridge.to_megatron_provider(load_weights=False)
    provider.perform_initialization = False

    int4_bridge = DeepSeekV3INT4Bridge()
    provider.register_pre_wrap_hook(
        partial(int4_bridge.load_weights_hf_to_megatron, auto_bridge.hf_pretrained)
    )

    if hasattr(provider, "finalize"):
        provider.finalize()
    megatron_model = provider.provide_distributed_model(
        wrap_with_ddp=False,
        use_cpu_initialization=True,
        mixed_precision_wrapper=None,  # Skip Float16Module — preserves INT4 buffers
    )

    auto_bridge.save_megatron_model(
        megatron_model,
        args.megatron_path,
        hf_tokenizer_path=args.hf_model_path,
        low_memory_save=True,
    )

    print(f"Done. INT4 Megatron checkpoint saved to: {args.megatron_path}")


if __name__ == "__main__":
    sys.exit(main())
