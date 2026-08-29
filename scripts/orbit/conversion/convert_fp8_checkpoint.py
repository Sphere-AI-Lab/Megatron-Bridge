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
Convert a HuggingFace FP8 checkpoint to Megatron format, preserving FP8 weights.

The standard ``AutoBridge.import_ckpt`` dequantises FP8 → BF16 during conversion
(because the bridge casts HF weights to the Megatron parameter dtype).  This
script uses ``Qwen3MoEFP8Bridge`` which skips that cast, so FP8 weights flow
through QKV merge and TP split unchanged.  Block-wise ``weight_scale_inv``
tensors are transformed in parallel and stored as module buffers.

Usage:
    python scripts/orbit/conversion/convert_fp8_checkpoint.py \\
        --hf-model-path /path/to/Qwen3-30B-A3B-FP8 \\
        --megatron-path ./checkpoints/Qwen3-30B-A3B-FP8

Then use the resulting checkpoint with finetune_qoft.py:
    torchrun --nproc_per_node=4 scripts/orbit/models/qwen3_moe/finetune_qoft.py \\
        --pretrained-checkpoint ./checkpoints/Qwen3-30B-A3B-FP8 \\
        --tp 2 --ep 2
"""

import argparse
import sys
from functools import partial

from megatron.bridge import AutoBridge
from megatron.bridge.orbit.conversion.fp8_preserve import fp8_bridge_for
from megatron.bridge.orbit.model_bridges.qwen3_moe_fp8_bridge import Qwen3MoEFP8Bridge


def main():
    """Convert an HF FP8 checkpoint to Megatron format, preserving FP8."""
    parser = argparse.ArgumentParser(
        description="Convert HF FP8 checkpoint to Megatron format (preserving FP8)",
    )
    parser.add_argument("--hf-model-path", required=True, help="Path to HF FP8 model directory")
    parser.add_argument("--megatron-path", required=True, help="Output Megatron checkpoint directory")
    args = parser.parse_args()

    print(f"Converting: {args.hf_model_path} -> {args.megatron_path}")
    print("Preserving FP8 weights through conversion (no BF16 intermediate)")

    # The standard AutoBridge.import_ckpt flow:
    #   1. AutoBridge.from_hf_pretrained(hf_model)
    #   2. bridge.to_megatron_model(wrap_with_ddp=False)
    #   3. bridge.save_megatron_model(model, path)
    #
    # Problem: step 2 calls provide_distributed_model() → get_model() →
    # Float16Module(module.bfloat16()) which casts ALL parameters to BF16,
    # destroying our FP8 weights.
    #
    # Fix: pass mixed_precision_wrapper=None to skip Float16Module entirely
    # during conversion.  FP8 weights survive into the checkpoint.

    auto_bridge = AutoBridge.from_hf_pretrained(args.hf_model_path)

    provider = auto_bridge.to_megatron_provider(load_weights=False)
    provider.perform_initialization = False

    architecture = getattr(auto_bridge._causal_lm_architecture, "__name__", str(auto_bridge._causal_lm_architecture))
    if architecture == "Qwen3MoeForCausalLM":
        # Named class keeps the orbit Qwen3-MoE provider extensions.
        fp8_bridge = Qwen3MoEFP8Bridge()
    else:
        # Any other registered architecture: compose the generic FP8 mixin
        # with its registered bridge (same load path the named class uses).
        print(f"No named FP8 bridge for {architecture}; composing the generic FP8 adapter")
        fp8_bridge = fp8_bridge_for(auto_bridge)
    provider.register_pre_wrap_hook(partial(fp8_bridge.load_weights_hf_to_megatron, auto_bridge.hf_pretrained))

    if hasattr(provider, "finalize"):
        provider.finalize()
    megatron_model = provider.provide_distributed_model(
        wrap_with_ddp=False,
        use_cpu_initialization=True,
        mixed_precision_wrapper=None,  # Skip Float16Module — preserves FP8 dtypes
    )

    # Save as Megatron checkpoint (torch.save preserves FP8 dtypes).
    auto_bridge.save_megatron_model(
        megatron_model,
        args.megatron_path,
        hf_tokenizer_path=args.hf_model_path,
        low_memory_save=True,
    )

    print(f"Done. FP8 Megatron checkpoint saved to: {args.megatron_path}")


if __name__ == "__main__":
    sys.exit(main())
