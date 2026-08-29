#!/usr/bin/env python3
"""Quantize a BF16 HF checkpoint's expert weights to Kimi-K2.5 native INT4 format.

Takes a BF16 HuggingFace model and quantizes all expert MLP weights to INT4,
producing a new checkpoint with weight_packed + weight_scale + weight_shape
tensors (same format as Kimi-K2.5).

Non-expert weights (attention, norms, embeddings, shared experts, dense layers)
stay in BF16.

Usage:
    python scripts/orbit/conversion/quantize_to_int4.py \
        --input /path/to/Moonlight-16B-A3B \
        --output /path/to/Moonlight-16B-A3B-INT4
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from megatron.bridge.orbit.low_precision.int4 import quantize_to_int4


def should_quantize(key: str) -> bool:
    """Check if a weight key is an expert MLP weight that should be INT4."""
    if not key.endswith(".weight"):
        return False
    if "experts." not in key:
        return False
    if "shared_expert" in key:
        return False
    # Expert gate_proj, up_proj, down_proj
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        if proj in key:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to BF16 HF model")
    parser.add_argument("--output", required=True, help="Output path for INT4 model")
    parser.add_argument("--group-size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Copy non-safetensor files (config, tokenizer, etc.)
    for f in Path(args.input).iterdir():
        if f.suffix != ".safetensors" and f.name != "model.safetensors.index.json":
            dst = Path(args.output) / f.name
            if not dst.exists():
                if f.is_dir():
                    shutil.copytree(f, dst)
                else:
                    shutil.copy2(f, dst)

    # Process each safetensors shard
    shard_files = sorted(Path(args.input).glob("model*.safetensors"))
    new_weight_map = {}
    total_quantized = 0
    total_kept = 0

    for shard_path in shard_files:
        print(f"Processing {shard_path.name}...")
        new_tensors = {}

        with safe_open(str(shard_path), framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                if should_quantize(key):
                    packed, scale, shape = quantize_to_int4(tensor, group_size=args.group_size)
                    base = key[:-len(".weight")]
                    new_tensors[f"{base}.weight_packed"] = packed
                    new_tensors[f"{base}.weight_scale"] = scale
                    new_tensors[f"{base}.weight_shape"] = shape
                    total_quantized += 1
                else:
                    new_tensors[key] = tensor
                    total_kept += 1

        # Save new shard
        out_path = Path(args.output) / shard_path.name
        save_file(new_tensors, str(out_path))

        for k in new_tensors:
            new_weight_map[k] = shard_path.name

    # Write new index
    index = {
        "metadata": {"total_size": sum(
            t.numel() * t.element_size()
            for shard in shard_files
            for t in [torch.zeros(1)]  # placeholder
        )},
        "weight_map": new_weight_map,
    }
    with open(Path(args.output) / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nDone. Quantized {total_quantized} expert weights, kept {total_kept} as BF16.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
