#!/usr/bin/env python3
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

# ruff: noqa: D101, D103  # operational scripts: helpers here are entrypoint plumbing, not API

import argparse
import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


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


def _prepare_output_dir(input_path: Path, output_path: Path) -> None:
    if not input_path.is_dir():
        raise SystemExit(f"input directory does not exist: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise SystemExit("input and output directories must be different")
    if output_path.exists():
        if not output_path.is_dir() or any(output_path.iterdir()):
            raise SystemExit(f"output directory must be empty or absent: {output_path}")
    else:
        output_path.mkdir(parents=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to BF16 HF model")
    parser.add_argument("--output", required=True, help="Output path for INT4 model")
    parser.add_argument("--group-size", type=int, default=32)
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    _prepare_output_dir(input_path, output_path)

    from megatron.bridge.orbit.low_precision.int4 import quantize_to_int4

    # Copy non-safetensor files (config, tokenizer, etc.)
    for f in input_path.iterdir():
        if f.suffix != ".safetensors" and f.name != "model.safetensors.index.json":
            dst = output_path / f.name
            if f.is_dir():
                shutil.copytree(f, dst)
            else:
                shutil.copy2(f, dst)

    # Process each safetensors shard
    shard_files = sorted(input_path.glob("model*.safetensors"))
    if not shard_files:
        raise SystemExit(f"no model*.safetensors shards found in {input_path}")
    new_weight_map = {}
    total_size = 0
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
                    base = key[: -len(".weight")]
                    new_tensors[f"{base}.weight_packed"] = packed
                    new_tensors[f"{base}.weight_scale"] = scale
                    new_tensors[f"{base}.weight_shape"] = shape
                    total_quantized += 1
                else:
                    new_tensors[key] = tensor
                    total_kept += 1

        # Save new shard
        out_path = output_path / shard_path.name
        save_file(new_tensors, str(out_path))

        for key, tensor in new_tensors.items():
            new_weight_map[key] = shard_path.name
            total_size += tensor.numel() * tensor.element_size()

    # Write new index
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": new_weight_map,
    }
    with open(output_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nDone. Quantized {total_quantized} expert weights, kept {total_kept} as BF16.")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
