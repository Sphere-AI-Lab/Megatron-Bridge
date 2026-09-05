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
import logging
import re
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


_ROUTED_EXPERT_WEIGHT = re.compile(r"^(?:[^.]+\.)*experts\.[0-9]+\.(?:gate_proj|up_proj|down_proj)\.weight$")
logger = logging.getLogger(__name__)


def should_quantize(key: str) -> bool:
    """Check if a weight key is an expert MLP weight that should be INT4."""
    return _ROUTED_EXPERT_WEIGHT.fullmatch(key) is not None


def _validate_paths(input_path: Path, output_path: Path) -> None:
    if not input_path.is_dir():
        raise SystemExit(f"input directory does not exist: {input_path}")
    resolved_input = input_path.resolve()
    resolved_output = output_path.resolve()
    if resolved_input == resolved_output:
        raise SystemExit("input and output directories must be different")
    if resolved_output.is_relative_to(resolved_input):
        raise SystemExit("output directory must not be inside input directory")
    if output_path.exists():
        if not output_path.is_dir() or any(output_path.iterdir()):
            raise SystemExit(f"output directory must be empty or absent: {output_path}")


def _load_config(config_path: Path) -> dict[str, object]:
    if not config_path.is_file():
        raise SystemExit(f"config.json is required for a self-describing INT4 output: {config_path}")

    try:
        config = json.loads(config_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"malformed config.json at {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise SystemExit(f"config.json must contain a JSON object: {config_path}")

    quantization_config = config.get("quantization_config")
    if quantization_config is not None and not isinstance(quantization_config, dict):
        raise SystemExit(f"quantization_config must be a JSON object in {config_path}")
    return config


def _with_int4_quantization_config(
    config: dict[str, object],
    group_size: int,
    target_modules: set[str],
) -> dict[str, object]:
    config["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "targets": sorted(target_modules),
                "weights": {
                    "type": "int",
                    "num_bits": 4,
                    "strategy": "group",
                    "group_size": group_size,
                    "symmetric": True,
                },
                "format": "pack-quantized",
            }
        },
    }
    return config


def _copy_metadata(input_path: Path, staging_path: Path, output_path: Path) -> None:
    """Copy non-shard model assets into the private staging directory."""
    resolved_output = output_path.resolve()
    for source in input_path.iterdir():
        if source.name == "config.json":
            continue
        if source.suffix == ".safetensors" or source.name == "model.safetensors.index.json":
            continue
        if source.resolve() == resolved_output:
            continue
        destination = staging_path / source.name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to BF16 HF model")
    parser.add_argument("--output", required=True, help="Output path for INT4 model")
    parser.add_argument("--group-size", type=int, default=32)
    args = parser.parse_args(argv)
    if args.group_size <= 0:
        raise SystemExit(f"group_size must be positive; received {args.group_size}")

    input_path = Path(args.input)
    output_path = Path(args.output)
    _validate_paths(input_path, output_path)

    config = _load_config(input_path / "config.json")
    shard_files = sorted(input_path.glob("model*.safetensors"))
    if not shard_files:
        raise SystemExit(f"no model*.safetensors shards found in {input_path}")

    from megatron.bridge.orbit.low_precision.int4 import quantize_to_int4

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.int4-staging-",
            dir=output_path.parent,
        )
    )
    try:
        _copy_metadata(input_path, staging_path, output_path)

        new_weight_map = {}
        total_size = 0
        total_quantized = 0
        total_kept = 0
        quantized_modules: set[str] = set()

        for shard_path in shard_files:
            logger.info("Processing %s...", shard_path.name)
            new_tensors = {}

            with safe_open(str(shard_path), framework="pt") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)

                    if should_quantize(key):
                        packed, scale, shape = quantize_to_int4(
                            tensor,
                            group_size=args.group_size,
                            scale_dtype=torch.float16,
                        )
                        base = key[: -len(".weight")]
                        new_tensors[f"{base}.weight_packed"] = packed
                        new_tensors[f"{base}.weight_scale"] = scale
                        new_tensors[f"{base}.weight_shape"] = shape
                        quantized_modules.add(base)
                        total_quantized += 1
                    else:
                        new_tensors[key] = tensor
                        total_kept += 1

            save_file(new_tensors, str(staging_path / shard_path.name))

            for key, tensor in new_tensors.items():
                new_weight_map[key] = shard_path.name
                total_size += tensor.numel() * tensor.element_size()

        if total_quantized == 0:
            raise SystemExit(f"no eligible expert weights found in {input_path}")

        index = {
            "metadata": {"total_size": total_size},
            "weight_map": new_weight_map,
        }
        (staging_path / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
        config = _with_int4_quantization_config(config, args.group_size, quantized_modules)
        (staging_path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        staging_path.replace(output_path)
    finally:
        if staging_path.exists():
            shutil.rmtree(staging_path)

    logger.info("Done. Quantized %s expert weights, kept %s as BF16.", total_quantized, total_kept)
    logger.info("Output: %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
