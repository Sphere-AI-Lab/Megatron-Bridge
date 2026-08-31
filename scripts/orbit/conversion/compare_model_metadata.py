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

"""Compare metadata from trusted HF and Megatron checkpoints.

The optional Hugging Face custom-code path executes Python supplied by the
model repository. Use this diagnostic only with checkpoint sources you trust.
"""

# ruff: noqa: D101, D103  # operational scripts: helpers here are entrypoint plumbing, not API
from __future__ import annotations

import argparse
import gc
from pathlib import Path

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.orbit.conversion.model_metadata_compare import (
    compare_config_fields,
    compare_logical_tensor_summaries,
    compare_tensor_summaries,
    extract_hf_config_fields,
    extract_megatron_config_fields,
    format_num_bytes,
    summarize_logical_tensor_collection,
    summarize_logical_torch_dist_checkpoint_metadata,
    summarize_tensor_collection,
    summarize_torch_dist_checkpoint_metadata,
)
from megatron.bridge.training.model_load_save import load_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HF checkpoint tensor metadata against a converted Megatron checkpoint.",
    )
    parser.add_argument("--hf-model-path", required=True, help="Path to the source HuggingFace model directory.")
    parser.add_argument(
        "--megatron-path",
        required=True,
        help="Path to the converted Megatron checkpoint directory.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Number of checkpoint tensors to load per batch when summarizing lazy state dicts.",
    )
    parser.set_defaults(trust_remote_code=True)
    parser.add_argument(
        "--trust-remote-code",
        dest="trust_remote_code",
        action="store_true",
        help="Execute trusted custom model code when loading HuggingFace config/state accessors (default).",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code when loading the HuggingFace source model.",
    )
    return parser.parse_args()


def resolve_megatron_checkpoint_path(megatron_path: str) -> Path:
    checkpoint_path = Path(megatron_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Megatron checkpoint path does not exist: {checkpoint_path}")

    iter_dirs = [child for child in checkpoint_path.iterdir() if child.is_dir() and child.name.startswith("iter_")]
    if not iter_dirs:
        return checkpoint_path

    def _iter_num(path: Path) -> int:
        try:
            return int(path.name.removeprefix("iter_"))
        except ValueError:
            return -1

    return max(iter_dirs, key=_iter_num)


def print_tensor_summary(title: str, summary) -> None:
    print(title)
    print(f"  tensor entries: {summary.tensor_count:,}")
    print(f"  total tensor elements: {summary.total_numel:,}")
    print(f"  total bytes: {summary.total_num_bytes:,} ({format_num_bytes(summary.total_num_bytes)})")
    print("  dtype breakdown:")
    for dtype_name, dtype_summary in summary.dtype_stats.items():
        print(
            f"    - {dtype_name}: tensors={dtype_summary.tensor_count:,}, "
            f"numel={dtype_summary.numel:,}, bytes={format_num_bytes(dtype_summary.num_bytes)}"
        )


def print_config_summary(title: str, fields: dict[str, object]) -> None:
    print(title)
    if not fields:
        print("  <no comparable config fields found>")
        return
    for key, value in fields.items():
        print(f"  {key}: {value}")


def print_logical_summary(title: str, summary) -> None:
    print(title)
    print(f"  logical tensors: {summary.tensor_count:,}")
    print(f"  total logical elements: {summary.total_numel:,}")
    print("  category breakdown:")
    for category, category_summary in summary.category_stats.items():
        print(f"    - {category}: tensors={category_summary.tensor_count:,}, numel={category_summary.numel:,}")


def main() -> int:
    args = parse_args()
    if args.trust_remote_code:
        print("WARNING: trust_remote_code is enabled; only use a trusted HuggingFace source.", flush=True)
        # Checkpoints of trust_remote_code models embed config targets under
        # transformers_modules.*; allow instantiating them on the Megatron
        # side too, mirroring AutoBridge's own registration.
        from megatron.bridge.utils.instantiate_utils import register_allowed_target_prefix

        register_allowed_target_prefix("transformers_modules.")
    checkpoint_path = resolve_megatron_checkpoint_path(args.megatron_path)

    print(f"HuggingFace source: {args.hf_model_path}", flush=True)
    print(f"Megatron checkpoint: {checkpoint_path}", flush=True)

    print("\nSummarizing HuggingFace checkpoint tensors...", flush=True)
    auto_bridge = AutoBridge.from_hf_pretrained(
        args.hf_model_path,
        trust_remote_code=args.trust_remote_code,
    )
    hf_summary = summarize_tensor_collection(auto_bridge.hf_pretrained.state, chunk_size=args.chunk_size)
    hf_logical_summary = summarize_logical_tensor_collection(
        auto_bridge.hf_pretrained.state, chunk_size=args.chunk_size
    )
    hf_fields = extract_hf_config_fields(auto_bridge.hf_pretrained.config)
    del auto_bridge
    gc.collect()

    print("\nReading and summarizing Megatron checkpoint metadata...", flush=True)
    megatron_cfg, _ = load_model_config(str(checkpoint_path))
    megatron_summary = summarize_torch_dist_checkpoint_metadata(checkpoint_path)
    megatron_logical_summary = summarize_logical_torch_dist_checkpoint_metadata(checkpoint_path)
    megatron_fields = extract_megatron_config_fields(megatron_cfg)
    gc.collect()

    print()
    print_tensor_summary("HF checkpoint summary:", hf_summary)
    print()
    print_tensor_summary("Megatron checkpoint summary:", megatron_summary)
    print()
    print_logical_summary("HF logical tensor summary:", hf_logical_summary)
    print()
    print_logical_summary("Megatron logical tensor summary:", megatron_logical_summary)
    print()
    print_config_summary("HF comparable config fields:", hf_fields)
    print()
    print_config_summary("Megatron comparable config fields:", megatron_fields)

    raw_storage_mismatches = compare_tensor_summaries(hf_summary, megatron_summary)
    mismatches = []
    mismatches.extend(compare_logical_tensor_summaries(hf_logical_summary, megatron_logical_summary))
    mismatches.extend(compare_config_fields(hf_fields, megatron_fields))

    if hf_summary.tensor_count != megatron_summary.tensor_count:
        print(
            "\nNote: tensor entry counts differ. That can be expected when Megatron fuses "
            "layouts like QKV or SwiGLU compared with the HuggingFace checkpoint.",
            flush=True,
        )

    if raw_storage_mismatches:
        print(
            "\nRaw storage differences detected. These can be expected when quantized formats "
            "use different checkpoint dtypes/layouts between HuggingFace and Megatron.",
            flush=True,
        )
        for mismatch in raw_storage_mismatches:
            print(f"  - {mismatch}", flush=True)

    if mismatches:
        print("\nLogical metadata comparison FAILED:", flush=True)
        for mismatch in mismatches:
            print(f"  - {mismatch}", flush=True)
        return 1

    print("\nLogical metadata comparison PASSED.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
