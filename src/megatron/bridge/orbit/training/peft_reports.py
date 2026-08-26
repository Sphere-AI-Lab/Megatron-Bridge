# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""PEFT parameter-partition TSV reports (orbit fork).

Extracted from ``megatron.bridge.training.setup``; the upstream module keeps a
rank-0 hook in ``_apply_peft_transformation``. See
``megatron/bridge/orbit/UPSTREAM_SEAMS.md``.
"""

import os
from typing import Any

from megatron.core.transformer.module import MegatronModule


def _collect_parameter_partition_entries(model: MegatronModule) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect trainable and frozen parameter metadata for reporting."""
    trainable_entries: list[dict[str, Any]] = []
    frozen_entries: list[dict[str, Any]] = []

    for name, param in model.named_parameters():
        entry = {
            "name": name,
            "numel": param.numel(),
            "dtype": str(param.dtype).replace("torch.", ""),
            "shape": tuple(param.shape),
        }
        if param.requires_grad:
            trainable_entries.append(entry)
        else:
            frozen_entries.append(entry)

    trainable_entries.sort(key=lambda entry: entry["name"])
    frozen_entries.sort(key=lambda entry: entry["name"])
    return trainable_entries, frozen_entries


def _write_parameter_partition_report(entries: list[dict[str, Any]], report_path: os.PathLike[str] | str) -> None:
    """Write a TSV report describing a parameter partition."""
    report_dir = os.path.dirname(os.fspath(report_path))
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("name\tnumel\tdtype\tshape\n")
        for entry in entries:
            report_file.write(
                f"{entry['name']}\t{entry['numel']}\t{entry['dtype']}\t{entry['shape']}\n"
            )


def _write_peft_parameter_reports(
    model: MegatronModule,
    report_dir: str = "/tmp",
) -> dict[str, str]:
    """Write rank-0 PEFT parameter reports to disk."""
    trainable_entries, frozen_entries = _collect_parameter_partition_entries(model)

    trainable_report = os.path.join(report_dir, "megatron_bridge_peft_trainable_params.txt")
    frozen_report = os.path.join(report_dir, "megatron_bridge_peft_frozen_params.txt")
    summary_report = os.path.join(report_dir, "megatron_bridge_peft_summary.txt")

    _write_parameter_partition_report(trainable_entries, trainable_report)
    _write_parameter_partition_report(frozen_entries, frozen_report)

    total_params = sum(entry["numel"] for entry in trainable_entries) + sum(
        entry["numel"] for entry in frozen_entries
    )
    trainable_params = sum(entry["numel"] for entry in trainable_entries)
    frozen_params = sum(entry["numel"] for entry in frozen_entries)
    with open(summary_report, "w", encoding="utf-8") as summary_file:
        summary_file.write(f"total_parameters\t{total_params}\n")
        summary_file.write(f"trainable_parameters\t{trainable_params}\n")
        summary_file.write(f"frozen_parameters\t{frozen_params}\n")
        summary_file.write(
            f"trainable_percentage\t{(100 * trainable_params / total_params) if total_params else 0.0:.2f}\n"
        )
        summary_file.write(
            f"frozen_percentage\t{(100 * frozen_params / total_params) if total_params else 0.0:.2f}\n"
        )

    return {
        "trainable": trainable_report,
        "frozen": frozen_report,
        "summary": summary_report,
    }
