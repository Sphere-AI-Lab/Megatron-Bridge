# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared PEFT parameter-name predicates.

CanonicalOFT split adapters use names like ``adapter_q.oft_r`` and
``adapter_gate.oft_r`` instead of the legacy single ``adapter.oft_r`` form.
Recompute, checkpoint filtering, and adapter-only mode all need to recognize
both shapes — this module is the single source of truth.
"""

from __future__ import annotations


CANONICAL_OFT_SLICE_NAMES = ("q", "k", "v", "gate", "up")

_CANONICAL_SPLIT_ADAPTER_TOKENS = tuple(f".adapter_{s}." for s in CANONICAL_OFT_SLICE_NAMES)

_DSV4_OFT_SUFFIXES = ("w1_oft_r", "w2_oft_r", "w3_oft_r")


def _is_peft_adapter_lowered(lowered: str) -> bool:
    return (
        "lora_" in lowered
        or ".adapter." in lowered
        or any(token in lowered for token in _CANONICAL_SPLIT_ADAPTER_TOKENS)
        or lowered.endswith(".oft_r")
        or lowered.endswith(_DSV4_OFT_SUFFIXES)
        or lowered.endswith(".adapters")
    )


def is_peft_adapter_param_name(name: str) -> bool:
    """Return True for trainable adapter parameters across LoRA and OFT variants."""
    return _is_peft_adapter_lowered(name.lower())


def is_trainable_base_param_name(name: str) -> bool:
    """Return True only for real base-model params that should block adapter-only mode."""
    lowered = name.lower()
    return ".to_wrap." not in lowered and not _is_peft_adapter_lowered(lowered)
