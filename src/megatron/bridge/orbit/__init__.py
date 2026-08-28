# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Orbit AI Lab additions to Megatron-Bridge.

Fork contract (since the 2026-08 upstream bump): every file that also exists
upstream is byte-identical to upstream — orbit integrates exclusively through
public extension points (subclassing, mixins, dispatch registration), never by
editing upstream files. The only exceptions are three framework-level seams in
``megatron/bridge/training/{setup,checkpointing,post_training/checkpointing}.py``,
each marked with an ``# orbit-seam(<tag>):`` comment.

Entry points:
- ``megatron.bridge.orbit.oft`` — OFT / CanonicalOFT PEFT methods (peers of
  upstream's LoRA; they plug into the same PEFT machinery via subclassing).
- ``megatron.bridge.orbit.model_bridges`` — quantized-checkpoint bridges,
  instantiated explicitly by the orbit conversion/finetune scripts.
- ``megatron.bridge.orbit.conversion.oft_export`` — OFT adapter export
  (``save_hf_oft_adapter``), the peer of ``AutoBridge.save_hf_adapter``.
- ``megatron.bridge.orbit.low_precision`` / ``quant`` — direct-load quantized
  checkpoint machinery and state-dict transforms.
- ``megatron.bridge.orbit.training`` — ModelOpt checkpoint helpers and PEFT
  parameter reports.
"""
