# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Orbit extensions for quantized training in Megatron-Bridge.

Most Orbit behavior integrates through public extension points such as
subclassing, mixins, and explicit bridge composition. Four narrow framework
seams support ModelOpt checkpoint restoration and pre-wrap hook ordering; each
is marked with an ``# orbit-seam(<tag>):`` comment.

Entry points:
- ``megatron.bridge.orbit.oft`` — OFT / CanonicalOFT PEFT methods (peers of
  upstream's LoRA; they plug into the same PEFT machinery via subclassing).
- ``megatron.bridge.orbit.model_bridges`` — quantized-checkpoint bridges,
  instantiated explicitly by the orbit conversion/finetune scripts.
- ``megatron.bridge.orbit.conversion.oft_export`` — OFT adapter export
  (``save_hf_oft_adapter``), the peer of ``AutoBridge.save_hf_adapter``.
- ``megatron.bridge.orbit.low_precision`` / ``quant`` — direct-load quantized
  checkpoint machinery and state-dict transforms.
- ``megatron.bridge.orbit.training`` — ModelOpt checkpoint helpers.
"""
