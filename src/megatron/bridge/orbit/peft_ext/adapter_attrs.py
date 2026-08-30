# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""OFT-specific adapter attribute extraction.

Upstream's :func:`get_adapter_attributes_from_linear` is written for LoRA: for
``TELayerNormColumnParallelLinear`` it flips ``return_layernorm_output`` (and
the SP-gather variant) on the base module so the LoRA branch can consume the
layernorm output. OFT rotates the linear's own input instead, so those side
effects are wrong for it. This wrapper delegates to upstream and then reverts
the LoRA-only side effects, keeping the fork free of in-place edits to
``megatron.bridge.peft.utils``.
"""

import dataclasses

from torch import nn

from megatron.bridge.peft.utils import AdapterAttributes, get_adapter_attributes_from_linear


try:
    from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear

    HAVE_TE = True
except ImportError:
    TELayerNormColumnParallelLinear = None
    HAVE_TE = False


def get_oft_adapter_attributes_from_linear(m: nn.Module, is_expert: bool = False) -> AdapterAttributes:
    """Adapter attributes for OFT: upstream's result minus the LoRA layernorm-output plumbing."""
    attrs = get_adapter_attributes_from_linear(m, is_expert=is_expert)

    if HAVE_TE and isinstance(m, TELayerNormColumnParallelLinear):
        # OFT must not consume the layernorm output; give the fused module its
        # default fprop back and recompute the SP-comm flag the way upstream
        # does outside the LoRA-specific branch.
        m.return_layernorm_output = False
        if hasattr(m, "return_layernorm_output_gathered"):
            m.return_layernorm_output_gathered = False
        disable_sequence_parallel_comm = not m.config.sequence_parallel or attrs.disable_tensor_parallel_comm
        attrs = dataclasses.replace(attrs, disable_sequence_parallel_comm=disable_sequence_parallel_comm)

    return attrs
