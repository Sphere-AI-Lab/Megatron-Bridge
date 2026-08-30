# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""OFT-aware variant of the PEFT+recompute grad fix.

Upstream's :func:`megatron.bridge.peft.recompute.maybe_enable_recompute_inputs_grad`
detects "adapter-only training" with LoRA-shaped name predicates (``.adapter.``),
so OFT runs — whose trainable params are named ``.adapter_q.`` / ``.oft_r`` /
etc. — are never patched. This is a port of that function with the predicates
widened via :mod:`megatron.bridge.orbit.oft.param_names`; it shares upstream's
``PEFT_RECOMPUTE_PATCHED`` registry so the two can never double-patch a model.
"""

from functools import wraps

import torch

from megatron.bridge.orbit.oft.param_names import is_peft_adapter_param_name, is_trainable_base_param_name
from megatron.bridge.peft.recompute import PEFT_RECOMPUTE_PATCHED, _iter_unwrapped_models
from megatron.bridge.utils.common_utils import print_rank_0


def maybe_enable_recompute_inputs_grad_orbit(model) -> None:
    """Enable grad on TransformerBlock inputs when only orbit adapters are trainable.

    See upstream ``maybe_enable_recompute_inputs_grad`` for the full root-cause
    analysis (PP=1 + frozen base + recompute means CheckpointFunction.backward
    is never invoked). This variant recognizes OFT/CanonicalOFT parameter names.
    """
    from megatron.core.transformer.transformer_block import TransformerBlock

    try:
        for unwrapped_model in _iter_unwrapped_models(model):
            cfg = getattr(unwrapped_model, "config", None)
            if cfg is None or getattr(cfg, "recompute_method", None) is None:
                continue

            if id(unwrapped_model) in PEFT_RECOMPUTE_PATCHED:
                continue

            params = list(unwrapped_model.named_parameters())
            trainable_adapter = any(p.requires_grad and is_peft_adapter_param_name(n) for n, p in params)
            trainable_base = any(p.requires_grad and is_trainable_base_param_name(n) for n, p in params)

            if not (trainable_adapter and not trainable_base):
                continue  # Not adapter-only training, no fix needed

            patched = False
            for module in unwrapped_model.modules():
                if isinstance(module, TransformerBlock):
                    original_forward = module.forward

                    @wraps(original_forward)
                    def patched_forward(hidden_states, *args, _original_forward=original_forward, **kwargs):
                        if (
                            torch.is_tensor(hidden_states)
                            and not hidden_states.requires_grad
                            and hidden_states.is_floating_point()
                        ):
                            hidden_states = hidden_states.detach().requires_grad_(True)
                        return _original_forward(hidden_states, *args, **kwargs)

                    module.forward = patched_forward
                    patched = True

            if patched:
                PEFT_RECOMPUTE_PATCHED.add(id(unwrapped_model))
                print_rank_0(
                    "[Orbit PEFT+Recompute] Patched TransformerBlock.forward to enable grad on "
                    "hidden_states input for adapter-only (OFT/CanonicalOFT) training."
                )
    except Exception as exc:  # pragma: no cover - best effort logging
        print_rank_0(f"[Orbit PEFT+Recompute] Warning: Failed to patch TransformerBlock: {exc}")
