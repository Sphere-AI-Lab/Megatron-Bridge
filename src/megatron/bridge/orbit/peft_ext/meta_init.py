# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Meta-device materialization that tolerates ModelOpt quantized tensors.

Divergent variant of ``megatron.bridge.models.common.unimodal.to_empty_if_meta_device``:
identical semantics, plus unwrapping of ModelOpt ``QTensorWrapper`` tensors
during materialization. ModelOpt's ``RealQuantLinear._apply()`` re-wraps its
parameters itself; handing it a tensor subclass makes PyTorch Parameter
construction reject the wrapper because ``detach()`` returns a plain Tensor
in this ModelOpt/PyTorch combination.

This cannot delegate to that function: the unwrap has to happen inside the
``_apply`` callback, and the rejection occurs partway through ``_apply``, so
there is no post-hoc fixup point. Keep the two in sync on upstream re-fetch.
"""

import torch


def _unwrap_modelopt_qtensor_wrapper(tensor: torch.Tensor) -> torch.Tensor:
    if type(tensor).__name__ == "QTensorWrapper" and hasattr(tensor, "as_subclass"):
        return tensor.as_subclass(torch.Tensor)
    return tensor


def to_empty_if_meta_device(module: torch.nn.Module, *, device: torch.device, recurse=True):
    """Move tensors to device if not meta device; otherwise materialize with empty_like().

    Only meta-device tensors are materialized (plain ``to_empty()`` would also
    clobber buffers that already hold precomputed values).
    """

    def _empty_like_if_meta(tensor: torch.Tensor, *, device: torch.device):
        if tensor.device == torch.device("meta"):
            materialized = torch.empty_like(tensor, device=device)
        else:
            materialized = tensor.to(device)
        return _unwrap_modelopt_qtensor_wrapper(materialized)

    return module._apply(lambda t: _empty_like_if_meta(t, device=device), recurse=recurse)
