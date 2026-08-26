# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""uint8 views for NCCL collectives over sub-IEEE dtypes (orbit fork).

Extracted from ``megatron.bridge.models.conversion.param_mapping``; that module
imports :func:`_maybe_byte_view_for_nccl` back as its only seam.
"""

import torch

# torch.distributed's NCCL backend (PyTorch's ``to_nccl_data_type`` dispatcher)
# is missing entries for several 1-byte sub-IEEE dtypes used by MX/NVFP4
# quantization — packed FP4 weights (``float4_e2m1fn_x2``, two FP4 elements
# per byte) and FP8 block scales (``float8_e8m0fnu``). NCCL transports them
# fine as raw bytes; only the PyTorch dispatcher lacks the mapping. We view
# such tensors as uint8 (same shape, shared storage) for the collective and
# let the receiver read the bytes back through the original typed view —
# bit-exact, no copy, no scratch buffer.
_NCCL_BYTE_VIEW_DTYPES: set = set()
for _name in ("float8_e8m0fnu", "float4_e2m1fn_x2"):
    _dt = getattr(torch, _name, None)
    if _dt is not None:
        _NCCL_BYTE_VIEW_DTYPES.add(_dt)


def _maybe_byte_view_for_nccl(t: torch.Tensor) -> torch.Tensor:
    """Return a uint8 view if ``t``'s dtype is in :data:`_NCCL_BYTE_VIEW_DTYPES`.

    Storage is shared, so writes to the returned view are visible through ``t``.
    Caller keeps ``t`` and returns it after the collective — no restore needed.
    """
    if t.dtype in _NCCL_BYTE_VIEW_DTYPES:
        return t.view(torch.uint8)
    return t
