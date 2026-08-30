# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""LoRA over INT4-packed base weights, as a plain subclass of upstream LoRA.

Upstream's :class:`LoRALinear` calls ``base_linear_forward`` on the wrapped
module, which assumes a dense ``weight``. Orbit's direct-load INT4 checkpoints
replace that with ``weight_packed``/``weight_scale``/``weight_shape``, so the
base forward must dequantize first (:func:`_base_linear_forward_int4`).
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from megatron.bridge.orbit.peft_ext.int4_lora_forward import _base_linear_forward_int4
from megatron.bridge.orbit.peft_ext.peft_mixin import OrbitPEFTMixin
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.peft.lora_layers import LoRALinear


def _is_int4_packed_linear(module: nn.Module) -> bool:
    return hasattr(module, "weight_packed") and hasattr(module, "weight_scale") and hasattr(module, "weight_shape")


class Int4LoRALinear(LoRALinear):
    """LoRALinear whose base forward dequantizes INT4-packed weights."""

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # pylint: disable=C0115,C0116
        linear_output, bias, layernorm_output = _base_linear_forward_int4(self, x)
        if not self._adapter_enabled:
            return linear_output, bias
        adapter_output = self.adapter_forward(self.adapter, layernorm_output.contiguous(), *args, **kwargs)
        adapter_output = adapter_output.reshape(linear_output.shape)
        return linear_output + adapter_output, bias


class Int4LoRA(OrbitPEFTMixin, LoRA):
    """Upstream LoRA that wraps INT4-packed base linears with :class:`Int4LoRALinear`."""

    def transform(self, module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None) -> nn.Module:
        out = super().transform(module, name, prefix)
        # Only the exact plain-LoRALinear wrapper gets the INT4 base forward
        # (TEFusedLoRALinear fuses the base matmul and cannot dequantize).
        if type(out) is LoRALinear and _is_int4_packed_linear(out.to_wrap):
            return Int4LoRALinear(out.to_wrap, out.adapter)
        return out
