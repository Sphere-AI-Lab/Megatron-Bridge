# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Compatibility shim: this module moved to ``megatron.bridge.sphere.quant.qwen3_fp8_gemm``.

Import from the new path; this alias keeps old dotted paths working.
"""

import sys as _sys

from megatron.bridge.sphere.quant import qwen3_fp8_gemm as _moved_module

_sys.modules[__name__] = _moved_module
