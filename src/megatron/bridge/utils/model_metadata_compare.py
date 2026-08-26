# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Compatibility shim: this module moved to ``megatron.bridge.orbit.conversion.model_metadata_compare``.

Import from the new path; this alias keeps old dotted paths working.
"""

import sys as _sys

from megatron.bridge.orbit.conversion import model_metadata_compare as _moved_module

_sys.modules[__name__] = _moved_module
