# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared helpers for low-precision HF -> Megatron checkpoint conversion."""

from . import fp8, int8
from .common import (
    add_tensor_entry,
    build_single_rank_meta_provider,
    patch_meta_init_for_te_modules,
    prepare_empty_model_state,
    retain_non_tensor_entries,
)
from .fp8 import (
    apply_modelopt_fp8_to_meta_model,
    build_fp8_direct_model_state_dict,
    build_fp8_scale_inv_key,
    build_merged_scale_inv_for_task,
)
from .int4 import (
    build_int4_direct_model_state_dict,
    convert_hf_weight_for_direct_save,
    dequantize_int4,
    hf_param_uses_int4,
    quantize_to_int4,
)
from .nvfp4 import (
    apply_modelopt_nvfp4_to_meta_model,
    build_fused_nvfp4_weight_entries,
    build_megatron_nvfp4_weight_entries,
    build_nvfp4_direct_model_state_dict,
    collect_nvfp4_target_module_names,
    extract_nvfp4_weight_bundle,
    is_nvfp4_source,
    is_nvfp4_weight_mapping,
    scale_to_amax,
)

__all__ = [
    "add_tensor_entry",
    "apply_modelopt_fp8_to_meta_model",
    "build_fp8_direct_model_state_dict",
    "build_fp8_scale_inv_key",
    "build_merged_scale_inv_for_task",
    "apply_modelopt_nvfp4_to_meta_model",
    "build_fused_nvfp4_weight_entries",
    "build_int4_direct_model_state_dict",
    "build_megatron_nvfp4_weight_entries",
    "build_nvfp4_direct_model_state_dict",
    "build_single_rank_meta_provider",
    "collect_nvfp4_target_module_names",
    "convert_hf_weight_for_direct_save",
    "dequantize_int4",
    "extract_nvfp4_weight_bundle",
    "fp8",
    "hf_param_uses_int4",
    "int8",
    "is_nvfp4_source",
    "is_nvfp4_weight_mapping",
    "patch_meta_init_for_te_modules",
    "prepare_empty_model_state",
    "quantize_to_int4",
    "retain_non_tensor_entries",
    "scale_to_amax",
]
