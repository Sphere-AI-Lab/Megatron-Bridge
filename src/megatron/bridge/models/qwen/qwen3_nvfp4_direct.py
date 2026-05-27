# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Compatibility wrappers for NVFP4 direct-conversion helpers."""

from megatron.bridge.models.conversion.low_precision.common import (
    add_tensor_entry,
    build_single_rank_meta_provider,
    patch_meta_init_for_te_modules,
    prepare_empty_model_state,
    retain_non_tensor_entries,
)
from megatron.bridge.models.conversion.low_precision.nvfp4 import (
    apply_modelopt_nvfp4_to_meta_model,
    build_fused_nvfp4_weight_entries,
    build_nvfp4_direct_model_state_dict,
    build_megatron_nvfp4_weight_entries,
    collect_nvfp4_target_module_names,
    extract_nvfp4_weight_bundle,
    is_nvfp4_source,
    is_nvfp4_weight_mapping,
)

__all__ = [
    "add_tensor_entry",
    "apply_modelopt_nvfp4_to_meta_model",
    "build_dense_qwen3_meta_provider",
    "build_direct_model_state_dict",
    "build_fused_nvfp4_weight_entries",
    "build_megatron_nvfp4_weight_entries",
    "build_nvfp4_meta_provider",
    "build_single_rank_meta_provider",
    "collect_nvfp4_target_module_names",
    "extract_nvfp4_weight_bundle",
    "is_nvfp4_source",
    "is_nvfp4_weight_mapping",
    "patch_meta_init_for_te_modules",
    "prepare_empty_model_state",
    "retain_non_tensor_entries",
]


def build_dense_qwen3_meta_provider(hf_model_path: str):
    """Build a dense Qwen3 provider without loading weights for direct conversion."""
    return build_single_rank_meta_provider(hf_model_path)


def build_nvfp4_meta_provider(
    hf_model_path: str,
    *,
    trust_remote_code: bool = False,
):
    """Build a single-rank provider without loading weights for direct conversion."""
    return build_single_rank_meta_provider(
        hf_model_path,
        trust_remote_code=trust_remote_code,
    )


def build_direct_model_state_dict(*args, **kwargs):
    """Compatibility wrapper for the generic NVFP4 direct-save builder."""
    return build_nvfp4_direct_model_state_dict(*args, **kwargs)
