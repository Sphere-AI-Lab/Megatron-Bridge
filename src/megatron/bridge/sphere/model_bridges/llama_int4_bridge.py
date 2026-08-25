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

"""INT4-preserving bridge for dense Llama models.

Handles HuggingFace Llama checkpoints where every linear weight is stored in
compressed-tensors `pack-quantized` triplet format
(``weight_packed`` + ``weight_scale`` + ``weight_shape``), while layernorms
and embeddings remain BF16.

This is the dense-model analog of `DeepSeekV3INT4Bridge`. Unlike the DeepSeek
path, every linear (attention Q/K/V/O, gated MLP gate/up/down, lm_head if
quantized) is INT4-quantized at the HF level, not just experts. Merging happens
post-dequant: separate HF Q/K/V triplets become one Megatron ``linear_qkv``,
gate+up become one ``linear_fc1``. Those merged BF16 tensors are re-quantized
by ``build_int4_direct_model_state_dict`` before the dist checkpoint is written.

Verified against:
    nm-testing/Meta-Llama-3-8B-Instruct-W4A16-compressed-tensors-test
    (compressed-tensors, format=pack-quantized, symmetric, group_size=128,
    no actorder / weight_g_idx)
"""

import logging
from typing import List, Mapping, Union

import torch

from megatron.bridge.sphere.low_precision.int4 import dequantize_int4
from megatron.bridge.models.llama.llama_bridge import LlamaBridge

logger = logging.getLogger(__name__)


class LlamaINT4Bridge(LlamaBridge):
    """Llama bridge that preserves INT4 through direct-write conversion."""

    # --------------------------------------------------------------------- #
    # Virtual key synthesis: make INT4 triplets visible as .weight keys
    # --------------------------------------------------------------------- #

    def build_conversion_tasks(self, hf_pretrained, megatron_model) -> List:
        """Synthesize virtual ``.weight`` keys from INT4 quantized triplets.

        The HF checkpoint stores quantized linear weights as triplets
        (``weight_packed``, ``weight_scale``, ``weight_shape``) with no plain
        ``.weight`` key. The parent bridge's mapping registry (QKVMapping,
        GatedMLPMapping, AutoMapping) all look up ``*.weight`` keys, so we
        synthesize those virtual keys during task construction. The
        dequant-on-read happens later in ``maybe_modify_loaded_hf_weight``.
        """
        original_get_all_keys = hf_pretrained.state.source.get_all_keys

        def _get_all_keys_with_virtual():
            keys = original_get_all_keys()
            all_keys_set = set(keys)
            virtual_keys = []
            for key in keys:
                if key.endswith("_packed"):
                    base = key[:-7]  # "...weight_packed" -> "...weight"
                    if f"{base}_scale" in all_keys_set and f"{base}_shape" in all_keys_set:
                        virtual_keys.append(base)
            return keys + virtual_keys

        hf_pretrained.state.source.get_all_keys = _get_all_keys_with_virtual
        try:
            return super().build_conversion_tasks(hf_pretrained, megatron_model)
        finally:
            hf_pretrained.state.source.get_all_keys = original_get_all_keys

    # --------------------------------------------------------------------- #
    # Dequant INT4 -> BF16 at weight read time
    # --------------------------------------------------------------------- #

    def maybe_modify_loaded_hf_weight(
        self, hf_param: Union[str, dict], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(hf_param, dict):
            return {role: self._maybe_dequant_int4(key, hf_state_dict)
                    for role, key in hf_param.items()}
        return self._maybe_dequant_int4(hf_param, hf_state_dict)

    def _maybe_dequant_int4(
        self, hf_key: str, hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        if hf_key.endswith(".weight"):
            base = hf_key[: -len(".weight")]
            packed_key = base + ".weight_packed"
            if packed_key in hf_state_dict:
                w = dequantize_int4(
                    hf_state_dict[packed_key],
                    hf_state_dict[base + ".weight_scale"],
                    hf_state_dict[base + ".weight_shape"],
                    device=hf_state_dict[packed_key].device,
                )
                logger.info("Dequantised INT4 -> BF16: %s  shape=%s", hf_key, list(w.shape))
                return w
        return hf_state_dict[hf_key]
