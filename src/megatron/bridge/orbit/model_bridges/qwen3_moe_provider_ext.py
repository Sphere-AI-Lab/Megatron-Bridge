# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Provider tweaks shared by orbit's Qwen3-MoE quantized bridges.

Upstream's :class:`Qwen3MoEBridge` leaves the router in the model dtype and
relies on the provider default for ``moe_layer_freq``. The orbit quantized
checkpoints were produced/validated with an fp32 router and an explicit
per-layer MoE pattern derived from the HF config (``decoder_sparse_step`` /
``mlp_only_layers``), so the quant bridges restore that behavior here instead
of editing the upstream base class.
"""


def apply_qwen3_moe_orbit_provider_settings(provider, hf_config):
    """Apply Orbit's Qwen3-MoE runtime layout to an actual model provider."""
    provider.moe_router_dtype = "fp32"

    decoder_sparse_step = getattr(hf_config, "decoder_sparse_step", 1) or 0
    mlp_only_layers = set(getattr(hf_config, "mlp_only_layers", []) or [])
    if getattr(hf_config, "num_experts", 0) > 0 and decoder_sparse_step > 0:
        provider.moe_layer_freq = [
            1 if (layer_idx not in mlp_only_layers) and (layer_idx + 1) % decoder_sparse_step == 0 else 0
            for layer_idx in range(hf_config.num_hidden_layers)
        ]
    else:
        provider.moe_layer_freq = [0] * hf_config.num_hidden_layers

    return provider


class Qwen3MoEOrbitProviderMixin:
    """Mixed in before ``Qwen3MoEBridge`` to adjust the generated provider."""

    def provider_bridge(self, hf_pretrained):
        provider = super().provider_bridge(hf_pretrained)
        return apply_qwen3_moe_orbit_provider_settings(provider, hf_pretrained.config)
