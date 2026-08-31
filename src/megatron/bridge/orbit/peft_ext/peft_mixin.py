# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared behavior for orbit PEFT methods, added by subclassing rather than by
editing upstream ``megatron.bridge.peft``.

Any orbit PEFT class (OFT, CanonicalOFT, and their merge variants) mixes
this in *before* :class:`megatron.bridge.peft.base.PEFT` in its MRO to get:

- bias-placeholder normalization on quantized base models before freezing,
- the OFT-aware adapter-only recompute grad fix after transformation,
- an ``adapter_key_filter`` that recognizes orbit adapter parameter names
  (``.adapter_q.``, ``.oft_r``, ...) in addition to upstream's LoRA names.
"""

from megatron.bridge.orbit.oft.param_names import is_peft_adapter_param_name
from megatron.bridge.orbit.peft_ext.bias_normalization import normalize_disabled_bias_placeholders
from megatron.bridge.orbit.peft_ext.recompute_ext import maybe_enable_recompute_inputs_grad_orbit


class OrbitPEFTMixin:
    """Orbit-side extensions to :class:`megatron.bridge.peft.base.PEFT`."""

    def __call__(self, model, training: bool = True):
        # Quantized checkpoints carry disabled-bias placeholders that must be
        # normalized before freeze/transform walk the module tree.
        self._walk_model(model, normalize_disabled_bias_placeholders)

        model = super().__call__(model, training=training)

        if training:
            # Upstream's recompute fix only recognizes LoRA parameter names;
            # run the orbit-aware variant (shared registry, never double-patches).
            maybe_enable_recompute_inputs_grad_orbit(model)
            self._write_parameter_reports(model)

        return model

    def _write_parameter_reports(self, model) -> None:
        """Rank-0 TSV reports of trainable/frozen parameter partitions."""
        from megatron.bridge.utils.common_utils import get_rank_safe, print_rank_0

        if get_rank_safe() != 0:
            return
        try:
            from megatron.bridge.orbit.training.peft_reports import _write_peft_parameter_reports

            report_paths = _write_peft_parameter_reports(model, report_dir="/tmp")
            print_rank_0(f"  Trainable parameter report: {report_paths['trainable']}")
            print_rank_0(f"  Frozen parameter report: {report_paths['frozen']}")
            print_rank_0(f"  PEFT summary report: {report_paths['summary']}")
        except Exception as exc:  # pragma: no cover - reports are best-effort
            print_rank_0(f"[Orbit PEFT] Warning: failed to write parameter reports: {exc}")

    def adapter_key_filter(self, key) -> bool:
        """Save-filter that recognizes orbit adapter parameter names."""
        if isinstance(key, tuple):
            return key[1].requires_grad
        return key in self.params_to_save or is_peft_adapter_param_name(key)
