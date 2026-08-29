"""Unit tests for the FP8/NVFP4 quant adapters and adapter chaining."""

import pytest
import torch

from megatron.bridge.orbit.conversion.bridge_compose import quant_bridge_class_for
from megatron.bridge.orbit.conversion.compressed_tensors_int4 import (
    CompressedTensorsINT4Mixin,
    int4_bridge_class_for,
)
from megatron.bridge.orbit.conversion.fp8_preserve import (
    FP8PreserveMixin,
    _is_fp8,
    _load_scale,
    _tp_chunk,
    fp8_bridge_class_for,
)
from megatron.bridge.orbit.conversion.modelopt_nvfp4 import (
    ModelOptNVFP4DequantMixin,
    hf_state_has_nvfp4_bundles,
    is_nvfp4_bundle_key,
    nvfp4_bridge_class_for,
)


class _PlusOneBridge:
    """Fake base bridge whose load hook adds one to every plain tensor."""

    def maybe_modify_loaded_hf_weight(self, hf_param, hf_state_dict):
        if isinstance(hf_param, str):
            return hf_state_dict[hf_param] + 1
        return {k: hf_state_dict[v] + 1 for k, v in hf_param.items()}


class TestSharedComposition:
    def test_prefix_defaults_and_cache(self):
        cls_a = quant_bridge_class_for(FP8PreserveMixin, _PlusOneBridge, name_prefix="FP8")
        cls_b = fp8_bridge_class_for(_PlusOneBridge)
        assert cls_a is cls_b  # same cache key -> same class
        assert cls_a.__name__ == "FP8_PlusOneBridge"
        # Default prefix derives from the mixin name and is a distinct cache entry.
        assert quant_bridge_class_for(FP8PreserveMixin, _PlusOneBridge).__name__ == "FP8Preserve_PlusOneBridge"

        nv = nvfp4_bridge_class_for(_PlusOneBridge)
        assert nv.__name__ == "NVFP4_PlusOneBridge"
        assert nv.__mro__[1] is ModelOptNVFP4DequantMixin

        i4 = int4_bridge_class_for(_PlusOneBridge)
        assert i4.__name__ == "INT4_PlusOneBridge"
        assert i4.__mro__[1] is CompressedTensorsINT4Mixin

    def test_distinct_mixins_get_distinct_classes(self):
        assert fp8_bridge_class_for(_PlusOneBridge) is not nvfp4_bridge_class_for(_PlusOneBridge)


class TestNVFP4Detection:
    def test_bundle_key(self):
        keys = {"a.weight", "a.weight_scale", "a.weight_scale_2", "b.weight", "b.weight_scale"}
        assert is_nvfp4_bundle_key("a.weight", keys)
        assert not is_nvfp4_bundle_key("b.weight", keys)  # missing _scale_2
        assert hf_state_has_nvfp4_bundles(keys)
        assert not hf_state_has_nvfp4_bundles({"b.weight", "b.weight_scale"})


class TestNVFP4Dequant:
    def _bundle(self, out_f=8, in_f=64):
        from megatron.bridge.orbit.low_precision.nvfp4 import quantize_to_nvfp4

        w = torch.randn(out_f, in_f, dtype=torch.bfloat16)
        packed, scale, scale_2, _shape = quantize_to_nvfp4(w)
        return w, packed, scale, scale_2

    def test_mixin_dequants_bundle_and_chains_plain_keys(self):
        torch.manual_seed(0)
        w, packed, scale, scale_2 = self._bundle()
        state = {
            "layer.q_proj.weight": packed,
            "layer.q_proj.weight_scale": scale,
            "layer.q_proj.weight_scale_2": scale_2,
            "plain.weight": torch.zeros(4),
        }
        bridge = nvfp4_bridge_class_for(_PlusOneBridge)()

        out = bridge.maybe_modify_loaded_hf_weight("layer.q_proj.weight", state)
        assert out.dtype == torch.bfloat16
        assert out.shape == w.shape

        from megatron.bridge.orbit.low_precision.nvfp4 import dequantize_nvfp4

        shape = torch.tensor([packed.shape[0], packed.shape[1] * 2], dtype=torch.int64)
        expected = dequantize_nvfp4(packed, scale, scale_2, shape, dtype=torch.bfloat16, device=packed.device)
        assert torch.equal(out, expected)

        # Plain key falls through to the base bridge hook (+1), proving chaining.
        chained = bridge.maybe_modify_loaded_hf_weight("plain.weight", state)
        assert torch.equal(chained, torch.ones(4))

        # Dict form mixes both behaviors per key.
        d = bridge.maybe_modify_loaded_hf_weight({"q": "layer.q_proj.weight", "p": "plain.weight"}, state)
        assert torch.equal(d["q"], expected)
        assert torch.equal(d["p"], torch.ones(4))


class TestINT4Chaining:
    def test_non_triplet_keys_defer_to_base_hook(self):
        from megatron.bridge.models.conversion.quantization_utils import quantize_to_int4

        torch.manual_seed(1)
        w = torch.randn(8, 64, dtype=torch.bfloat16)
        packed, scale, shape = quantize_to_int4(w, group_size=32)
        base = "layer.q_proj.weight"
        state = {
            f"{base}_packed": packed,
            f"{base}_scale": scale,
            f"{base}_shape": shape,
            "plain.weight": torch.zeros(3),
        }
        bridge = int4_bridge_class_for(_PlusOneBridge)()
        assert bridge.maybe_modify_loaded_hf_weight(base, state).shape == w.shape
        assert torch.equal(bridge.maybe_modify_loaded_hf_weight("plain.weight", state), torch.ones(3))


class TestFP8Helpers:
    def test_is_fp8(self):
        fp8 = torch.zeros(2, 2).to(torch.float8_e4m3fn)
        bf16 = torch.zeros(2, 2, dtype=torch.bfloat16)
        assert _is_fp8(fp8)
        assert not _is_fp8(bf16)
        assert _is_fp8({"q": bf16, "k": fp8})
        assert not _is_fp8({"q": bf16})

    def test_load_scale_prefers_scale_inv_then_reciprocal(self):
        state = {"a.weight_scale_inv": torch.tensor([2.0]), "b.weight_scale": torch.tensor([4.0])}
        assert torch.equal(_load_scale("a.weight", state), torch.tensor([2.0]))
        assert torch.equal(_load_scale("b.weight", state), torch.tensor([0.25]))
        assert _load_scale("c.weight", state) is None

    def test_tp_chunk(self):
        t = torch.arange(8.0).reshape(2, 4)
        assert torch.equal(_tp_chunk(t, 1, 0, dim=0), t)
        assert torch.equal(_tp_chunk(t, 2, 1, dim=1), t[:, 2:])


class TestNamedBridgesUseGenericMixins:
    def test_named_classes_inherit_mixins(self):
        pytest.importorskip("transformer_engine")
        from megatron.bridge.orbit.model_bridges.kimi_k25_vl_nvfp4_bridge import KimiK25VLNVFP4Bridge
        from megatron.bridge.orbit.model_bridges.qwen3_moe_fp8_bridge import Qwen3MoEFP8Bridge

        assert issubclass(Qwen3MoEFP8Bridge, FP8PreserveMixin)
        assert issubclass(KimiK25VLNVFP4Bridge, ModelOptNVFP4DequantMixin)
