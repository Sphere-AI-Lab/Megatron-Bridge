"""Unit tests for the architecture-independent compressed-tensors INT4 adapter."""

import pytest
import torch

from megatron.bridge.models.conversion.quantization_utils import dequantize_int4, quantize_to_int4
from megatron.bridge.orbit.conversion.compressed_tensors_int4 import (
    CompressedTensorsINT4DequantMixin,
    hf_state_has_int4_triplets,
    int4_bridge_class_for,
    synthesize_virtual_weight_keys,
)


TRIPLET_KEYS = [
    "model.layers.0.self_attn.q_proj.weight_packed",
    "model.layers.0.self_attn.q_proj.weight_scale",
    "model.layers.0.self_attn.q_proj.weight_shape",
]
PLAIN_KEYS = ["model.embed_tokens.weight", "model.norm.weight"]


class _RecordingBridge:
    """Stand-in base bridge that records how it was called."""

    def __init__(self):
        self.seen_keys = None

    def build_conversion_tasks(self, hf_pretrained, megatron_model):
        self.seen_keys = list(hf_pretrained.state.source.get_all_keys())
        return ["sentinel-task"]

    def maybe_modify_loaded_hf_weight(self, hf_param, hf_state_dict):
        # Upstream-default behavior: plain indexing (model_bridge.py).
        if isinstance(hf_param, str):
            return hf_state_dict[hf_param]
        return {k: hf_state_dict[v] for k, v in hf_param.items()}


class _Source:
    def __init__(self, keys):
        self._keys = list(keys)

    def get_all_keys(self):
        return list(self._keys)


class _HFPretrained:
    def __init__(self, keys):
        class _State:
            pass

        self.state = _State()
        self.state.source = _Source(keys)


def _make_triplet(weight: torch.Tensor, group_size: int):
    packed, scale, shape = quantize_to_int4(weight, group_size=group_size)
    return packed, scale, shape


class TestVirtualKeySynthesis:
    def test_triplets_gain_virtual_weight_key(self):
        out = synthesize_virtual_weight_keys(TRIPLET_KEYS + PLAIN_KEYS)
        assert out[: len(TRIPLET_KEYS) + len(PLAIN_KEYS)] == TRIPLET_KEYS + PLAIN_KEYS
        assert out[-1] == "model.layers.0.self_attn.q_proj.weight"
        assert len(out) == len(TRIPLET_KEYS) + len(PLAIN_KEYS) + 1

    def test_incomplete_triplet_is_ignored(self):
        out = synthesize_virtual_weight_keys(TRIPLET_KEYS[:2] + PLAIN_KEYS)
        assert out == TRIPLET_KEYS[:2] + PLAIN_KEYS

    def test_existing_weight_key_is_not_duplicated(self):
        keys = TRIPLET_KEYS + ["model.layers.0.self_attn.q_proj.weight"]
        out = synthesize_virtual_weight_keys(keys)
        assert out == keys

    def test_detection(self):
        assert hf_state_has_int4_triplets(TRIPLET_KEYS)
        assert not hf_state_has_int4_triplets(PLAIN_KEYS)
        assert not hf_state_has_int4_triplets([])


class TestQuantDequantRoundtrip:
    @pytest.mark.parametrize("group_size", [32, 128])
    def test_roundtrip_close(self, group_size):
        torch.manual_seed(0)
        w = torch.randn(16, 256, dtype=torch.bfloat16)
        packed, scale, shape = _make_triplet(w, group_size)
        assert packed.dtype == torch.int32
        assert shape.tolist() == [16, 256]
        deq = dequantize_int4(packed, scale, shape)
        assert deq.dtype == torch.bfloat16
        assert deq.shape == w.shape
        # Symmetric 4-bit with per-group scale: max error is scale/2 per group.
        err = (deq.float() - w.float()).abs()
        bound = scale.float().repeat_interleave(group_size, dim=1)
        assert (err <= bound).all()

    def test_group_count_derived_from_scale(self):
        """One dequant call must serve both Kimi (32) and W4A16 (128) layouts."""
        torch.manual_seed(1)
        w = torch.randn(8, 256, dtype=torch.bfloat16)
        for group_size in (32, 128):
            packed, scale, shape = _make_triplet(w, group_size)
            deq = dequantize_int4(packed, scale, shape)
            assert deq.shape == w.shape


class TestMixinBehavior:
    def _composed_bridge(self):
        cls = int4_bridge_class_for(_RecordingBridge)
        return cls()

    def test_class_composition_and_cache(self):
        cls_a = int4_bridge_class_for(_RecordingBridge)
        cls_b = int4_bridge_class_for(_RecordingBridge)
        assert cls_a is cls_b
        assert cls_a.__mro__[0] is cls_a
        assert cls_a.__mro__[1] is CompressedTensorsINT4DequantMixin
        assert issubclass(cls_a, _RecordingBridge)
        assert cls_a.__name__ == "INT4_RecordingBridge"

    def test_build_tasks_sees_virtual_keys_and_restores_source(self):
        bridge = self._composed_bridge()
        hf = _HFPretrained(TRIPLET_KEYS + PLAIN_KEYS)
        result = bridge.build_conversion_tasks(hf, megatron_model=None)
        assert result == ["sentinel-task"]
        assert "model.layers.0.self_attn.q_proj.weight" in bridge.seen_keys
        # The temporary widening must not leak past the call.
        assert "model.layers.0.self_attn.q_proj.weight" not in hf.state.source.get_all_keys()

    def test_dequant_on_read_str_and_dict(self):
        torch.manual_seed(2)
        w = torch.randn(8, 64, dtype=torch.bfloat16)
        packed, scale, shape = _make_triplet(w, group_size=32)
        base = "model.layers.0.self_attn.q_proj.weight"
        state = {
            f"{base}_packed": packed,
            f"{base}_scale": scale,
            f"{base}_shape": shape,
            "model.norm.weight": torch.ones(4),
        }
        bridge = self._composed_bridge()

        out = bridge.maybe_modify_loaded_hf_weight(base, state)
        assert out.dtype == torch.bfloat16 and out.shape == w.shape
        assert torch.equal(out, dequantize_int4(packed, scale, shape))

        out_dict = bridge.maybe_modify_loaded_hf_weight({"q": base, "norm": "model.norm.weight"}, state)
        assert torch.equal(out_dict["q"], out)
        assert torch.equal(out_dict["norm"], state["model.norm.weight"])

    def test_plain_weight_passthrough(self):
        bridge = self._composed_bridge()
        state = {"model.norm.weight": torch.arange(4.0)}
        out = bridge.maybe_modify_loaded_hf_weight("model.norm.weight", state)
        assert torch.equal(out, state["model.norm.weight"])


class TestNamedBridgesUseGenericMixin:
    def test_named_classes_inherit_mixin(self):
        pytest.importorskip("transformer_engine")
        from megatron.bridge.orbit.model_bridges.deepseek_v3_int4_bridge import DeepSeekV3INT4Bridge
        from megatron.bridge.orbit.model_bridges.llama_int4_bridge import LlamaINT4Bridge
        from megatron.bridge.orbit.model_bridges.qwen3_int4_bridge import Qwen3INT4Bridge, Qwen3MoEINT4Bridge

        for cls in (Qwen3INT4Bridge, Qwen3MoEINT4Bridge, LlamaINT4Bridge, DeepSeekV3INT4Bridge):
            assert issubclass(cls, CompressedTensorsINT4DequantMixin)
