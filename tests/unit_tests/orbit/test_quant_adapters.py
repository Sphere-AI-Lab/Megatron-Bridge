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

"""Unit tests for the FP8/NVFP4 quant adapters and adapter chaining."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.bridge.models.conversion.param_mapping import GatedMLPMapping, QKVMapping, RowParallelMapping
from megatron.bridge.orbit.conversion.bridge_compose import quant_bridge_class_for
from megatron.bridge.orbit.conversion.compressed_tensors_int4 import (
    CompressedTensorsINT4DequantMixin,
    int4_bridge_class_for,
)
from megatron.bridge.orbit.conversion.fp8_preserve import (
    BlockFP8PreserveMixin,
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


pytestmark = pytest.mark.unit


class _PlusOneBridge:
    """Fake base bridge whose load hook adds one to every plain tensor."""

    def maybe_modify_loaded_hf_weight(self, hf_param, hf_state_dict):
        if isinstance(hf_param, str):
            return hf_state_dict[hf_param] + 1
        return {k: hf_state_dict[v] + 1 for k, v in hf_param.items()}


class _FP8LoadBridge(BlockFP8PreserveMixin):
    def __init__(self, task):
        self.task = task

    def build_conversion_tasks(self, _hf, _model):
        return [self.task]

    def _with_progress_tracking(self, tasks, _description):
        return tasks

    def maybe_modify_loaded_hf_weight(self, hf_param, hf_state):
        return {role: hf_state[key] for role, key in hf_param.items()}

    def _broadcast_shared_embeddings(self, _model):
        return None


class TestSharedComposition:
    def test_prefix_defaults_and_cache(self):
        cls_a = quant_bridge_class_for(BlockFP8PreserveMixin, _PlusOneBridge, name_prefix="FP8")
        cls_b = fp8_bridge_class_for(_PlusOneBridge)
        assert cls_a is cls_b  # same cache key -> same class
        assert cls_a.__name__ == "FP8_PlusOneBridge"
        # Default prefix derives from the mixin name and is a distinct cache entry.
        assert (
            quant_bridge_class_for(BlockFP8PreserveMixin, _PlusOneBridge).__name__ == "BlockFP8Preserve_PlusOneBridge"
        )

        nv = nvfp4_bridge_class_for(_PlusOneBridge)
        assert nv.__name__ == "NVFP4_PlusOneBridge"
        assert nv.__mro__[1] is ModelOptNVFP4DequantMixin

        i4 = int4_bridge_class_for(_PlusOneBridge)
        assert i4.__name__ == "INT4_PlusOneBridge"
        assert i4.__mro__[1] is CompressedTensorsINT4DequantMixin

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
        from megatron.bridge.models.kimi_vl.utils import quantize_to_int4

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
        with pytest.raises(ValueError, match="mixes E4M3FN and non-FP8"):
            _is_fp8({"q": bf16, "k": fp8})
        assert not _is_fp8({"q": bf16})
        with pytest.raises(TypeError, match="must be tensors"):
            _is_fp8({"q": fp8, "metadata": None})

    def test_load_scale_prefers_scale_inv_then_reciprocal(self):
        state = {
            "a.weight": torch.ones((2, 2)).to(torch.float8_e4m3fn),
            "a.weight_scale_inv": torch.tensor([2.0]),
            "b.weight": torch.ones((2, 2)).to(torch.float8_e4m3fn),
            "b.weight_scale": torch.tensor([4.0]),
        }
        assert torch.equal(_load_scale("a.weight", state), torch.tensor([[2.0]]))
        assert torch.equal(_load_scale("b.weight", state), torch.tensor([[0.25]]))
        assert _load_scale("c.weight", state) is None

    @pytest.mark.parametrize("scale", [torch.tensor([0.0]), torch.tensor([float("inf")])])
    def test_load_scale_rejects_nonpositive_or_nonfinite_reciprocal(self, scale: torch.Tensor):
        state = {
            "a.weight": torch.ones((2, 2)).to(torch.float8_e4m3fn),
            "a.weight_scale": scale,
        }

        with pytest.raises(ValueError, match="finite and positive"):
            _load_scale("a.weight", state)

    def test_tp_chunk(self):
        t = torch.arange(8.0).reshape(2, 4)
        assert torch.equal(_tp_chunk(t, 1, 0, dim=0), t)
        assert torch.equal(_tp_chunk(t, 2, 1, dim=1), t[:, 2:])

    def test_store_scale_inv_is_persistent_model_state(self):
        """A converted FP8 scale must survive ordinary state_dict/save paths."""
        module = nn.Linear(4, 3, bias=False)
        task = SimpleNamespace(
            mapping=SimpleNamespace(hf_param="hf.weight", tp_rank=0, tp_size=1),
            megatron_module=module,
            param_name="weight",
            param_weight=module.weight,
        )
        scale = torch.tensor([[0.5]])

        BlockFP8PreserveMixin()._store_scale_inv(
            task,
            {
                "hf.weight": torch.ones((3, 4)).to(torch.float8_e4m3fn),
                "hf.weight_scale_inv": scale,
            },
        )

        assert "weight_scale_inv" in module.state_dict()
        assert torch.equal(module.state_dict()["weight_scale_inv"], scale)

    def test_store_scale_inv_uses_indexed_grouped_expert_buffer_name(self):
        module = nn.Module()
        module.register_parameter("weight0", nn.Parameter(torch.empty((128, 128), dtype=torch.bfloat16)))
        task = SimpleNamespace(
            mapping=SimpleNamespace(hf_param="hf.weight", tp_rank=0, tp_size=1),
            megatron_module=module,
            param_name="weight0",
            param_weight=module.weight0,
        )
        scale = torch.tensor([[0.5]])

        BlockFP8PreserveMixin()._store_scale_inv(
            task,
            {
                "hf.weight": torch.ones((128, 128)).to(torch.float8_e4m3fn),
                "hf.weight_scale_inv": scale,
            },
        )

        assert "weight0_scale_inv" in module.state_dict()
        assert "weight_scale_inv" not in module.state_dict()
        assert torch.equal(module.state_dict()["weight0_scale_inv"], scale)

    @pytest.mark.parametrize("sharded_dim", [0, 1], ids=["column", "row"])
    def test_store_scale_inv_rejects_tp_cut_inside_128_element_block(self, sharded_dim: int):
        local_shape = [256, 256]
        local_shape[sharded_dim] = 192
        global_shape = list(local_shape)
        global_shape[sharded_dim] *= 2
        scale_shape = tuple((size + 127) // 128 for size in global_shape)
        module = nn.Linear(local_shape[1], local_shape[0], bias=False)
        inner_mapping = RowParallelMapping.__new__(RowParallelMapping) if sharded_dim == 1 else None
        mapping = SimpleNamespace(
            hf_param="hf.weight",
            tp_rank=0,
            tp_size=2,
            _mapping=inner_mapping,
        )
        task = SimpleNamespace(
            mapping=mapping,
            megatron_module=module,
            param_name="weight",
            param_weight=module.weight,
        )

        with pytest.raises(ValueError, match="128-element FP8 block"):
            BlockFP8PreserveMixin()._store_scale_inv(
                task,
                {
                    "hf.weight": torch.ones(global_shape).to(torch.float8_e4m3fn),
                    "hf.weight_scale_inv": torch.ones(scale_shape),
                },
            )

    @pytest.mark.parametrize("sharded_dim", [0, 1], ids=["column", "row"])
    def test_store_scale_inv_accepts_block_aligned_tp_cut(self, sharded_dim: int):
        local_shape = [256, 256]
        global_shape = list(local_shape)
        global_shape[sharded_dim] *= 2
        scale_shape = tuple(size // 128 for size in global_shape)
        source_scale = torch.arange(1.0, float(scale_shape[0] * scale_shape[1]) + 1.0).reshape(scale_shape)
        module = nn.Linear(local_shape[1], local_shape[0], bias=False)
        inner_mapping = RowParallelMapping.__new__(RowParallelMapping) if sharded_dim == 1 else None
        mapping = SimpleNamespace(
            hf_param="hf.weight",
            tp_rank=1,
            tp_size=2,
            _mapping=inner_mapping,
        )
        task = SimpleNamespace(
            mapping=mapping,
            megatron_module=module,
            param_name="weight",
            param_weight=module.weight,
        )

        BlockFP8PreserveMixin()._store_scale_inv(
            task,
            {
                "hf.weight": torch.ones(global_shape).to(torch.float8_e4m3fn),
                "hf.weight_scale_inv": source_scale,
            },
        )

        expected = torch.chunk(source_scale, 2, dim=sharded_dim)[1]
        assert torch.equal(module.weight_scale_inv, expected)

    @pytest.mark.parametrize(
        ("mapping", "missing_key"),
        [
            (
                GatedMLPMapping("linear_fc1.weight", "gate.weight", "up.weight"),
                "up.weight_scale_inv",
            ),
            (
                QKVMapping("linear_qkv.weight", "q.weight", "k.weight", "v.weight"),
                "k.weight_scale_inv",
            ),
            (
                QKVMapping("linear_qkv.weight", "q.weight", "k.weight", "v.weight"),
                "v.weight_scale_inv",
            ),
        ],
        ids=["gated-up", "qkv-k", "qkv-v"],
    )
    def test_load_preflight_rejects_incomplete_fp8_family_before_mutating_target(
        self,
        mapping,
        missing_key: str,
    ):
        module = nn.Linear(128, 256, bias=False, dtype=torch.bfloat16)
        task = SimpleNamespace(
            mapping=mapping,
            megatron_module=module,
            param_name=mapping.megatron_param,
            param_weight=module.weight,
        )
        state = {}
        for source_key in mapping.hf_param.values():
            state[source_key] = torch.ones((128, 128)).to(torch.float8_e4m3fn)
            state[f"{source_key}_scale_inv"] = torch.ones((1, 1))
        state.pop(missing_key)

        hf_pretrained = SimpleNamespace(
            state=state,
            model_name_or_path="test",
        )

        with pytest.raises(ValueError, match=missing_key):
            _FP8LoadBridge(task).load_weights_hf_to_megatron(hf_pretrained, module)

        assert module.weight.dtype == torch.bfloat16

    def test_load_preflight_rejects_mixed_fused_family_before_mutating_target(self):
        mapping = GatedMLPMapping("linear_fc1.weight", "gate.weight", "up.weight")
        module = nn.Linear(128, 256, bias=False, dtype=torch.bfloat16)
        task = SimpleNamespace(
            mapping=mapping,
            megatron_module=module,
            param_name=mapping.megatron_param,
            param_weight=module.weight,
        )
        state = {
            "gate.weight": torch.ones((128, 128)).to(torch.float8_e4m3fn),
            "gate.weight_scale_inv": torch.ones((1, 1)),
            "up.weight": torch.ones((128, 128), dtype=torch.bfloat16),
        }

        hf_pretrained = SimpleNamespace(state=state, model_name_or_path="test")

        with pytest.raises(ValueError, match="mixes E4M3FN and non-FP8"):
            _FP8LoadBridge(task).load_weights_hf_to_megatron(hf_pretrained, module)

        assert module.weight.dtype == torch.bfloat16

    def test_load_preserves_complete_gated_fp8_family_and_scale(self):
        mapping = GatedMLPMapping("linear_fc1.weight", "gate.weight", "up.weight")
        model = nn.Module()
        model.add_module("linear_fc1", nn.Linear(128, 256, bias=False, dtype=torch.bfloat16))
        task = SimpleNamespace(
            mapping=mapping,
            megatron_module=model,
            param_name=mapping.megatron_param,
            param_weight=model.linear_fc1.weight,
        )
        gate = torch.full((128, 128), 1.0).to(torch.float8_e4m3fn)
        up = torch.full((128, 128), 2.0).to(torch.float8_e4m3fn)
        state = {
            "gate.weight": gate,
            "gate.weight_scale_inv": torch.full((1, 1), 3.0),
            "up.weight": up,
            "up.weight_scale_inv": torch.full((1, 1), 4.0),
        }

        result = _FP8LoadBridge(task).load_weights_hf_to_megatron(
            SimpleNamespace(state=state, model_name_or_path="test"),
            model,
        )

        assert result == [model]
        assert model.linear_fc1.weight.dtype == torch.float8_e4m3fn
        torch.testing.assert_close(model.linear_fc1.weight, torch.cat([gate, up], dim=0))
        torch.testing.assert_close(
            model.linear_fc1.weight_scale_inv,
            torch.tensor([[3.0], [4.0]]),
        )


class TestNamedBridgesUseGenericMixins:
    def test_named_classes_inherit_mixins(self):
        pytest.importorskip("transformer_engine")
        from megatron.bridge.orbit.model_bridges.kimi_k25_vl_nvfp4_bridge import KimiK25VLNVFP4Bridge
        from megatron.bridge.orbit.model_bridges.qwen3_moe_fp8_bridge import Qwen3MoEFP8Bridge

        assert issubclass(Qwen3MoEFP8Bridge, BlockFP8PreserveMixin)
        assert issubclass(KimiK25VLNVFP4Bridge, ModelOptNVFP4DequantMixin)
