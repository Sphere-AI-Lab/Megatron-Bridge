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

import pytest
import torch
from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory

from megatron.bridge.orbit.quant.fp8_utils import (
    dequant_fp8,
    merge_qkv_scale_inv,
    transform_sharded_state_dict_for_fp8,
)


@pytest.mark.unit
def test_dense_fp8_transform_preserves_local_shape_with_prepended_axis() -> None:
    """A checkpoint-only prepended axis must not remove a local weight axis."""
    runtime_key = "decoder.layers.0.self_attention.linear_proj.weight"
    source = ShardedTensor.from_rank_offsets(
        runtime_key,
        torch.empty((5, 8), dtype=torch.bfloat16),
        prepend_axis_num=1,
    )

    transformed = transform_sharded_state_dict_for_fp8({runtime_key: source})

    weight = transformed[runtime_key]
    assert weight.key == f"{runtime_key}_w"
    assert weight.local_shape == (5, 8)
    assert weight.global_shape == (5, 8)
    assert weight.global_offset == (0, 0)
    assert weight.axis_fragmentations == (1, 1)
    assert weight.prepend_axis_num == 0

    scale = transformed[f"{runtime_key}_scale_inv"]
    assert scale.local_shape == (1, 1)
    assert scale.global_shape == (1, 1)
    assert scale.global_offset == (0, 0)
    assert scale.axis_fragmentations == (1, 1)
    assert scale.prepend_axis_num == 0


@pytest.mark.unit
def test_dense_fp8_factory_scale_drops_prepended_layer_axis() -> None:
    """Split weights and their scale must use the same per-layer metadata."""
    runtime_key = "decoder.layers.0.mlp.linear_fc1.weight"
    fused_weight = torch.empty((10, 8), dtype=torch.bfloat16)

    def build_factory(
        factory_key: str,
        data: torch.Tensor,
        replica_id: int,
        flattened_range: slice | None,
    ) -> list[ShardedTensor]:
        assert flattened_range is None
        gate, up = torch.chunk(data, 2, dim=0)
        return [
            ShardedTensor.from_rank_offsets(
                f"{factory_key}_w",
                gate,
                (0, 0, 36),
                replica_id=replica_id,
                prepend_axis_num=1,
            ),
            ShardedTensor.from_rank_offsets(
                f"{factory_key}_v",
                up,
                (0, 0, 36),
                replica_id=replica_id,
                prepend_axis_num=1,
            ),
        ]

    factory = ShardedTensorFactory(
        key=runtime_key,
        data=fused_weight,
        build_fn=build_factory,
        merge_fn=lambda shards: torch.cat(shards, dim=0),
    )

    transformed = transform_sharded_state_dict_for_fp8({runtime_key: factory}, block_size=4)

    scale = transformed[f"{runtime_key}_scale_inv"]
    assert scale.local_shape == (3, 2)


@pytest.mark.unit
@pytest.mark.parametrize("axis", [0, 1], ids=["output", "input"])
def test_fp8_scale_schema_rejects_tp_boundary_inside_quantization_block(axis: int) -> None:
    """A local scale grid cannot restart halfway through a global 128-wide block."""
    runtime_key = "decoder.layers.0.self_attention.linear_proj.weight"
    local_shape = [256, 256]
    local_shape[axis] = 192
    source = ShardedTensor.from_rank_offsets(
        runtime_key,
        torch.empty(tuple(local_shape), dtype=torch.bfloat16),
        (axis, 1, 2),
    )

    with pytest.raises(ValueError, match="128-element FP8 block"):
        transform_sharded_state_dict_for_fp8({runtime_key: source})


@pytest.mark.unit
def test_fp8_scale_schema_accepts_block_aligned_tp_boundary() -> None:
    runtime_key = "decoder.layers.0.self_attention.linear_proj.weight"
    source = ShardedTensor.from_rank_offsets(
        runtime_key,
        torch.empty((256, 256), dtype=torch.bfloat16),
        (1, 1, 2),
    )

    transformed = transform_sharded_state_dict_for_fp8({runtime_key: source})

    scale = transformed[f"{runtime_key}_scale_inv"]
    assert scale.local_shape == (2, 2)
    assert scale.global_shape == (2, 4)
    assert scale.global_offset == (0, 2)


@pytest.mark.unit
def test_fp8_scale_schema_preserves_high_rank_prefix_offset() -> None:
    """Only the final two weight axes are expressed in 128-wide scale blocks."""
    runtime_key = "decoder.layers.0.self_attention.linear_proj.weight"
    source = ShardedTensor.from_rank_offsets(
        runtime_key,
        torch.empty((1, 256, 256), dtype=torch.bfloat16),
        (0, 1, 2),
    )

    transformed = transform_sharded_state_dict_for_fp8({runtime_key: source})

    weight = transformed[runtime_key]
    assert weight.local_shape == (1, 256, 256)
    assert weight.global_shape == (2, 256, 256)
    assert weight.global_offset == (1, 0, 0)

    scale = transformed[f"{runtime_key}_scale_inv"]
    assert scale.local_shape == (1, 2, 2)
    assert scale.global_shape == (2, 2, 2)
    assert scale.global_offset == (1, 0, 0)


@pytest.mark.unit
def test_merge_qkv_scale_inv_separates_output_gate_rows_per_gqa_group() -> None:
    """Scale rows must follow the fused weight's Q, gate, K, V ordering."""
    config = type(
        "Config",
        (),
        {
            "num_attention_heads": 4,
            "num_query_groups": 2,
            "attention_output_gate": True,
        },
    )()
    # HF stores each query head as [Q, gate]. One scale row represents each
    # head-sized span in this deliberately small fixture.
    q_scale = torch.arange(8, dtype=torch.float32).unsqueeze(1)
    k_scale = torch.tensor([[10.0], [11.0]])
    v_scale = torch.tensor([[20.0], [21.0]])

    merged = merge_qkv_scale_inv(config, q_scale, k_scale, v_scale)

    expected = torch.tensor([[0.0], [2.0], [1.0], [3.0], [10.0], [20.0], [4.0], [6.0], [5.0], [7.0], [11.0], [21.0]])
    torch.testing.assert_close(merged, expected)


@pytest.mark.unit
def test_dequant_fp8_uses_128_wide_scale_blocks_at_partial_edges() -> None:
    """A ceil-grid scale keeps 128-wide blocks instead of repartitioning them."""
    weight = torch.ones((130, 129), dtype=torch.float32)
    scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    actual = dequant_fp8(weight, scale, out_dtype=torch.float32)

    expected = torch.ones_like(weight)
    expected[:128, 128:] = 2.0
    expected[128:, :128] = 3.0
    expected[128:, 128:] = 4.0
    torch.testing.assert_close(actual, expected)
