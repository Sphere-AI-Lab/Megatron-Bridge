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

from megatron.bridge.orbit.quant.fp8_utils import transform_sharded_state_dict_for_fp8


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
