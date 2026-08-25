# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Quantization-scale parameter mapping (sphere fork).

Extracted from ``megatron.bridge.models.conversion.param_mapping``; consumed by
``megatron.bridge.sphere.model_bridges.deepseek_v4_bridge``.
"""

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping


class QuantScaleMapping(MegatronParamMapping[torch.Tensor]):
    """Mapping for FP4/FP8 quantization scale tensors that live as siblings to a quant weight.

    The scale parameter typically has shape (ceil(out/B), ceil(in/B)) for FP8
    (B=128) or (out, in//32) for FP4 — the dim-0 sharding dimension is
    out_features (in scale-block units for FP8). This mapping treats TP semantics
    the same as ColumnParallelMapping (when ``parent_kind="column"``) but DOES
    NOT cast dtypes. FP8 E8M0 (``torch.float8_e8m0fnu``) is a non-IEEE dtype;
    ``.to(target.dtype)`` on it produces garbage. The mapping therefore raises
    if the source and target dtypes don't already match.

    Lookup contract: the second arg passed by ``load_weights_hf_to_megatron`` is
    the *parent* module of the target Parameter. For ``decoder.layers.0.self_attention.wq_a.scale``
    the parent is the ``DSV4Linear`` named ``wq_a`` and ``.scale`` is its
    registered :class:`torch.nn.Parameter`.

    For TP=1 this is a direct copy, identical to :class:`DirectMapping` except
    for the strict no-cast assertion. For TP>1 with ``parent_kind="column"`` the
    sharding rule is the same as the parent weight's :class:`ColumnParallelMapping`
    (split dim 0). Row-parallel and replicated scales are NOT split: for FP8
    block-scaling, the scale block index along the input dim is local to each
    rank's input shard already.
    """

    def __init__(self, megatron_param: str, hf_param: str, parent_kind: str = "column"):
        super().__init__(megatron_param, hf_param)
        assert parent_kind in ("column", "row", "replicated"), (
            f"parent_kind must be one of 'column', 'row', 'replicated'; got {parent_kind!r}"
        )
        self.parent_kind = parent_kind

    def resolve(self, captures: Tuple[str, ...]) -> "MegatronParamMapping":
        """Return a new resolved QuantScaleMapping preserving parent_kind."""
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        return type(self)(resolved_megatron_param, resolved_hf_param, self.parent_kind)

    def hf_to_megatron(
        self,
        hf_weights: torch.Tensor,
        megatron_module: nn.Module,
    ) -> torch.Tensor:
        """Direct copy with optional dim-0 sharding for column-parallel parents.

        Refuses to cast non-IEEE scale dtypes (FP8 E8M0). Raises on dtype mismatch.
        """
        if hf_weights is None:
            raise ValueError(f"hf_weights must not be None for {self.megatron_param}")

        target_param = getattr(megatron_module, "scale", None)
        if target_param is None:
            raise ValueError(
                f"QuantScaleMapping target module {type(megatron_module).__name__} has no `.scale` attribute "
                f"(megatron_param={self.megatron_param})"
            )

        # No dtype cast: FP8 E8M0 is non-IEEE; preserve byte-exact value.
        if hf_weights.dtype != target_param.dtype:
            raise ValueError(
                f"Scale dtype mismatch for {self.megatron_param}: HF {hf_weights.dtype} vs Megatron "
                f"{target_param.dtype}; casting non-IEEE scale dtypes is unsafe"
            )

        if self.tp_size == 1 or self.parent_kind == "replicated":
            return hf_weights.to(device=target_param.device)

        # TP > 1, parent is column- or row-parallel: shard the corresponding axis.
        # column-parallel parent → out_features sharded → scale dim 0 sharded.
        # row-parallel parent    → in_features  sharded → scale dim 1 sharded
        # (each rank holds the scale-block columns covering its input shard).
        shard_dim = 0 if self.parent_kind == "column" else 1
        if self.tp_rank == 0:
            full_size = hf_weights.shape[shard_dim]
            if full_size % self.tp_size != 0:
                raise ValueError(
                    f"Scale tensor dim-{shard_dim} size {full_size} not divisible by tp_size "
                    f"{self.tp_size} for {self.megatron_param}"
                )
            splits = list(torch.chunk(hf_weights, self.tp_size, dim=shard_dim))
        else:
            splits = None
        return self.scatter_to_tp_ranks(
            splits,
            target_param.shape,
            target_param.dtype,
            target_param.device,
        )

    def megatron_to_hf(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """Gather scale shards across TP ranks and emit under the HF name.

        Mirrors :class:`ColumnParallelMapping` but skips the dequantize step
        (scales are already non-IEEE FP8 E8M0; don't touch them).
        """
        # Handle cross-PP broadcast first.
        megatron_weights = self.broadcast_from_pp_rank(megatron_weights, cache_key=str(self.hf_param))

        if megatron_weights is None:
            return {}

        if self.tp_size == 1 or self.parent_kind == "replicated":
            full_weights = megatron_weights
        else:
            gather_dim = 0 if self.parent_kind == "column" else 1
            gathered = self.gather_from_tp_ranks(megatron_weights)
            full_weights = torch.cat(gathered, dim=gather_dim)

        if self.is_expert and not self.is_adapter:
            return self.gather_from_ep_ranks(full_weights, megatron_module, self.hf_param)

        return {str(self.hf_param): full_weights}
