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

"""CanonicalOFT — per-projection input rotations on Megatron's fused QKV / FC1.

Unlike the legacy ``OFT`` class (one shared rotation R for the fused linear_qkv),
CanonicalOFT attaches three independent rotations (R_q, R_k, R_v) and applies
each to its own projection slice. This is the only correct OFT semantics for
fused projections, and is what ``--oft-type canonical_oft`` -- the default on
both orbit launchers -- selects. ``--oft-type oft`` opts back into the legacy
class.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
)

from megatron.bridge.orbit.oft.oft import _SplitLNOFTLinear
from megatron.bridge.orbit.oft.oft_layers import (
    OFTLinear,
    OFTRotationModule,
    OFTVocabParallelEmbedding,
    _clear_disabled_bias_parameters,
    _is_direct_fp8_runtime_weight,
    _make_expert_ep_sharded_tensor,
    _module_bias_enabled,
)
from megatron.bridge.orbit.peft_ext.adapter_attrs import get_oft_adapter_attributes_from_linear
from megatron.bridge.orbit.peft_ext.meta_init import to_empty_if_meta_device
from megatron.bridge.orbit.peft_ext.peft_mixin import OrbitPEFTMixin
from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.module_matcher import ModuleMatcher
from megatron.bridge.peft.utils import is_expert_linear, is_grouped_expert_linear
from megatron.bridge.utils.import_utils import safe_import_from


try:
    from megatron.bridge.orbit.oft.triton_oft import oft_r_by_expert
except ImportError:
    oft_r_by_expert = None


logger = logging.getLogger(__name__)


def _split_wrapper_sharded_state_dict(
    module: nn.Module,
    prefix: str = "",
    sharded_offsets: tuple = (),
    metadata: Optional[dict] = None,
):
    """Build a sharded state dict by delegating to each child module.

    The split wrappers replace their target instead of subclassing ``AdapterWrapper``,
    so mcore's plain-``nn.Module`` fallback would snapshot the whole subtree at once and
    mark it replicated, discarding ``to_wrap``'s TP sharding and ``oft_r``'s axis map.

    Key names are unchanged, but sharding metadata is not: ``oft_r`` becomes TP-sharded
    on axis 0 where the fallback marked it replicated, so checkpoints written before this
    fix are not interchangeable with later ones at TP>1. At TP=1 nothing changes.
    """
    from megatron.core.transformer.utils import sharded_state_dict_default

    sharded_state_dict = {}
    for name, child in module.named_children():
        sharded_state_dict.update(
            sharded_state_dict_default(
                child,
                f"{prefix}{name}.",
                sharded_offsets,
                metadata,
                tp_group=child.tp_group,
            )
        )
    return sharded_state_dict


_QUANTIZED_SHARED_FALLBACK_WARNED: set[tuple[str, str]] = set()


def _has_dequantized_nvfp4_buffers(module: nn.Module) -> bool:
    return hasattr(module, "_nvfp4_weight_scale") and hasattr(module, "_nvfp4_weight_double_scale")


_DEQUANT_HANDLE_TAG = object()


def _dequantize_single_weight_base(module: nn.Module, dtype: torch.dtype):
    """BF16 view of a quantized single-weight fused base, plus a rebuild closure.

    Returns ``None`` when the base weight is directly usable for compute (plain
    BF16, or the materialized dequantized-NVFP4 layout whose ``weight`` is a
    persistent BF16 parameter) -- callers then use ``module.weight`` with no
    hooks, since autograd saving a persistent parameter costs nothing extra.

    Otherwise returns ``(w_compute, rebuild)``: the transient dequantized copy
    to run the split GEMMs on, and a closure re-dequantizing it from the low-bit
    buffers for backward (via ``_single_weight_dequant_hooks``). One dequantize
    is exactly what the retired shared-R fallback paid per forward; the split
    math on top is what it never did.
    """
    if hasattr(module, "weight_packed"):
        from megatron.bridge.orbit.low_precision.int4 import dequantize_int4

        packed = module.weight_packed
        scale = module.weight_scale
        shape = module.weight_shape

        def rebuild() -> torch.Tensor:
            return dequantize_int4(packed, scale, shape, device=packed.device).to(dtype)

        return rebuild(), rebuild

    quantizer = getattr(module, "weight_quantizer", None)
    weight = getattr(module, "weight", None)
    if quantizer is not None and getattr(weight, "dtype", None) == torch.uint8:
        from megatron.bridge.orbit.low_precision.nvfp4 import NVFP4_AMAX_SCALE, dequantize_nvfp4

        packed = weight
        scale = getattr(quantizer, "_scale", None)
        scale_2 = getattr(quantizer, "_double_scale", None)
        if scale_2 is None:
            amax = getattr(quantizer, "_amax", None)
            if amax is not None:
                scale_2 = amax.to(torch.float32) / NVFP4_AMAX_SCALE
        if scale is None or scale_2 is None:
            raise RuntimeError(
                f"{type(module).__name__}: NVFP4 base carries weight_quantizer but no "
                "_scale/_double_scale (or _amax) buffers to dequantize from"
            )
        shape = getattr(module, "weight_shape", None)
        if shape is None:
            shape = (int(packed.shape[0]), int(packed.shape[1]) * 2)

        def rebuild() -> torch.Tensor:
            return dequantize_nvfp4(packed, scale, scale_2, shape, device=packed.device, dtype=dtype)

        return rebuild(), rebuild

    if _has_dequantized_nvfp4_buffers(module):
        return None

    scale_inv = getattr(module, "weight_scale_inv", None)
    if _is_direct_fp8_runtime_weight(weight, scale_inv):
        from megatron.bridge.orbit.quant.fp8_utils import dequant_fp8

        w_fp8 = weight

        def rebuild() -> torch.Tensor:
            return dequant_fp8(w_fp8, scale_inv, out_dtype=dtype)

        return rebuild(), rebuild

    return None


def _single_weight_dequant_hooks(w_ptr: int, rebuild) -> torch.autograd.graph.saved_tensors_hooks:
    """saved_tensors_hooks that swap one dequantized weight for its rebuild handle.

    ``pack`` recognizes the transient dequantized copy by data pointer and hands
    autograd the closure instead, so the BF16 copy dies with the forward frame;
    ``unpack`` re-dequantizes during backward and restores whatever view autograd
    had saved. Same discipline as OFTLinear's per-format forwards.
    """

    def pack(tensor: torch.Tensor):
        if tensor.data_ptr() == w_ptr:
            return (
                _DEQUANT_HANDLE_TAG,
                rebuild,
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.storage_offset(),
            )
        return tensor

    def unpack(handle):
        if isinstance(handle, tuple) and len(handle) == 5 and handle[0] is _DEQUANT_HANDLE_TAG:
            _, rebuild_fn, saved_shape, saved_stride, saved_storage_offset = handle
            return rebuild_fn().as_strided(saved_shape, saved_stride, saved_storage_offset)
        return handle

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


def _fused_base_linear(to_wrap: nn.Module, x: torch.Tensor, W: torch.Tensor):
    """Adapter-disabled base behavior on a transient dequantized weight."""
    bias = getattr(to_wrap, "bias", None)
    out = F.linear(x, W, None)
    if bias is not None and not getattr(to_wrap, "skip_bias_add", False):
        return out + bias, None
    return out, bias


def _should_treat_linear_fc1_as_unfused(full_name: str) -> bool:
    """Return True when CanonicalOFT must keep linear_fc1 as a single adapter."""

    return full_name.startswith("vision_model.")


TELayerNormColumnParallelLinear, HAVE_TE_LN_COL_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine",
    "TELayerNormColumnParallelLinear",
)


def _oft_fast_path_supported(modules: list[OFTRotationModule]) -> bool:
    """Return True iff every module in the bank is at the configuration the
    rotation-bank fast path handles: no COFT, no block share, no dropout, and
    homogeneous (in_features / block_size / r / input_is_parallel).
    """
    if not modules:
        return False
    first = modules[0]
    dropout_p = getattr(getattr(first, "dropout", None), "p", 0.0)
    if first.coft or first.block_share or dropout_p != 0.0:
        return False
    for module in modules[1:]:
        module_dropout_p = getattr(getattr(module, "dropout", None), "p", 0.0)
        if (
            module.in_features != first.in_features
            or module.block_size != first.block_size
            or module.r != first.r
            or module.coft
            or module.block_share
            or module_dropout_p != 0.0
            or module.input_is_parallel != first.input_is_parallel
        ):
            return False
    return True


def _stack_oft_r_for_tp(modules: list[OFTRotationModule]) -> torch.Tensor:
    """Stack ``oft_r`` from a list of OFTRotationModule, mirroring
    OFTRotationModule.forward's TP collective: when ``input_is_parallel`` is
    False, gradients flow through ``copy_to_tensor_model_parallel_region`` to
    keep TP-replicated parameters in sync.
    """
    stacked = []
    for module in modules:
        if module.input_is_parallel:
            stacked.append(module.oft_r)
        else:
            stacked.append(copy_to_tensor_model_parallel_region(module.oft_r, group=module.tp_group))
    return torch.stack(stacked, dim=0)


def _compute_oft_rotation_bank(modules: list[OFTRotationModule]) -> torch.Tensor:
    """Return ``(num_modules, num_blocks, block_size, block_size)`` rotations
    via one batched Cayley call. All modules must satisfy
    ``_oft_fast_path_supported``."""
    assert _oft_fast_path_supported(modules)
    template = modules[0]
    stacked = _stack_oft_r_for_tp(modules)
    num_modules, num_blocks, n_elements = stacked.shape
    flat = stacked.reshape(num_modules * num_blocks, n_elements)
    R = template._cayley_batch(flat, template.block_size)
    return R.reshape(num_modules, num_blocks, template.block_size, template.block_size)


def _apply_precomputed_oft_rotation_to_x(
    x: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:
    """Apply ``num_slices`` block-diagonal rotations to a shared input ``x``.

    Inputs:
        x:  shape ``(..., K)`` where ``K = num_blocks * block_size``.
        R:  shape ``(num_slices, num_blocks, block_size, block_size)``.

    Returns:
        Stacked rotated activations of shape ``(num_slices, ..., K)``, in the
        same dtype as ``x``. Slot ``i`` equals
        ``einsum("...rk,rkc->...rc", x.view(..., num_blocks, block_size), R[i])``
        reshaped back to ``K``.

    The single-launch einsum replaces ``num_slices`` per-adapter Cayley+rotation
    calls in the dense split forward; the rotation bank that built ``R`` already
    paid one batched Cayley up front via ``_compute_oft_rotation_bank``.
    """
    num_slices, num_blocks, block_size, block_size_ = R.shape
    assert block_size == block_size_, f"R is not square per block: {R.shape}"
    K = num_blocks * block_size
    assert x.shape[-1] == K, f"x last dim {x.shape[-1]} != K={K}"

    leading_shape = x.shape[:-1]
    x_blocked = x.reshape(*leading_shape, num_blocks, block_size)
    out = torch.einsum("...rk,srkc->s...rc", x_blocked, R)
    return out.reshape(num_slices, *leading_shape, K)


def _batched_equal_output_linear_with_bias(
    x_stack: torch.Tensor,
    W_stack: torch.Tensor,
    b_stack: torch.Tensor | None,
) -> torch.Tensor:
    """Run ``num_slices`` linears with the same output size as one batched bmm.

    Inputs:
        x_stack: shape ``(num_slices, *leading, K)`` — rotated activations from
                 ``_apply_precomputed_oft_rotation_to_x``. ``leading`` may have
                 any rank; Megatron passes ``(S, B, K)`` at train time.
        W_stack: shape ``(num_slices, H, K)`` — output slices stacked along dim 0.
        b_stack: shape ``(num_slices, H)`` or ``None``.

    Returns:
        ``(*leading, num_slices * H)`` — equivalent to
        ``torch.cat([F.linear(x_stack[i], W_stack[i], b_stack[i]) for i in range(num_slices)], dim=-1)``.

    Replaces ``num_slices`` ``F.linear`` launches with a single ``torch.bmm``.
    Leading dims are flattened to satisfy ``torch.bmm`` 3-D contract, then
    reshaped back at the end (view-only on contiguous tensors).
    """
    num_slices = x_stack.shape[0]
    if W_stack.shape[0] != num_slices:
        raise ValueError(f"x_stack and W_stack disagree on num_slices: {x_stack.shape[0]} vs {W_stack.shape[0]}")
    K = x_stack.shape[-1]
    leading_shape = x_stack.shape[1:-1]
    H = W_stack.shape[1]
    x_flat = x_stack.reshape(num_slices, -1, K)  # (S, M, K)
    out = torch.bmm(x_flat, W_stack.transpose(1, 2))  # (S, M, H)
    if b_stack is not None:
        out = out + b_stack[:, None, :]
    # Permute (S, M, H) -> (M, S, H) -> (M, S*H) so slice 0 comes first in the
    # output, matching torch.cat(..., dim=-1) on per-slice outputs, then
    # restore the caller's leading dims.
    out = out.permute(1, 0, 2).reshape(-1, num_slices * H)
    return out.reshape(*leading_shape, num_slices * H)


class OFTLinearSplitFC1UpGate(nn.Module):
    """Wraps a fused ``linear_fc1`` (ColumnParallelLinear producing ``[gate; up]``)
    with two independent input rotations applied per output slice.

    Layout: ``W_fc1.shape == (2 * ffn_hidden_size / TP, hidden_size)`` — first
    half gate, second half up. Megatron's ``--swiglu`` convention.
    """

    def __init__(
        self,
        orig_module: nn.Module,
        in_features: int,
        r: int = 0,
        block_size: int = 32,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        module_dropout: float = 0.0,
        model_parallel_config: Any = None,
        input_is_parallel: bool = False,
        is_expert: bool = False,
    ) -> None:
        super().__init__()
        self.to_wrap = orig_module

        def _make_R() -> OFTRotationModule:
            return OFTRotationModule(
                in_features=in_features,
                r=r,
                block_size=block_size,
                coft=coft,
                eps=eps,
                block_share=block_share,
                module_dropout=module_dropout,
                model_parallel_config=model_parallel_config,
                input_is_parallel=input_is_parallel,
                is_expert=is_expert,
            )

        self.adapter_gate = _make_R()
        self.adapter_up = _make_R()
        self._adapter_enabled = True

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        return _split_wrapper_sharded_state_dict(self, prefix, sharded_offsets, metadata)

    def _split_output_weight(self, W: torch.Tensor):
        out_features = W.shape[0]
        assert out_features % 2 == 0, f"linear_fc1 out dim {out_features} must be even"
        half = out_features // 2
        return W[:half], W[half:]

    def _fused_fast_path_supported(self) -> bool:
        """Both gate and up adapters must satisfy the rotation-bank contract:
        same dtype, same block_size, same num_blocks, no per-block share. The
        fast path falls back to the eager loop otherwise.
        """
        return _oft_fast_path_supported([self.adapter_gate, self.adapter_up])

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        # Quantized fused base: dequantize once (the same cost the retired
        # shared-R fallback paid) and run the REAL split math on the BF16 copy,
        # with hooks so the graph keeps only the low-bit rebuild handle.
        dequant = _dequantize_single_weight_base(self.to_wrap, x.dtype)
        if dequant is None:
            if not self._adapter_enabled:
                return self.to_wrap(x, *args, **kwargs)
            return self._forward_with_weight(x, self.to_wrap.weight)
        w_compute, rebuild = dequant
        try:
            with _single_weight_dequant_hooks(w_compute.data_ptr(), rebuild):
                if not self._adapter_enabled:
                    return _fused_base_linear(self.to_wrap, x.contiguous(), w_compute)
                return self._forward_with_weight(x, w_compute)
        finally:
            del w_compute

    def _forward_with_weight(self, x: torch.Tensor, W: torch.Tensor):
        x = x.contiguous()
        bias = getattr(self.to_wrap, "bias", None)

        if self._fused_fast_path_supported():
            R = _compute_oft_rotation_bank([self.adapter_gate, self.adapter_up])
            # Match eager OFTRotationModule.forward dtype contract: rotation runs
            # in oft_r.dtype, then the result is cast back to x.dtype. See
            # oft_layers.py:649-651 and oft_layers.py:673.
            required_dtype = x.dtype
            if R.dtype != x.dtype:
                x_for_einsum = x.to(R.dtype)
            else:
                x_for_einsum = x
            x_stack = _apply_precomputed_oft_rotation_to_x(x_for_einsum, R).to(required_dtype)
            # W is contiguous (2H, K) laid out as [gate; up]; the view is zero-copy
            # because _split_output_weight returns the consecutive row halves.
            assert W.shape[0] % 2 == 0
            W_stack = W.view(2, W.shape[0] // 2, W.shape[1])
            out = _batched_equal_output_linear_with_bias(x_stack, W_stack, None)
        else:
            W_gate, W_up = self._split_output_weight(W)
            x_gate = self.adapter_gate(x)
            x_up = self.adapter_up(x)
            out_gate = F.linear(x_gate, W_gate)
            out_up = F.linear(x_up, W_up)
            out = torch.cat([out_gate, out_up], dim=-1)

        if bias is not None and not getattr(self.to_wrap, "skip_bias_add", False):
            return out + bias, None
        return out, bias

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.to_wrap, name)

    def enable_adapter_layers(self) -> None:
        self._adapter_enabled = True

    def disable_adapter_layers(self) -> None:
        self._adapter_enabled = False


class GroupedOFTRotation(nn.Module):
    """Stacked per-expert OFT rotations for grouped MoE experts.

    Holds ``oft_r`` as a single 3D ``nn.Parameter`` of shape
    ``(num_local_experts, num_blocks, n_elements)`` instead of an
    ``nn.ModuleList`` of per-expert ``OFTRotationModule``s. This makes the train
    forward run one batched Cayley call covering all local experts and lets the
    weight-export side ship a single 3D tensor through the existing
    grouped-3D EP-gather path — mirroring the DSV4-grouped ``w{1,2,3}`` layout.

    Conceptually equivalent to ``nn.ModuleList([OFTRotationModule(...) for _ in
    range(num_local_experts)])`` but stored contiguously and with a single
    distributed-optimizer entry per side per layer.
    """

    def __init__(
        self,
        num_local_experts: int,
        in_features: int,
        r: int = 0,
        block_size: int = 0,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        module_dropout: float = 0.0,
        model_parallel_config: Any = None,
        input_is_parallel: bool = False,
        is_expert: bool = True,
    ) -> None:
        super().__init__()
        if num_local_experts <= 0:
            raise ValueError(f"GroupedOFTRotation requires num_local_experts > 0, got {num_local_experts}")

        # Borrow OFTRotationModule for r/block_size normalization and as the
        # source-of-truth implementation of the Cayley transform. We swap its
        # internal 2D oft_r for our 3D parameter so the existing kernel paths
        # (triton + Neumann fallback) keep working without duplication.
        template = OFTRotationModule(
            in_features=in_features,
            r=r,
            block_size=block_size,
            coft=coft,
            eps=eps,
            block_share=block_share,
            module_dropout=module_dropout,
            model_parallel_config=model_parallel_config,
            input_is_parallel=input_is_parallel,
            is_expert=is_expert,
        )

        # Mirror the public attributes that callers + the export path read.
        self.num_local_experts = num_local_experts
        self.in_features = template.in_features
        self.r = template.r
        self.block_size = template.block_size
        self.coft = template.coft
        self.eps = template.eps
        self.block_share = template.block_share
        self.input_is_parallel = template.input_is_parallel
        self.is_expert = template.is_expert
        self.tp_group = template.tp_group
        self.config = template.config
        self.module_dropout = module_dropout

        # 3D parameter: (E, num_blocks, n_elements). Per-expert oft_r is a
        # contiguous slice along dim 0.
        template_dtype = template.oft_r.dtype
        template_device = template.oft_r.device
        num_blocks, n_elements = template.oft_r.shape
        self.oft_r = nn.Parameter(
            torch.zeros(
                num_local_experts,
                num_blocks,
                n_elements,
                dtype=template_dtype,
                device=template_device,
            )
        )
        if self.is_expert:
            self.oft_r.allreduce = False

        # Keep the upper-triangle indices on the same device as the parameter.
        self.register_buffer("rows", template.rows.clone(), persistent=False)
        self.register_buffer("cols", template.cols.clone(), persistent=False)

        # Delegate Cayley + skew construction to the template. We never read
        # template.oft_r, so drop it from the registered parameters to avoid
        # exposing a phantom adapter to the optimizer or state_dict.
        del template.oft_r
        self._template = template

    def __len__(self) -> int:
        return self.num_local_experts

    def __iter__(self):
        for i in range(self.num_local_experts):
            yield self[i]

    def __getitem__(self, expert_idx: int) -> "_PerExpertOFTRotationView":
        """Return a view exposing ``.oft_r`` for the i-th expert.

        Lets legacy callers that did ``module_list[i].oft_r`` keep reading the
        per-expert slice. Writes via ``view.oft_r.data.copy_(...)`` propagate
        back into the underlying 3D parameter because the slice is a view.
        Calling the view forwards through ``GroupedOFTRotation.forward``.
        """
        if not (0 <= expert_idx < self.num_local_experts):
            raise IndexError(f"expert_idx={expert_idx} out of range for num_local_experts={self.num_local_experts}")
        return _PerExpertOFTRotationView(self, expert_idx)

    def get_oft_r_for_expert(self, expert_idx: int) -> torch.Tensor:
        """Return the 2D ``(num_blocks, n_elements)`` view for a single expert."""
        return self.oft_r[expert_idx]

    def compute_rotation_bank(self) -> torch.Tensor:
        """Return ``(num_local_experts, num_blocks, block_size, block_size)``
        via one batched Cayley call across all local experts.

        Equivalent to stacking ``OFTRotationModule._compute_rotation()`` over
        every local expert but in a single launch — the same code path the
        rotation-bank fast path takes for the dense split FC1.
        """
        if self.input_is_parallel:
            oft_r_parallel = self.oft_r
        else:
            oft_r_parallel = copy_to_tensor_model_parallel_region(self.oft_r, group=self.tp_group)

        if self.coft:
            with torch.no_grad():
                # _project_batch expects (b, n_elements). Reshape, project,
                # reshape back so the COFT clamp acts per-block.
                flat = oft_r_parallel.reshape(-1, oft_r_parallel.shape[-1])
                projected = self._template._project_batch(flat, eps=self.eps)
                oft_r_parallel.copy_(projected.reshape_as(oft_r_parallel))

        E, num_blocks, n_elements = oft_r_parallel.shape
        flat = oft_r_parallel.reshape(E * num_blocks, n_elements)
        R = self._template._cayley_batch(flat, self.block_size)
        return R.reshape(E, num_blocks, self.block_size, self.block_size)

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: Tuple = (), metadata: Optional[Dict] = None):
        """Create the sharded state dict for the stacked per-expert ``oft_r``.

        ``oft_r`` is one ``(num_local_experts, num_blocks, n_elements)`` parameter:
        each EP rank holds a *different*, non-overlapping slice of experts along
        axis 0, not a replica of the same data. Without this override, mcore's
        plain-``nn.Module`` fallback (``sharded_state_dict_default``) marks the
        whole tensor replicated and records its *local* shape as the global
        shape: on save, every EP rank but the elected primary silently drops its
        experts' ``oft_r``; on load, one rank's slice gets broadcast back to
        every rank instead of each rank recovering its own data.

        Mirrors how ``megatron.core.transformer.moe.experts`` computes the
        EP-axis offset for its own per-expert weights
        (``num_global_experts = ep_size * num_local_experts``,
        ``local_expert_offset = ep_rank * num_local_experts``), via
        ``_make_expert_ep_sharded_tensor``: EP offset on the packed expert
        axis, replica tag from the expert-DP group. TEGroupedMLP's own
        per-expert mechanism only augments keys literally named
        ``weight{i}``/``bias{i}`` and rewrites no adapter replica ids, so it
        never reaches ``oft_r``.

        Column-parallel fc1 (the only place ``GroupedOFTRotation`` is used) is
        not ``input_is_parallel``, so the other two axes stay TP-replicated
        here, matching ``OFTRotationModule.sharded_state_dict``'s
        ``tp_axis_map={}`` branch for the same case.

        Unverified at runtime: no EP>1 hardware is available here. The
        pre-existing expert ``replica_id`` rewrite at ``oft_layers.py:655`` for
        the non-grouped path is untouched by this change.
        """
        state_dict = self.state_dict(prefix="", keep_vars=True)
        key = f"{prefix}oft_r"

        if self.is_expert:
            # EP shards the packed local-expert axis 0 (that part was already
            # right when this used make_tp_sharded_tensor_for_checkpoint with
            # the EP group standing in for TP). What that helper got wrong is
            # the replica tag: it defaulted to the dense dp_cp rank, but expert
            # tensors replicate across the expert-DP group. At EP>1 layouts
            # where those groups differ, nonzero-EP-rank slices were marked
            # non-main replicas and keep_only_main_replica saves dropped them.
            sharded_tensor = _make_expert_ep_sharded_tensor(
                state_dict["oft_r"],
                key,
                ep_new_axis=False,
                blocks_local_axis=1,
                blocks_tp_sharded=self.input_is_parallel and not self.block_share,
                sharded_offsets=sharded_offsets,
            )
        else:
            from megatron.core.utils import make_sharded_tensor_for_checkpoint

            sharded_tensor = make_sharded_tensor_for_checkpoint(
                state_dict["oft_r"],
                key,
                prepend_offsets=sharded_offsets,
                tp_group=self.tp_group,
                dp_cp_group=metadata["dp_cp_group"],
            )
        return {key: sharded_tensor}

    def forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """Apply this expert's rotation to ``x``. Eager-loop fallback only —
        the fast path uses ``compute_rotation_bank`` once and the
        ``oft_r_by_expert`` triton kernel for the whole batch.
        """
        required_dtype = x.dtype
        if required_dtype != self.oft_r.dtype:
            x = x.to(self.oft_r.dtype)

        if self.input_is_parallel:
            oft_r = self.oft_r[expert_idx]
        else:
            oft_r = copy_to_tensor_model_parallel_region(self.oft_r[expert_idx], group=self.tp_group)
        # Reuse the template's _cayley_batch which honors triton when available.
        R = self._template._cayley_batch(oft_r, self.block_size)

        rank = self.in_features // self.block_size if self.block_share else self.r
        if self.block_share:
            R = R.repeat(rank, 1, 1)
        orig_shape = x.shape
        x = x.reshape(*orig_shape[:-1], rank, self.block_size)
        x = torch.einsum("...rk, rkc -> ...rc", x, R)
        x = x.reshape(orig_shape)
        return x.to(required_dtype)


class _PerExpertOFTRotationView:
    """Thin per-expert handle over a slice of a ``GroupedOFTRotation``.

    Lets legacy code that did ``module_list[i].oft_r`` keep working after the
    layout flipped to a single 3D parameter. Reads/writes go through a slice of
    the parent's ``oft_r``; calling the view delegates to
    ``GroupedOFTRotation.forward``.
    """

    __slots__ = ("_parent", "_expert_idx")

    def __init__(self, parent: GroupedOFTRotation, expert_idx: int) -> None:
        self._parent = parent
        self._expert_idx = expert_idx

    @property
    def oft_r(self) -> torch.Tensor:
        return self._parent.oft_r[self._expert_idx]

    @property
    def block_size(self) -> int:
        return self._parent.block_size

    @property
    def r(self) -> int:
        return self._parent.r

    @property
    def coft(self) -> bool:
        return self._parent.coft

    @property
    def block_share(self) -> bool:
        return self._parent.block_share

    @property
    def input_is_parallel(self) -> bool:
        return self._parent.input_is_parallel

    @property
    def is_expert(self) -> bool:
        return self._parent.is_expert

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._parent.forward(x, self._expert_idx)


class OFTLinearGroupedSplitFC1UpGate(nn.Module):
    """Grouped expert fused FC1 with independent gate/up rotations per local expert.

    ``forward(x, tokens_per_expert)`` matches Megatron-Core ``GroupedMLP``'s
    call convention; one shared R across gate/up halves is mathematically wrong.
    """

    def __init__(
        self,
        orig_module: nn.Module,
        in_features: int,
        r: int = 0,
        block_size: int = 32,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        module_dropout: float = 0.0,
        model_parallel_config: Any = None,
        input_is_parallel: bool = False,
        is_expert: bool = True,
    ) -> None:
        super().__init__()
        self.to_wrap = orig_module
        self.num_gemms = int(getattr(orig_module, "num_gemms", 0))
        if self.num_gemms <= 0:
            raise ValueError(f"{type(self).__name__} requires a grouped expert module with num_gemms > 0")

        def _make_R() -> GroupedOFTRotation:
            return GroupedOFTRotation(
                num_local_experts=self.num_gemms,
                in_features=in_features,
                r=r,
                block_size=block_size,
                coft=coft,
                eps=eps,
                block_share=block_share,
                module_dropout=module_dropout,
                model_parallel_config=model_parallel_config,
                input_is_parallel=input_is_parallel,
                is_expert=is_expert,
            )

        # Single 3D ``oft_r`` per side: shape (num_local_experts, num_blocks, n_elements).
        # The export path recognizes this layout and ships one tensor per (layer, side)
        # through the existing grouped 3D EP-gather instead of one tensor per expert.
        self.adapter_gate = _make_R()
        self.adapter_up = _make_R()
        self._adapter_enabled = True
        self._te_grouped_half_modules: dict[tuple[Any, ...], nn.Module] = {}

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        return _split_wrapper_sharded_state_dict(self, prefix, sharded_offsets, metadata)

    @staticmethod
    def _normalize_tokens_per_expert(tokens_per_expert: Any) -> list[int]:
        if isinstance(tokens_per_expert, torch.Tensor):
            return [int(v) for v in tokens_per_expert.detach().cpu().tolist()]
        return [int(v) for v in tokens_per_expert]

    @staticmethod
    def _split_output_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out_features = weight.shape[0]
        assert out_features % 2 == 0, f"grouped linear_fc1 out dim {out_features} must be even"
        half = out_features // 2
        return weight[:half], weight[half:]

    def _is_base_int4(self) -> bool:
        """True when the wrapped grouped module carries Kimi-style INT4 triplets.

        Checked per-call rather than cached at ``__init__`` because the INT4
        triplet buffers are registered *after* construction by
        ``register_int4_buffers_after_load_dense`` during checkpoint load.
        """
        return (
            hasattr(self.to_wrap, "weight0_packed")
            and hasattr(self.to_wrap, "weight0_scale")
            and hasattr(self.to_wrap, "weight0_shape")
        )

    def _int4_triplet_for_expert(self, expert_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            getattr(self.to_wrap, f"weight{expert_idx}_packed"),
            getattr(self.to_wrap, f"weight{expert_idx}_scale"),
            getattr(self.to_wrap, f"weight{expert_idx}_shape"),
        )

    def _nvfp4_scale_suffix_for(self, expert_idx: int) -> Optional[str]:
        """Buffer suffix carrying this local expert's NVFP4 scales.

        Direct-checkpoint buffers are usually keyed by local index
        (``weight_scale0``), but EP-sharded checkpoints can retain global expert
        suffixes; map local order onto the sorted numeric suffixes then.
        """
        if hasattr(self.to_wrap, f"weight_scale{expert_idx}") and hasattr(
            self.to_wrap, f"weight_double_scale{expert_idx}"
        ):
            return str(expert_idx)
        suffixes = sorted(
            (
                int(name[len("weight_scale") :])
                for name in dir(self.to_wrap)
                if name.startswith("weight_scale") and name[len("weight_scale") :].isdigit()
            ),
        )
        if len(suffixes) == self.num_gemms and 0 <= expert_idx < len(suffixes):
            suffix = str(suffixes[expert_idx])
            if hasattr(self.to_wrap, f"weight_double_scale{suffix}"):
                return suffix
        return None

    def _grouped_quant_kind(self) -> str:
        """Which quantized representation the grouped base carries.

        ``bf16`` means the expert weights are directly usable; every other kind
        routes through ``_forward_dequant_split_eager``. Unknown quantized
        representations raise rather than silently running BF16 math on packed
        payloads (the old ``_assert_unquantized`` let direct-FP8 grouped slip
        through exactly that way).
        """
        # NVFP4 direct-checkpoint buffers: unique weight_double_scale{N} marker.
        if self._nvfp4_scale_suffix_for(0) is not None and (
            hasattr(self.to_wrap, "weight0_w_packed") or hasattr(self.to_wrap, "weight0_packed")
        ):
            return "nvfp4_buffers"
        if self._is_base_int4():
            return "int4"
        first_weight = getattr(self.to_wrap, "weight0", None)
        if (
            getattr(self.to_wrap, "weight_quantizer", None) is not None
            and getattr(first_weight, "dtype", None) == torch.uint8
        ):
            return "nvfp4_modelopt"
        if _is_direct_fp8_runtime_weight(first_weight, getattr(self.to_wrap, "weight0_scale_inv", None)):
            return "fp8_direct"
        if getattr(self.to_wrap, "weight_quantizer", None) is not None:
            raise RuntimeError(
                f"{type(self).__name__}: unrecognized ModelOpt quantized representation on "
                "grouped FC1 (weight_quantizer present but weight0 is not uint8-packed NVFP4)."
            )
        if getattr(first_weight, "dtype", None) == torch.uint8:
            raise RuntimeError(
                f"{type(self).__name__}: weight0 is uint8 but carries no recognized NVFP4/INT4 companion buffers."
            )
        return "bf16"

    def _quantizer_buffer(self, attr: str, expert_idx: int) -> Optional[torch.Tensor]:
        """Fetch a per-expert ModelOpt quantizer buffer (suffixed or stacked)."""
        quantizer = self.to_wrap.weight_quantizer
        value = getattr(quantizer, f"{attr}{expert_idx}", None)
        if value is not None:
            return value
        value = getattr(quantizer, attr, None)
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == self.num_gemms:
            return value[expert_idx]
        return value

    def _dequant_expert_halves(self, kind: str, expert_idx: int, dtype: torch.dtype, device):
        """Dequantized (gate_w, up_w) for one expert plus their rebuild closures.

        Returns ``(gate_w, up_w, entries)`` where ``entries`` maps each half's
        ``data_ptr`` to a rebuild closure producing the tensor autograd saved a
        view of. BOTH halves are registered: the up half is a nonzero-offset
        view, and a pack that only knows the base pointer saves the raw view --
        which pins the entire fused BF16 buffer (the INT4 path leaked exactly
        this before the generalization).
        """
        if kind == "int4":
            from megatron.bridge.orbit.low_precision.int4 import dequantize_int4

            packed, scale, shape = self._int4_triplet_for_expert(expert_idx)

            def rebuild() -> torch.Tensor:
                return dequantize_int4(packed, scale, shape, device=packed.device).to(dtype)

            fused = rebuild()
            gate_w, up_w = self._split_output_weight(fused)
            return gate_w, up_w, {gate_w.data_ptr(): rebuild, up_w.data_ptr(): rebuild}

        if kind == "nvfp4_buffers":
            from megatron.bridge.orbit.low_precision.nvfp4 import dequantize_nvfp4

            suffix = self._nvfp4_scale_suffix_for(expert_idx)
            if suffix is None:
                raise RuntimeError(
                    f"{type(self).__name__}: missing weight_scale/weight_double_scale for expert {expert_idx}"
                )
            scale = getattr(self.to_wrap, f"weight_scale{suffix}")
            double_scale = getattr(self.to_wrap, f"weight_double_scale{suffix}")
            w_half = getattr(self.to_wrap, f"weight{expert_idx}_w_packed", None)
            v_half = getattr(self.to_wrap, f"weight{expert_idx}_v_packed", None)
            if w_half is not None and v_half is not None:
                # Gate/up already stored as separate packed halves: dequantize
                # each against its slice of the fused scale -- no fused buffer,
                # no row splitting, and forward peak is one half at a time.
                half_out = int(w_half.shape[0])
                local_in = int(w_half.shape[1]) * 2
                gate_scale, up_scale = scale[:half_out], scale[half_out : 2 * half_out]

                def rebuild_gate() -> torch.Tensor:
                    return dequantize_nvfp4(
                        w_half, gate_scale, double_scale, (half_out, local_in), device=w_half.device, dtype=dtype
                    )

                def rebuild_up() -> torch.Tensor:
                    return dequantize_nvfp4(
                        v_half, up_scale, double_scale, (half_out, local_in), device=v_half.device, dtype=dtype
                    )

                gate_w, up_w = rebuild_gate(), rebuild_up()
                return gate_w, up_w, {gate_w.data_ptr(): rebuild_gate, up_w.data_ptr(): rebuild_up}

            packed = getattr(self.to_wrap, f"weight{expert_idx}_packed")
            local_out, local_in = int(packed.shape[0]), int(packed.shape[1]) * 2

            def rebuild() -> torch.Tensor:
                return dequantize_nvfp4(
                    packed, scale, double_scale, (local_out, local_in), device=packed.device, dtype=dtype
                )

            fused = rebuild()
            gate_w, up_w = self._split_output_weight(fused)
            return gate_w, up_w, {gate_w.data_ptr(): rebuild, up_w.data_ptr(): rebuild}

        if kind == "nvfp4_modelopt":
            from megatron.bridge.orbit.low_precision.nvfp4 import NVFP4_AMAX_SCALE, dequantize_nvfp4

            packed = getattr(self.to_wrap, f"weight{expert_idx}")
            scale = self._quantizer_buffer("_scale", expert_idx)
            double_scale = self._quantizer_buffer("_double_scale", expert_idx)
            if double_scale is None:
                amax = self._quantizer_buffer("_amax", expert_idx)
                if amax is not None:
                    double_scale = amax.to(torch.float32) / NVFP4_AMAX_SCALE
            if scale is None or double_scale is None:
                raise RuntimeError(
                    f"{type(self).__name__}: ModelOpt NVFP4 expert {expert_idx} lacks _scale/_double_scale buffers"
                )
            local_out, local_in = int(packed.shape[0]), int(packed.shape[1]) * 2

            def rebuild() -> torch.Tensor:
                return dequantize_nvfp4(
                    packed, scale, double_scale, (local_out, local_in), device=packed.device, dtype=dtype
                )

            fused = rebuild()
            gate_w, up_w = self._split_output_weight(fused)
            return gate_w, up_w, {gate_w.data_ptr(): rebuild, up_w.data_ptr(): rebuild}

        if kind == "fp8_direct":
            from megatron.bridge.orbit.quant.fp8_utils import dequant_fp8

            w_fp8 = getattr(self.to_wrap, f"weight{expert_idx}")
            scale_inv = getattr(self.to_wrap, f"weight{expert_idx}_scale_inv")

            def rebuild() -> torch.Tensor:
                return dequant_fp8(w_fp8, scale_inv, out_dtype=dtype)

            fused = rebuild()
            gate_w, up_w = self._split_output_weight(fused)
            return gate_w, up_w, {gate_w.data_ptr(): rebuild, up_w.data_ptr(): rebuild}

        raise AssertionError(kind)  # pragma: no cover

    def _forward_dequant_split_eager(
        self,
        kind: str,
        gate_x: torch.Tensor,
        up_x: torch.Tensor,
        tokens_per_expert: list,
    ) -> tuple[torch.Tensor, None]:
        """Per-expert dequant + split F.linear for every quantized grouped kind.

        ``gate_x`` and ``up_x`` are the rotated activations the caller already
        produced -- quantization is orthogonal to the rotation, so we never
        re-rotate here. Autograd hooks store rebuild closures instead of the
        BF16 halves, and each expert's registrations are popped right after its
        GEMMs, so a freed address can never be mistaken for a live weight and
        forward peak stays at one expert's weights.
        """
        ptr_to_rebuild: dict[int, Any] = {}

        def pack(tensor):
            rebuild = ptr_to_rebuild.get(tensor.data_ptr())
            if rebuild is not None:
                return (
                    _DEQUANT_HANDLE_TAG,
                    rebuild,
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    tensor.storage_offset(),
                )
            return tensor

        def unpack(handle):
            if isinstance(handle, tuple) and len(handle) == 5 and handle[0] is _DEQUANT_HANDLE_TAG:
                _, rebuild_fn, saved_shape, saved_stride, saved_offset = handle
                return rebuild_fn().as_strided(saved_shape, saved_stride, saved_offset)
            return handle

        outputs: list[torch.Tensor] = []
        offset = 0
        first_out_features: Optional[int] = None
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            for expert_idx, token_count in enumerate(tokens_per_expert):
                token_count = int(token_count)
                if token_count <= 0:
                    continue
                gate_chunk = gate_x[offset : offset + token_count]
                up_chunk = up_x[offset : offset + token_count]
                offset += token_count

                gate_w, up_w, entries = self._dequant_expert_halves(kind, expert_idx, gate_x.dtype, gate_x.device)
                if first_out_features is None:
                    first_out_features = int(gate_w.shape[0]) + int(up_w.shape[0])
                ptr_to_rebuild.update(entries)
                try:
                    gate_bias = self._bias_for(f"bias_gate{expert_idx}")
                    up_bias = self._bias_for(f"bias_up{expert_idx}")
                    outputs.append(
                        torch.cat(
                            [
                                F.linear(gate_chunk, gate_w, gate_bias),
                                F.linear(up_chunk, up_w, up_bias),
                            ],
                            dim=-1,
                        )
                    )
                finally:
                    for ptr in entries:
                        ptr_to_rebuild.pop(ptr, None)
                    del gate_w, up_w

        if not outputs:
            out_features = first_out_features or getattr(self.to_wrap, "out_features", 0)
            return gate_x.new_empty((0, out_features)), None
        return torch.cat(outputs, dim=0), None

    def _bias_for(self, name: str) -> torch.Tensor | None:
        return getattr(self.to_wrap, name, None)

    def _grouped_base_weights_require_grad(self) -> bool:
        """True if any base expert weight is trainable — disables the fast path
        because the ``.data`` aliasing in ``_bind_te_grouped_half_weights`` would
        bypass autograd routing for that param."""
        for expert_idx in range(self.num_gemms):
            weight = getattr(self.to_wrap, f"weight{expert_idx}", None)
            if isinstance(weight, torch.Tensor) and weight.requires_grad:
                return True
        return False

    def _has_active_split_bias(self) -> bool:
        for expert_idx in range(self.num_gemms):
            if self._bias_for(f"bias_gate{expert_idx}") is not None:
                return True
            if self._bias_for(f"bias_up{expert_idx}") is not None:
                return True
        return False

    def _te_grouped_half_cls_and_mode(self):
        if not torch.cuda.is_available():
            return None, None

        from megatron.bridge.peft.utils import (
            HAVE_TE_COL_GRP_LINEAR,
            HAVE_TE_ROW_GRP_LINEAR,
            TEColumnParallelGroupedLinear,
            TERowParallelGroupedLinear,
        )

        if HAVE_TE_COL_GRP_LINEAR and isinstance(self.to_wrap, TEColumnParallelGroupedLinear):
            return TEColumnParallelGroupedLinear, "column"
        if HAVE_TE_ROW_GRP_LINEAR and isinstance(self.to_wrap, TERowParallelGroupedLinear):
            return TERowParallelGroupedLinear, "row"
        return None, None

    def _te_grouped_tp_size(self) -> int:
        tp_group = getattr(self.to_wrap, "_tp_group", None)
        if tp_group is None:
            return 1
        try:
            from megatron.core.utils import get_pg_size

            return max(1, int(get_pg_size(tp_group)))
        except Exception:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                try:
                    return max(1, int(torch.distributed.get_world_size(group=tp_group)))
                except Exception:
                    return 1
            return 1

    def _get_or_create_te_grouped_half_module(
        self,
        *,
        side: str,
        local_input_size: int,
        local_output_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Module | None:
        chunk_cls, parallel_mode = self._te_grouped_half_cls_and_mode()
        if chunk_cls is None or parallel_mode is None:
            return None

        cache_key = (
            side,
            local_input_size,
            local_output_size,
            device.type,
            device.index,
            str(dtype),
        )
        cached = self._te_grouped_half_modules.get(cache_key)
        if cached is not None:
            cached.train(self.training)
            return cached

        input_size = local_input_size
        output_size = local_output_size
        tp_size = self._te_grouped_tp_size()
        # explicit_expert_comm + TP > 1 expects the *global* per-side dim, not the per-rank dim.
        if getattr(self.to_wrap, "explicit_expert_comm", False) and tp_size > 1:
            if parallel_mode == "column":
                output_size *= tp_size
            elif parallel_mode == "row":
                input_size *= tp_size

        module = chunk_cls(
            num_gemms=self.num_gemms,
            input_size=input_size,
            output_size=output_size,
            config=self.to_wrap.config,
            init_method=lambda w: None,
            bias=False,
            skip_bias_add=False,
            is_expert=True,
            pg_collection=getattr(self.to_wrap, "_pg_collection", None),
        )
        _clear_disabled_bias_parameters(module)
        has_meta_tensors = any(p.device.type == "meta" for p in module.parameters())
        if not has_meta_tensors:
            has_meta_tensors = any(b.device.type == "meta" for b in module.buffers())
        if has_meta_tensors:
            to_empty_if_meta_device(module, device=device)
            module.to(dtype=dtype)
        else:
            module.to(device=device, dtype=dtype)
        module.train(self.training)
        # TE caches transpose + first-microbatch state across calls; the .data
        # alias rotates the underlying weight every forward, so caches go stale.
        if hasattr(module, "disable_parameter_transpose_cache"):
            module.disable_parameter_transpose_cache = True
        if hasattr(module, "is_first_microbatch"):
            module.is_first_microbatch = False
        for param in module.parameters():
            param.requires_grad_(False)

        self._te_grouped_half_modules[cache_key] = module
        logger.debug(
            "Enabled TE grouped split-OFT half GEMM for %s side=%s num_gemms=%d input=%d output=%d device=%s dtype=%s",
            type(self.to_wrap).__name__,
            side,
            self.num_gemms,
            local_input_size,
            local_output_size,
            device,
            dtype,
        )
        return module

    def _bind_te_grouped_half_weights(self, module: nn.Module, side: str) -> None:
        """Alias the gate or up half of each base ``weight{i}`` into the cached
        module's parameters. No copy — the ``.data`` write is a view assignment."""
        if side not in {"gate", "up"}:
            raise ValueError(f"side must be 'gate' or 'up', got {side!r}")
        for expert_idx in range(self.num_gemms):
            weight = getattr(self.to_wrap, f"weight{expert_idx}")
            gate_w, up_w = self._split_output_weight(weight)
            source = gate_w if side == "gate" else up_w
            target = getattr(module, f"weight{expert_idx}")
            target.data = source.to(device=target.device, dtype=target.dtype, copy=False)

    def _can_use_te_grouped_half_gemm(
        self,
        gate_x: torch.Tensor,
        up_x: torch.Tensor,
    ) -> bool:
        if not gate_x.is_cuda or not up_x.is_cuda:
            return False
        if self._grouped_base_weights_require_grad():
            return False
        if self._has_active_split_bias():
            return False
        if _module_bias_enabled(self.to_wrap):
            return False
        chunk_cls, parallel_mode = self._te_grouped_half_cls_and_mode()
        return chunk_cls is not None and parallel_mode is not None

    def _forward_te_grouped_half_gemm(
        self,
        gate_x: torch.Tensor,
        up_x: torch.Tensor,
        tokens_per_expert: list[int],
    ) -> tuple[torch.Tensor, None]:
        first_weight = getattr(self.to_wrap, "weight0")
        gate_w, _ = self._split_output_weight(first_weight)
        local_output_size = gate_w.shape[0]
        local_input_size = gate_x.shape[-1]

        gate_module = self._get_or_create_te_grouped_half_module(
            side="gate",
            local_input_size=local_input_size,
            local_output_size=local_output_size,
            device=gate_x.device,
            dtype=gate_x.dtype,
        )
        up_module = self._get_or_create_te_grouped_half_module(
            side="up",
            local_input_size=local_input_size,
            local_output_size=local_output_size,
            device=up_x.device,
            dtype=up_x.dtype,
        )
        if gate_module is None or up_module is None:
            raise RuntimeError("TE grouped half-GEMM modules are unavailable")

        self._bind_te_grouped_half_weights(gate_module, "gate")
        self._bind_te_grouped_half_weights(up_module, "up")
        gate_out, gate_bias = gate_module(gate_x, tokens_per_expert)
        up_out, up_bias = up_module(up_x, tokens_per_expert)
        if gate_bias is not None or up_bias is not None:
            raise RuntimeError("TE grouped split-OFT half GEMM unexpectedly returned bias")
        return torch.cat([gate_out, up_out], dim=-1), None

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        quant_kind = self._grouped_quant_kind()
        if len(args) == 0:
            raise ValueError(f"{type(self).__name__} requires tokens_per_expert as the first positional argument")
        tokens_per_expert = self._normalize_tokens_per_expert(args[0])
        if len(tokens_per_expert) != self.num_gemms:
            raise ValueError(f"Expected {self.num_gemms} token splits for grouped FC1, got {len(tokens_per_expert)}")
        if sum(tokens_per_expert) != x.shape[0]:
            raise ValueError(
                f"tokens_per_expert sums to {sum(tokens_per_expert)}, but grouped FC1 input has {x.shape[0]} rows"
            )

        if not self._adapter_enabled:
            gate_x = x
            up_x = x
        else:
            can_segment = (
                oft_r_by_expert is not None
                and x.is_cuda
                and not self.adapter_gate.coft
                and not self.adapter_gate.block_share
                and self.adapter_gate.module_dropout == 0.0
                and not self.adapter_up.coft
                and not self.adapter_up.block_share
                and self.adapter_up.module_dropout == 0.0
            )
            if can_segment:
                counts = torch.tensor(tokens_per_expert, device=x.device, dtype=torch.int64)
                R_gate = self.adapter_gate.compute_rotation_bank()
                R_up = self.adapter_up.compute_rotation_bank()
                x_in = x.contiguous()
                if x_in.dtype != R_gate.dtype:
                    x_in = x_in.to(R_gate.dtype)
                gate_x = oft_r_by_expert(x_in, R_gate, counts)
                up_x = oft_r_by_expert(x_in, R_up, counts)
                if gate_x.dtype != x.dtype:
                    gate_x = gate_x.to(x.dtype)
                    up_x = up_x.to(x.dtype)
            else:
                gate_chunks: list[torch.Tensor] = []
                up_chunks: list[torch.Tensor] = []
                offset = 0
                for expert_idx, token_count in enumerate(tokens_per_expert):
                    token_count = int(token_count)
                    if token_count <= 0:
                        continue
                    chunk = x[offset : offset + token_count].contiguous()
                    offset += token_count
                    gate_chunks.append(self.adapter_gate(chunk, expert_idx))
                    up_chunks.append(self.adapter_up(chunk, expert_idx))
                gate_x = torch.cat(gate_chunks, dim=0) if gate_chunks else x.new_empty(x.shape)
                up_x = torch.cat(up_chunks, dim=0) if up_chunks else x.new_empty(x.shape)

        if quant_kind != "bf16":
            return self._forward_dequant_split_eager(quant_kind, gate_x, up_x, tokens_per_expert)

        if self._can_use_te_grouped_half_gemm(gate_x, up_x):
            try:
                return self._forward_te_grouped_half_gemm(gate_x, up_x, tokens_per_expert)
            except Exception as exc:
                logger.warning(
                    "Falling back to eager grouped split-OFT GEMM for %s because TE grouped half-GEMM failed: %s",
                    type(self.to_wrap).__name__,
                    exc,
                )

        outputs: list[torch.Tensor] = []
        offset = 0
        for expert_idx, token_count in enumerate(tokens_per_expert):
            token_count = int(token_count)
            if token_count <= 0:
                continue
            gate_chunk = gate_x[offset : offset + token_count]
            up_chunk = up_x[offset : offset + token_count]
            offset += token_count

            weight = getattr(self.to_wrap, f"weight{expert_idx}")
            gate_w, up_w = self._split_output_weight(weight)
            gate_bias = self._bias_for(f"bias_gate{expert_idx}")
            up_bias = self._bias_for(f"bias_up{expert_idx}")

            outputs.append(
                torch.cat(
                    [
                        F.linear(gate_chunk.to(gate_w.dtype), gate_w, gate_bias),
                        F.linear(up_chunk.to(up_w.dtype), up_w, up_bias),
                    ],
                    dim=-1,
                )
            )

        if not outputs:
            out_features = getattr(self.to_wrap, "out_features", 0)
            return x.new_empty((0, out_features)), None
        return torch.cat(outputs, dim=0), None

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.to_wrap, name)

    def enable_adapter_layers(self) -> None:
        self._adapter_enabled = True

    def disable_adapter_layers(self) -> None:
        self._adapter_enabled = False


class OFTLinearSplitQKV(nn.Module):
    """Wraps a fused ``linear_qkv`` with three independent input rotations applied
    per Q/K/V slice in Megatron's GQA-interleaved layout."""

    def __init__(
        self,
        orig_module: nn.Module,
        in_features: int,
        provider: Any,
        r: int = 0,
        block_size: int = 32,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        module_dropout: float = 0.0,
        model_parallel_config: Any = None,
        input_is_parallel: bool = False,
        is_expert: bool = False,
    ) -> None:
        super().__init__()
        self.to_wrap = orig_module
        self._provider = provider

        def _make_R() -> OFTRotationModule:
            return OFTRotationModule(
                in_features=in_features,
                r=r,
                block_size=block_size,
                coft=coft,
                eps=eps,
                block_share=block_share,
                module_dropout=module_dropout,
                model_parallel_config=model_parallel_config,
                input_is_parallel=input_is_parallel,
                is_expert=is_expert,
            )

        self.adapter_q = _make_R()
        self.adapter_k = _make_R()
        self.adapter_v = _make_R()
        self._adapter_enabled = True

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        return _split_wrapper_sharded_state_dict(self, prefix, sharded_offsets, metadata)

    def _packed_qkv_layout(self, packed_dim: int) -> tuple[int, int, int]:
        """Infer the local packed-QKV layout from the tensor's packed width.

        Megatron stores ``linear_qkv`` output rows sharded by tensor parallelism.
        On Qwen3-30B-A3B with TP=4, for example, a local shard contains one GQA
        group (8 Q heads + K + V), not all four global GQA groups.  The converter
        helper splits global tensors, so CanonicalOFT needs a local split here.
        """
        cfg = self._provider
        head_num = cfg.num_attention_heads
        num_query_groups = cfg.num_query_groups
        head_size = cfg.kv_channels or (cfg.hidden_size // head_num)
        heads_per_group = head_num // num_query_groups

        if packed_dim % head_size != 0:
            raise ValueError(f"linear_qkv packed dim {packed_dim} must be divisible by head_size={head_size}")

        has_output_gate = getattr(cfg, "attention_output_gate", False)
        units_per_group = (2 * heads_per_group + 2) if has_output_gate else (heads_per_group + 2)
        packed_units = packed_dim // head_size
        if packed_units % units_per_group != 0:
            raise ValueError(
                f"Cannot infer local QKV layout: packed_units={packed_units} is not divisible by "
                f"units_per_group={units_per_group} (heads_per_group={heads_per_group}, "
                f"num_query_groups={num_query_groups})."
            )

        local_query_groups = packed_units // units_per_group
        if local_query_groups <= 0 or local_query_groups > num_query_groups:
            raise ValueError(
                f"Invalid local QKV group count {local_query_groups}; global num_query_groups={num_query_groups}"
            )

        return local_query_groups, heads_per_group, head_size

    def _split_qkv_weight(self, W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_query_groups, heads_per_group, head_size = self._packed_qkv_layout(W.shape[0])
        in_features = W.shape[1]
        has_output_gate = getattr(self._provider, "attention_output_gate", False)
        if has_output_gate:
            raise NotImplementedError("CanonicalOFT split-QKV does not yet support attention_output_gate")

        qkv = W.reshape(local_query_groups, heads_per_group + 2, head_size, in_features)
        q = qkv[:, :heads_per_group].reshape(-1, in_features)
        k = qkv[:, heads_per_group : heads_per_group + 1].reshape(-1, in_features)
        v = qkv[:, heads_per_group + 1 : heads_per_group + 2].reshape(-1, in_features)
        return q, k, v

    def _interleave_qkv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Reassemble [Q,K,V] in Megatron's per-query-group interleaved layout."""
        cfg = self._provider
        head_num = cfg.num_attention_heads
        num_query_groups = cfg.num_query_groups
        head_size = cfg.kv_channels or (cfg.hidden_size // head_num)
        heads_per_group = head_num // num_query_groups

        local_head_num = q.shape[-1] // head_size
        local_query_groups = k.shape[-1] // head_size
        if (
            q.shape[-1] % head_size != 0
            or k.shape[-1] % head_size != 0
            or v.shape[-1] % head_size != 0
            or v.shape[-1] != k.shape[-1]
            or local_query_groups <= 0
            or local_head_num != local_query_groups * heads_per_group
        ):
            raise ValueError(
                f"Cannot interleave local QKV outputs: q={tuple(q.shape)}, k={tuple(k.shape)}, "
                f"v={tuple(v.shape)}, head_size={head_size}, heads_per_group={heads_per_group}"
            )

        leading_shape = q.shape[:-1]
        q = q.reshape(-1, local_query_groups, heads_per_group, head_size)
        k = k.reshape(-1, local_query_groups, 1, head_size)
        v = v.reshape(-1, local_query_groups, 1, head_size)
        return torch.cat([q, k, v], dim=2).reshape(*leading_shape, -1)

    def _fused_fast_path_supported(self) -> bool:
        """Q/K/V rotation banks must agree on dtype, block_size, num_blocks."""
        return _oft_fast_path_supported([self.adapter_q, self.adapter_k, self.adapter_v])

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        # Quantized fused base: dequantize once (the same cost the retired
        # shared-R fallback paid) and run the REAL split math on the BF16 copy,
        # with hooks so the graph keeps only the low-bit rebuild handle.
        dequant = _dequantize_single_weight_base(self.to_wrap, x.dtype)
        if dequant is None:
            if not self._adapter_enabled:
                return self.to_wrap(x, *args, **kwargs)
            return self._forward_with_weight(x, self.to_wrap.weight)
        w_compute, rebuild = dequant
        try:
            with _single_weight_dequant_hooks(w_compute.data_ptr(), rebuild):
                if not self._adapter_enabled:
                    return _fused_base_linear(self.to_wrap, x.contiguous(), w_compute)
                return self._forward_with_weight(x, w_compute)
        finally:
            del w_compute

    def _forward_with_weight(self, x: torch.Tensor, W: torch.Tensor):
        W_q, W_k, W_v = self._split_qkv_weight(W)

        x = x.contiguous()
        bias = getattr(self.to_wrap, "bias", None)

        if self._fused_fast_path_supported():
            R = _compute_oft_rotation_bank([self.adapter_q, self.adapter_k, self.adapter_v])
            # Match eager OFTRotationModule.forward dtype contract: rotation runs
            # in oft_r.dtype, then the result is cast back to x.dtype.
            required_dtype = x.dtype
            if R.dtype != x.dtype:
                x_for_einsum = x.to(R.dtype)
            else:
                x_for_einsum = x
            x_stack = _apply_precomputed_oft_rotation_to_x(x_for_einsum, R).to(required_dtype)
            # x_stack: (3, M, K). Slot 0 is Q (Q_out != K_out under GQA); slots
            # 1-2 are K and V which share the per-group KV dim. W_k and W_v are
            # *not* contiguous halves of W (they come from _split_qkv_weight's
            # reshape-and-slice over the per-group-interleaved layout), so the
            # KV bmm must use torch.stack — there is no zero-copy view path here.
            out_q = F.linear(x_stack[0], W_q)
            if W_k.shape[0] == W_v.shape[0]:
                W_kv_stack = torch.stack([W_k, W_v], dim=0)
                kv_out = _batched_equal_output_linear_with_bias(x_stack[1:3], W_kv_stack, None)
                out_k, out_v = kv_out.chunk(2, dim=-1)
            else:
                # Defensive fallback: should not happen under canonical Megatron
                # GQA layouts, but preserves correctness if a non-GQA exotic
                # config emits unequal K and V output sizes.
                out_k = F.linear(x_stack[1], W_k)
                out_v = F.linear(x_stack[2], W_v)
            out = self._interleave_qkv(out_q, out_k, out_v)
        else:
            x_q = self.adapter_q(x)
            x_k = self.adapter_k(x)
            x_v = self.adapter_v(x)
            out_q = F.linear(x_q, W_q)
            out_k = F.linear(x_k, W_k)
            out_v = F.linear(x_v, W_v)
            out = self._interleave_qkv(out_q, out_k, out_v)

        if bias is not None and not getattr(self.to_wrap, "skip_bias_add", False):
            return out + bias, None
        return out, bias

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.to_wrap, name)

    def enable_adapter_layers(self) -> None:
        self._adapter_enabled = True

    def disable_adapter_layers(self) -> None:
        self._adapter_enabled = False


class _SplitLNCanonicalOFTQKV(nn.Module):
    """LN -> [R_q, R_k, R_v] -> split QKV GEMM -> per-group interleave.

    Reuses ``_SplitLNOFTLinear._apply_norm`` for the LN step (shares the fused
    module's LN weight, no copy) and delegates split-Q/K/V forward to
    ``OFTLinearSplitQKV``.
    """

    def __init__(self, orig_module: nn.Module, qkv_wrapper: "OFTLinearSplitQKV") -> None:
        super().__init__()
        self._orig_module = orig_module
        self._qkv = qkv_wrapper
        self._ln_ref = _SplitLNOFTLinear(orig_module, adapter=nn.Identity())

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        ln_out = self._ln_ref._apply_norm(x)
        if getattr(self._orig_module, "sequence_parallel", False):
            ln_out = gather_from_sequence_parallel_region(ln_out, tensor_parallel_output_grad=True)
        ln_out = ln_out.contiguous()
        return self._qkv(ln_out)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        # Without this override, __getattr__ resolves `sharded_state_dict` to
        # self._orig_module's own bound method (nn.Module has no such attribute
        # to short-circuit the lookup first), silently dropping self._qkv's
        # oft_r adapters from every checkpoint. self._qkv.to_wrap is the same
        # orig_module instance, so delegating to it covers both the base
        # weight/bias/LN weight and the three Q/K/V rotations.
        return self._qkv.sharded_state_dict(prefix, sharded_offsets, metadata)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._orig_module, name)

    def enable_adapter_layers(self) -> None:
        self._qkv.enable_adapter_layers()

    def disable_adapter_layers(self) -> None:
        self._qkv.disable_adapter_layers()


class _SplitLNCanonicalOFTFC1(nn.Module):
    """LN -> [R_gate, R_up] -> split FC1 GEMM -> concat."""

    def __init__(self, orig_module: nn.Module, fc1_wrapper: "OFTLinearSplitFC1UpGate") -> None:
        super().__init__()
        self._orig_module = orig_module
        self._fc1 = fc1_wrapper
        self._ln_ref = _SplitLNOFTLinear(orig_module, adapter=nn.Identity())

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        ln_out = self._ln_ref._apply_norm(x)
        if getattr(self._orig_module, "sequence_parallel", False):
            ln_out = gather_from_sequence_parallel_region(ln_out, tensor_parallel_output_grad=True)
        ln_out = ln_out.contiguous()
        return self._fc1(ln_out)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        # See _SplitLNCanonicalOFTQKV.sharded_state_dict: without this override
        # __getattr__ silently resolves to self._orig_module's own method,
        # dropping self._fc1's oft_r adapters from every checkpoint.
        return self._fc1.sharded_state_dict(prefix, sharded_offsets, metadata)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._orig_module, name)

    def enable_adapter_layers(self) -> None:
        self._fc1.enable_adapter_layers()

    def disable_adapter_layers(self) -> None:
        self._fc1.disable_adapter_layers()


# Fused Megatron leaf -> the canonical split suffixes that replace it. This is the
# inverse of ``CanonicalOFT._SPLIT_SUFFIX_TO_FUSED``, kept at module level so a
# launcher can translate a legacy target list without building the PEFT object.
_FUSED_TO_SPLIT_SUFFIXES: Dict[str, Tuple[str, ...]] = {
    "linear_qkv": ("linear_q", "linear_k", "linear_v"),
    "linear_fc1": ("linear_fc1_gate", "linear_fc1_up"),
}


def canonical_target_modules(target_modules: Iterable[str]) -> List[str]:
    """Translate a legacy ``OFT`` target list into ``CanonicalOFT`` split names.

    ``CanonicalOFT.__post_init__`` rejects the fused leaves ``linear_qkv`` and
    ``linear_fc1``: a single input rotation on a fused projection forces Q/K/V
    (and gate/up) to share one rotation, which is the semantics CanonicalOFT
    exists to replace. Each fused name expands to its split siblings, keeping
    any wildcard prefix intact::

        ["linear_qkv", "linear_proj"] -> ["linear_q", "linear_k", "linear_v", "linear_proj"]
        ["*.layers.0.*.linear_fc1"]   -> ["*.layers.0.*.linear_fc1_gate",
                                          "*.layers.0.*.linear_fc1_up"]

    Names that are already split, and names that are not fused leaves at all
    (``linear_proj``, ``linear_fc2``, Kimi's MLA projections), pass through
    unchanged. Order is preserved and duplicates are dropped.
    """
    expanded: List[str] = []
    for target in target_modules:
        for fused, split_suffixes in _FUSED_TO_SPLIT_SUFFIXES.items():
            if target.endswith(fused):
                prefix = target[: -len(fused)]
                expanded.extend(prefix + suffix for suffix in split_suffixes)
                break
        else:
            expanded.append(target)
    # dict.fromkeys de-duplicates while preserving first-seen order.
    return list(dict.fromkeys(expanded))


@dataclass
class CanonicalOFT(OrbitPEFTMixin, PEFT, ModuleMatcher):
    """OFT with split Q/K/V and split gate/up rotations on fused Megatron linears.

    Parallel to ``CanonicalLoRA``: independent rotations are attached to each of
    Q/K/V (for the fused ``linear_qkv``) and to each of gate/up (for the fused
    ``linear_fc1``). ``linear_proj`` and dense ``linear_fc2`` use plain
    ``OFTLinear`` with a single rotation.

    Grouped MoE experts: ``experts.linear_fc1`` gets *per-expert* gate/up
    rotations (``OFTLinearGroupedSplitFC1UpGate`` over ``GroupedOFTRotation``),
    while ``experts.linear_fc2`` uses a single rotation shared by every local
    expert. This asymmetry is a known limitation, present since the original
    implementation: fc1's ``oft_r`` is replicated, whereas fc2 is RowParallel so
    its blocks are TP-sharded, which makes per-expert 3D ``oft_r`` there
    materially harder.

    Args:
        target_modules: HF-style split names. Must NOT contain ``linear_qkv`` or
            ``linear_fc1`` — those map to the split forms below.
        r: Number of OFT blocks. Mutually exclusive with ``block_size``.
        block_size: Size of each orthogonal block. Default 32.
        coft: Whether to use Constrained OFT.
        eps: Epsilon controlling rotation strength for COFT.
        block_share: Whether all blocks share parameters.
        module_dropout: Multiplicative dropout probability for blocks.
    """

    target_modules: List[str] = field(
        default_factory=lambda: [
            "linear_q",
            "linear_k",
            "linear_v",
            "linear_proj",
            "linear_fc1_up",
            "linear_fc1_gate",
            "linear_fc2",
        ]
    )
    r: int = 0
    block_size: int = 32
    coft: bool = False
    eps: float = 6e-5
    block_share: bool = False
    module_dropout: float = 0.0

    # Suffix -> fused Megatron leaf the suffix maps into. Self-mapping (target ->
    # target) is added unconditionally so unfused topologies still match.
    _SPLIT_SUFFIX_TO_FUSED = {
        "linear_q": "linear_qkv",
        "linear_k": "linear_qkv",
        "linear_v": "linear_qkv",
        "linear_fc1_up": "linear_fc1",
        "linear_fc1_gate": "linear_fc1",
    }

    def __post_init__(self) -> None:
        for target in self.target_modules:
            assert not target.endswith("linear_qkv"), (
                "CanonicalOFT does not accept target 'linear_qkv'. Use 'linear_q', 'linear_k', "
                "'linear_v' (split). Legacy `OFT` with 'linear_qkv' is mathematically "
                "incorrect for fused projections — use CanonicalOFT."
            )
            assert not target.endswith("linear_fc1"), (
                "CanonicalOFT does not accept target 'linear_fc1'. Use 'linear_fc1_up', "
                "'linear_fc1_gate' (split). Legacy `OFT` with 'linear_fc1' is mathematically "
                "incorrect for fused projections — use CanonicalOFT."
            )

            for suffix, fused_leaf in self._SPLIT_SUFFIX_TO_FUSED.items():
                if target.endswith(suffix):
                    self.canonical_mapping[target.replace(suffix, fused_leaf)].add(suffix)
                    self.canonical_mapping[target].add(suffix)
                    break
            else:
                self.canonical_mapping[target].add(target)

    def transform(self, module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None) -> nn.Module:
        """Apply CanonicalOFT to a module: split wrappers for fused linears,
        plain OFTLinear for everything else."""
        if isinstance(
            module,
            (
                OFTLinear,
                OFTLinearSplitQKV,
                OFTLinearSplitFC1UpGate,
                OFTLinearGroupedSplitFC1UpGate,
                OFTVocabParallelEmbedding,
                _SplitLNCanonicalOFTQKV,
                _SplitLNCanonicalOFTFC1,
            ),
        ):
            return module

        ans = self.match(module, name, prefix)
        if ans is None:
            return module
        matched_pattern, full_name = ans

        # word_embeddings (VocabParallelEmbedding) — rotation lives on the
        # hidden dim (replicated across TP). Routed here BEFORE the linear-attrs
        # extraction below because get_adapter_attributes_from_linear assumes a
        # Linear-shaped module and would fail on a VocabParallelEmbedding.
        # NOTE: no tied-embedding guard here — share_embeddings_and_output_weights
        # lives on the GPTModel, not on VocabParallelEmbedding.config. Tied
        # models will silently double-rotate; Qwen2.5-7B is tie_word_embeddings:false.
        if matched_pattern == "word_embeddings":
            embedding_dim = getattr(module, "embedding_dim", None)
            if embedding_dim is None and hasattr(module, "weight"):
                embedding_dim = module.weight.shape[-1]
            assert embedding_dim is not None, f"Cannot infer embedding_dim from {type(module).__name__} at {full_name}"
            adapter = OFTRotationModule(
                in_features=embedding_dim,
                r=self.r,
                block_size=self.block_size,
                coft=self.coft,
                eps=self.eps,
                block_share=self.block_share,
                module_dropout=self.module_dropout,
                model_parallel_config=getattr(module, "config", None),
                input_is_parallel=False,
                is_expert=False,
            )
            logger.info("CanonicalOFT: OFTVocabParallelEmbedding at %s", full_name)
            return OFTVocabParallelEmbedding(module, adapter)

        model_parallel_config = getattr(module, "config", None)
        if model_parallel_config is not None:
            is_expert = is_expert_linear(full_name)
            attrs = get_oft_adapter_attributes_from_linear(module, is_expert=is_expert)
            if attrs.input_is_parallel:
                if is_expert:
                    tp_size = parallel_state.get_expert_tensor_parallel_world_size()
                else:
                    tp_size = parallel_state.get_tensor_model_parallel_world_size()
                rotation_in_features = attrs.in_features // tp_size
            else:
                rotation_in_features = attrs.in_features
            input_is_parallel = attrs.input_is_parallel
        else:
            is_expert = False
            input_is_parallel = False
            rotation_in_features = module.in_features

        kwargs = dict(
            in_features=rotation_in_features,
            r=self.r,
            block_size=self.block_size,
            coft=self.coft,
            eps=self.eps,
            block_share=self.block_share,
            module_dropout=self.module_dropout,
            model_parallel_config=model_parallel_config,
            input_is_parallel=input_is_parallel,
            is_expert=is_expert,
        )

        if matched_pattern.endswith("linear_fc1") and is_grouped_expert_linear(full_name):
            logger.info("CanonicalOFT: OFTLinearGroupedSplitFC1UpGate at %s", full_name)
            return OFTLinearGroupedSplitFC1UpGate(module, **kwargs)

        if matched_pattern.endswith("linear_fc1") and _should_treat_linear_fc1_as_unfused(full_name):
            logger.info(
                "CanonicalOFT: OFTLinear at %s (treating unsupported canonical linear_fc1 as unfused)", full_name
            )
            adapter = OFTRotationModule(**kwargs)
            return OFTLinear(module, adapter)

        if HAVE_TE_LN_COL_LINEAR and isinstance(module, TELayerNormColumnParallelLinear):
            if matched_pattern.endswith("linear_qkv"):
                logger.debug("CanonicalOFT: _SplitLNCanonicalOFTQKV at %s", full_name)
                qkv = OFTLinearSplitQKV(module, provider=model_parallel_config, **kwargs)
                return _SplitLNCanonicalOFTQKV(module, qkv)
            if matched_pattern.endswith("linear_fc1"):
                logger.debug("CanonicalOFT: _SplitLNCanonicalOFTFC1 at %s", full_name)
                fc1 = OFTLinearSplitFC1UpGate(module, **kwargs)
                return _SplitLNCanonicalOFTFC1(module, fc1)
            # Fall through for linear_proj / linear_fc2 (RowParallel — never LN-fused).

        if matched_pattern.endswith("linear_qkv"):
            assert model_parallel_config is not None, "linear_qkv must be a Megatron parallel linear"
            logger.debug("CanonicalOFT: OFTLinearSplitQKV at %s", full_name)
            return OFTLinearSplitQKV(module, provider=model_parallel_config, **kwargs)

        if matched_pattern.endswith("linear_fc1"):
            logger.debug("CanonicalOFT: OFTLinearSplitFC1UpGate at %s", full_name)
            return OFTLinearSplitFC1UpGate(module, **kwargs)

        logger.debug("CanonicalOFT: OFTLinear at %s", full_name)
        adapter = OFTRotationModule(**kwargs)
        return OFTLinear(module, adapter)


@dataclass
class CanonicalOFTMerge(OrbitPEFTMixin, PEFT):
    """Folds per-projection R's into the fused W_qkv / W_fc1.

    ``split_qkv_weights`` returns a row-gather (built via
    ``torch.cat([torch.arange(...)])`` — see ``param_mapping.py:2639``), so it
    is *not* a contiguous view of W_qkv. We compute the merged slices
    out-of-place and re-interleave back into ``W_qkv.data`` via the same
    per-group layout ``_interleave_qkv`` uses at forward time.
    """

    @torch.no_grad()
    def transform(self, module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None) -> nn.Module:
        if isinstance(module, OFTLinearSplitQKV):
            self._merge_qkv(module)
            return module
        if isinstance(module, OFTLinearSplitFC1UpGate):
            self._merge_fc1(module)
            return module
        return module

    @staticmethod
    @torch.no_grad()
    def _merge_qkv(wrapper: OFTLinearSplitQKV) -> None:
        W = wrapper.to_wrap.weight
        W_q, W_k, W_v = wrapper._split_qkv_weight(W)
        # OFTRotationModule.forward applies rotation as ``x @ R``. For
        # ``F.linear(x, W_merged) == F.linear(x @ R, W)`` we need
        # ``W_merged = W @ R.T`` (so ``x @ W_merged.T = x @ R @ W.T``).
        R_q = wrapper.adapter_q.get_delta_weight().to(W.device, W.dtype)
        R_k = wrapper.adapter_k.get_delta_weight().to(W.device, W.dtype)
        R_v = wrapper.adapter_v.get_delta_weight().to(W.device, W.dtype)

        W_q_merged = W_q @ R_q.T
        W_k_merged = W_k @ R_k.T
        W_v_merged = W_v @ R_v.T

        local_query_groups, heads_per_group, head_size = wrapper._packed_qkv_layout(W.shape[0])
        local_head_num = local_query_groups * heads_per_group
        in_features = W.shape[1]

        W_q_merged = W_q_merged.reshape(local_head_num, head_size, in_features)
        W_k_merged = W_k_merged.reshape(local_query_groups, head_size, in_features)
        W_v_merged = W_v_merged.reshape(local_query_groups, head_size, in_features)

        chunks = []
        for i in range(local_query_groups):
            chunks.append(W_q_merged[i * heads_per_group : (i + 1) * heads_per_group])
            chunks.append(W_k_merged[i : i + 1])
            chunks.append(W_v_merged[i : i + 1])
        W_new = torch.cat(chunks, dim=0).reshape(-1, in_features)
        wrapper.to_wrap.weight.data.copy_(W_new)

    @staticmethod
    @torch.no_grad()
    def _merge_fc1(wrapper: OFTLinearSplitFC1UpGate) -> None:
        W = wrapper.to_wrap.weight
        half = W.shape[0] // 2
        # See ``_merge_qkv``: forward uses ``x @ R``, so the merged base weight
        # must be ``W @ R.T``.
        R_gate = wrapper.adapter_gate.get_delta_weight().to(W.device, W.dtype)
        R_up = wrapper.adapter_up.get_delta_weight().to(W.device, W.dtype)
        W_gate_new = W[:half] @ R_gate.T
        W_up_new = W[half:] @ R_up.T
        W.data.copy_(torch.cat([W_gate_new, W_up_new], dim=0))
