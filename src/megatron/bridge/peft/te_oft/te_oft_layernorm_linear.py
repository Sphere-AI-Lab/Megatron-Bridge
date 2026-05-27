# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
OFT-compatible de-fused TELayerNormColumnParallelLinear.

Strategy:
    Copy TE's _LayerNormLinear autograd Function and insert an OFT rotation
    hook between the LayerNorm and GEMM steps. This preserves all of TE's
    optimizations (FP8 quantization, Userbuffers comm overlap, SP all-gather,
    etc.) while allowing OFT to rotate the normalized input before the GEMM.

    The modification is minimal: after apply_normalization() produces ln_out,
    we call adapter(ln_out) to apply the block-diagonal orthogonal rotation,
    then continue with the rest of the forward as normal.

    For backward, the OFT rotation is differentiable via autograd (the rotation
    module uses standard PyTorch ops), so we don't need to modify the backward
    pass — PyTorch will chain the gradients correctly.

Files copied/adapted from:
    - transformer_engine/pytorch/module/layernorm_linear.py
      Class: _LayerNormLinear (autograd Function, lines 85-1030)
      Class: LayerNormLinear (nn.Module, lines 1033-1608)
      Reference copy: te_oft/ref_te_layernorm_linear.py

    - transformer_engine/pytorch/module/_common.py
      Function: apply_normalization (line 34)
      Reference copy: te_oft/ref_te_common.py

    - megatron/core/extensions/transformer_engine.py
      Class: TELayerNormColumnParallelLinear (lines 711-930)
      Class: TEColumnParallelLinear (lines 934-1012)
      Class: TELinear (lines 470-690)
      Function: _get_fp8_autocast_for_quant_params
      Reference copy: te_oft/ref_mcore_transformer_engine.py
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


if TYPE_CHECKING:
    from megatron.core.dist_checkpointing.mapping import ShardedStateDict

try:
    import transformer_engine as te
    import transformer_engine_torch as tex

    # --- From Megatron ---
    from megatron.core.extensions.transformer_engine import (
        TELayerNormColumnParallelLinear,
        _get_fp8_autocast_for_quant_params,
    )
    from transformer_engine.pytorch.cpp_extensions import general_gemm

    # --- From other submodules ---
    from transformer_engine.pytorch.cpu_offload import (
        is_cpu_offload_enabled,
        mark_activation_offload,
        mark_not_offload,
        start_offload,
    )

    # --- From distributed ---
    from transformer_engine.pytorch.distributed import (
        _fsdp_gather_tensors,
        _fsdp_scatter_tensors,
        allreduce,
        gather_along_first_dim,
        get_distributed_world_size,
        in_fp8_activation_recompute_phase,
        reduce_scatter_along_first_dim,
        symmetric_all_reduce,
    )
    from transformer_engine.pytorch.graph import is_graph_capturing

    # --- From module._common ---
    from transformer_engine.pytorch.module._common import apply_normalization

    # --- From module.base (many things live here, not where you'd expect) ---
    from transformer_engine.pytorch.module.base import (
        _2X_ACC_DGRAD,
        _2X_ACC_FPROP,
        _2X_ACC_WGRAD,
        TransformerEngineBaseModule,
        fill_userbuffers_buffer_for_all_gather,
        get_dummy_wgrad,
        get_ub,
    )

    # --- From quantization / quantized_tensor ---
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager
    from transformer_engine.pytorch.quantized_tensor import (
        QuantizedTensorStorage,
        prepare_for_saving,
        restore_from_saved,
    )
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    from transformer_engine.pytorch.tensor.storage.float8_blockwise_tensor_storage import Float8BlockwiseQTensorStorage
    from transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage

    # --- From tensor submodules ---
    from transformer_engine.pytorch.tensor.utils import is_custom

    # --- From utils ---
    from transformer_engine.pytorch.utils import (
        assert_dim_for_fp8_exec,
        cast_if_needed,
        clear_tensor_data,
        get_nvtx_range_context,
        needs_quantized_gemm,
        nvtx_range_pop,
        nvtx_range_push,
        requires_grad,
    )

    HAVE_TE = True
except (ImportError, RuntimeError) as e:
    import warnings

    warnings.warn(f"TEOFTLayerNormColumnParallelLinear: TE import failed: {e}")
    exit(1)
    HAVE_TE = False


class _OFTLayerNormLinear(torch.autograd.Function):
    """Modified _LayerNormLinear that applies an OFT rotation between LN and GEMM.

    This is a copy of TE's _LayerNormLinear.forward with one insertion point:
    after apply_normalization() produces ln_out, we apply the OFT adapter
    to rotate the normalized input before proceeding to the GEMM.

    The backward pass is unchanged — the OFT rotation is a standard
    differentiable operation that PyTorch's autograd handles automatically.

    IMPORTANT: This function is tightly coupled to the TE version installed.
    Copied from transformer_engine==2.13.0+28777046, updated for TE 2.14.0.
    If TE is updated, this may need to be updated to match.
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: Union[torch.Tensor, None],
        weight: torch.Tensor,
        bias: torch.Tensor,
        non_tensor_args: Tuple,
        # === OFT addition ===
        adapter_fn: Optional[Callable] = None,
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        """Forward pass with OFT rotation inserted between LN and GEMM.

        The adapter_fn parameter is the OFT adapter's forward method.
        It is called on the LN output before the GEMM:
            ln_out = apply_normalization(inp, ...)
            ln_out = adapter_fn(ln_out)  # <-- OFT rotation inserted here
            out = general_gemm(weight, ln_out, ...)
        """
        (
            eps,
            is_first_microbatch,
            fp8,
            fp8_calibration,
            wgrad_store,
            fuse_wgrad_accumulation,
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            grad_input_quantizer,
            grad_weight_quantizer,
            grad_output_quantizer,
            cpu_offloading,
            tp_group,
            tp_size,
            sequence_parallel,
            tensor_parallel,
            activation_dtype,
            parallel_mode,
            return_layernorm_output,
            return_layernorm_output_gathered,
            is_grad_enabled,
            fwd_ln_sm_margin,
            bwd_ln_sm_margin,
            zero_centered_gamma,
            normalization,
            ub_overlap_ag_fprop,
            ub_overlap_rs_fprop,
            ub_overlap_ag_dgrad,
            ub_overlap_rs_dgrad,
            ub_bulk_wgrad,
            ub_bulk_dgrad,
            ub_name,
            fsdp_group,
            module,
            skip_fp8_weight_update,
            symmetric_ar_type,
            debug,
        ) = non_tensor_args

        nvtx_label = "te_oft._OFTLayerNormLinear.forward"
        if ub_name is not None:
            nvtx_label = f"{nvtx_label}.{ub_name}"

        with_input_all_gather = parallel_mode == "column" and sequence_parallel

        out_features, in_features = weight.shape
        inp_shape = inp.shape
        inp_requires_grad = inp.requires_grad
        assert inp_shape[-1] == in_features, "GEMM not possible"
        inp = inp.view((-1, in_features))
        inputmat = inp
        if fp8:
            assert_dim_for_fp8_exec(inputmat, weight)

        nvtx_range_push(f"{nvtx_label}.norm_input_cast")
        inputmat = cast_if_needed(inputmat, activation_dtype)
        ln_weight = cast_if_needed(ln_weight, activation_dtype)
        if ln_bias is not None:
            ln_bias = cast_if_needed(ln_bias, activation_dtype)
        nvtx_range_pop(f"{nvtx_label}.norm_input_cast")

        if is_cpu_offload_enabled():
            start_offload(inputmat)

        tp_world_size = get_distributed_world_size(tp_group)
        weight_requires_grad = weight.requires_grad
        backward_needs_input = is_grad_enabled and weight_requires_grad

        if debug:
            ub_overlap_ag_fprop = False
            ub_overlap_rs_fprop = False
            ub_overlap_ag_dgrad = False
            ub_overlap_rs_dgrad = False
            ub_bulk_wgrad = False
            ub_bulk_dgrad = False
        ub_obj = None
        ub_type = None
        ub_overlap_ag_fprop = ub_overlap_ag_fprop and is_grad_enabled and not return_layernorm_output
        if ub_overlap_rs_fprop:
            ub_obj = get_ub(ub_name + "_fprop", fp8)
            ub_type = tex.CommOverlapType.RS
        elif ub_overlap_ag_fprop:
            ub_obj = get_ub(ub_name + "_fprop", fp8)
            ub_type = tex.CommOverlapType.AG

        if fp8:
            if input_quantizer is None:
                raise ValueError("Missing quantizer for input tensor")
            input_quantizer.set_usage(rowwise=True, columnwise=backward_needs_input)
            if with_input_all_gather and input_quantizer.supports_only_rowwise_all_gather():
                input_quantizer.set_usage(columnwise=False)

        custom = is_custom(input_quantizer)
        with_quantized_norm = (
            fp8 and not debug and not return_layernorm_output and not return_layernorm_output_gathered and not custom
        )

        # ============================================================
        # Step 1: Apply normalization (same as original TE)
        # ============================================================
        nvtx_range_push(f"{nvtx_label}.norm")
        ln_out, mu, rsigma = apply_normalization(
            inputmat,
            None,
            ln_weight,
            ln_bias,
            eps,
            input_quantizer if with_quantized_norm else None,
            inputmat.dtype,
            normalization,
            fwd_ln_sm_margin,
            zero_centered_gamma,
        )
        nvtx_range_pop(f"{nvtx_label}.norm")

        # Store unquantized layer norm output if we need to return it
        ln_out_return = None
        if return_layernorm_output or return_layernorm_output_gathered:
            ln_out_return = ln_out

        # ============================================================
        # Step 2: OFT ROTATION (the only modification vs original TE)
        # ============================================================
        if adapter_fn is not None:
            # Apply OFT rotation to the normalized output.
            # ln_out shape: [seq*batch, hidden] (2D after view)
            # adapter_fn expects (..., hidden) and returns same shape.
            #
            # When fused quantized norm is active (with_quantized_norm=True),
            # ln_out is a Float8TensorStorage. OFT rotation operates in BF16,
            # so we dequantize first. Step 3 will re-quantize for the GEMM.
            # Flow: fused LN→FP8 → dequant→BF16 → OFT rotation → requant→FP8 → GEMM
            nvtx_range_push(f"{nvtx_label}.oft_rotation")
            if with_quantized_norm and hasattr(ln_out, 'dequantize'):
                ln_out = ln_out.dequantize(dtype=inputmat.dtype)
                with_quantized_norm = False  # Force re-quantization in Step 3
            ln_out = adapter_fn(ln_out.contiguous())
            nvtx_range_pop(f"{nvtx_label}.oft_rotation")

        # ============================================================
        # Step 3: Prepare GEMM input (all-gather for SP, same as original)
        # ============================================================
        nvtx_range_push(f"{nvtx_label}.gemm_input_cast_comm")
        ln_out_total = None
        if with_input_all_gather:
            if return_layernorm_output_gathered:
                ln_out_total, _ = gather_along_first_dim(ln_out, tp_group)
                ln_out_return = ln_out_total
                if fp8 or debug:
                    ln_out = input_quantizer(ln_out)
                    input_quantizer.set_usage(rowwise=True, columnwise=False)
                    ln_out_total = input_quantizer(ln_out_total)
            else:
                quantizer = None
                if fp8 or debug:
                    quantizer = input_quantizer
                    if not with_quantized_norm and not custom:
                        ln_out = quantizer(ln_out)
                    quantizer.set_usage(rowwise=True, columnwise=False)
                if ub_overlap_ag_fprop:
                    ln_out_total, _ = fill_userbuffers_buffer_for_all_gather(
                        ub_obj,
                        ln_out,
                        quantizer,
                        tp_group,
                    )
                else:
                    ln_out_total, _ = gather_along_first_dim(
                        ln_out,
                        tp_group,
                        quantizer=quantizer,
                    )
        else:
            if (fp8 or debug) and not with_quantized_norm:
                ln_out = input_quantizer(ln_out)
            ln_out_total = ln_out
        nvtx_range_pop(f"{nvtx_label}.gemm_input_cast_comm")

        # ============================================================
        # Prepare weight tensor (same as original)
        # ============================================================
        weightmat = weight
        is_weight_param_quantized = False
        if fp8 or debug:
            is_weight_param_quantized = isinstance(weight, QuantizedTensorStorage)
            if is_weight_param_quantized and not debug:
                weight_quantizer = weight._quantizer
            elif weight_quantizer is not None:
                weight_quantizer.set_usage(rowwise=True, columnwise=is_grad_enabled)
            update_workspace = is_first_microbatch is None or is_first_microbatch
            weightmat = module.get_weight_workspace(
                tensor=weight,
                quantizer=weight_quantizer,
                cache_name=(None if is_first_microbatch is None else "weight"),
                update_workspace=update_workspace,
                skip_update_flag=skip_fp8_weight_update,
                fsdp_group=fsdp_group,
                workspace_dtype=activation_dtype,
            )
            weightmat.update_usage(rowwise_usage=True)
        else:
            weightmat = cast_if_needed(weightmat, activation_dtype)

        bias_dtype = activation_dtype
        if needs_quantized_gemm(ln_out_total) and activation_dtype == torch.float32:
            bias_dtype = torch.bfloat16
        bias = cast_if_needed(bias, bias_dtype) if bias is not None else bias

        if not fp8 and fp8_calibration:
            if input_quantizer is not None:
                input_quantizer.calibrate(ln_out_total)
            if weight_quantizer is not None:
                weight_quantizer.calibrate(weight)

        use_split_accumulator = _2X_ACC_FPROP
        if fp8:
            recipe = FP8GlobalStateManager.get_fp8_recipe()
            if hasattr(recipe, "fp8_gemm_fprop"):
                use_split_accumulator = recipe.fp8_gemm_fprop.use_split_accumulator

        if output_quantizer is not None:
            output_quantizer.set_usage(rowwise=True, columnwise=False)

        reduce_scatter_out = None
        if ub_overlap_rs_fprop:
            out_shape = list(inp_shape)
            out_shape[0] //= tp_world_size
            out_shape[-1] = out_features
            reduce_scatter_out = torch.empty(out_shape, dtype=activation_dtype, device=inp.device)

        # ============================================================
        # Forward GEMM (same as original)
        # ============================================================
        nvtx_range_push(f"{nvtx_label}.gemm")
        gemm_out, *_, reduce_scatter_out = general_gemm(
            weightmat,
            ln_out_total,
            quantization_params=output_quantizer,
            out_dtype=activation_dtype,
            bias=bias,
            use_split_accumulator=use_split_accumulator,
            ub=ub_obj,
            ub_type=ub_type,
            extra_output=reduce_scatter_out,
        )
        nvtx_range_pop(f"{nvtx_label}.gemm")

        if not weight.requires_grad and not return_layernorm_output:
            clear_tensor_data(ln_out, ln_out_total)
            ln_out = ln_out_total = None
        elif with_input_all_gather and not return_layernorm_output_gathered:
            clear_tensor_data(ln_out_total)
            ln_out_total = None

        out = None
        if ub_overlap_rs_fprop:
            out = reduce_scatter_out
        elif parallel_mode == "row" and tp_size > 1:
            nvtx_range_push(f"{nvtx_label}.row_parallel_comm")
            out = gemm_out
            if sequence_parallel:
                out, _ = reduce_scatter_along_first_dim(out, tp_group)
            elif tensor_parallel:
                if symmetric_ar_type is not None:
                    out, _ = symmetric_all_reduce(out, tp_group, all_reduce_type=symmetric_ar_type)
                else:
                    out, _ = allreduce(out, tp_group)
            nvtx_range_pop(f"{nvtx_label}.row_parallel_comm")
        else:
            out = gemm_out
        out = out.view(-1, *inp_shape[1:-1], out_features)

        # ============================================================
        # Cache state for backward (same as original)
        # ============================================================
        if is_grad_enabled:
            ctx.weight_quantizer = weight_quantizer
            ctx.ln_out_needs_gather = weight.requires_grad and parallel_mode == "column" and sequence_parallel
            if backward_needs_input:
                if isinstance(ln_out, QuantizedTensorStorage):
                    if (
                        isinstance(ln_out, (MXFP8TensorStorage, Float8BlockwiseQTensorStorage))
                        or not ctx.ln_out_needs_gather
                    ):
                        ln_out.update_usage(rowwise_usage=False)
            if cpu_offloading:
                mark_activation_offload(inputmat, mu, rsigma, ln_out)

            nvtx_range_push(f"{nvtx_label}.fsdp_scatter")
            ctx.fsdp_group = fsdp_group
            ctx.fsdp_shapes = _fsdp_scatter_tensors(
                fsdp_group,
                mu,
                rsigma,
                weightmat if fp8 and not is_weight_param_quantized else None,
                ln_out if weight.requires_grad else None,
            )
            nvtx_range_pop(f"{nvtx_label}.fsdp_scatter")

            if cpu_offloading:
                mark_not_offload(weightmat, weight, bias, ln_weight, ln_bias)
                ctx.grad_added_to_main_grad = hasattr(weight, "grad_added_to_main_grad")
                if ctx.grad_added_to_main_grad:
                    ctx.weight_object = weight

            tensors_to_save, tensor_objects = prepare_for_saving(
                inputmat,
                weightmat,
                weight,
                bias,
                ln_weight,
                ln_out,
                mu,
                rsigma,
            )
            ctx.save_for_backward(*tensors_to_save)
            ctx.tensor_objects = tensor_objects
            ctx.requires_dgrad = inp_requires_grad
            ctx.requires_wgrad = weight.requires_grad
            ctx.is_weight_param_quantized = is_weight_param_quantized
            if fuse_wgrad_accumulation and weight.requires_grad:
                if hasattr(weight, "__fsdp_param__"):
                    ctx.main_grad_func = weight.get_main_grad
                else:
                    ctx.main_grad_func = lambda: weight.main_grad
            ctx.grad_input_quantizer = grad_input_quantizer
            ctx.grad_weight_quantizer = grad_weight_quantizer
            ctx.grad_output_quantizer = grad_output_quantizer
            ctx.input_quantizer = input_quantizer
            ctx.owns_input = inputmat is not inp
            ctx.weight = weight
            ctx.activation_dtype = activation_dtype
            ctx.fp8 = fp8
            ctx.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if fp8 else None
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.cpu_offloading = cpu_offloading
            ctx.is_first_microbatch = is_first_microbatch
            ctx.use_bias = bias is not None
            ctx.sequence_parallel = sequence_parallel
            ctx.tensor_parallel = tensor_parallel
            ctx.inp_shape = inp_shape
            ctx.parallel_mode = parallel_mode
            ctx.tp_group = tp_group
            ctx.tp_size = tp_size
            ctx.return_layernorm_output = return_layernorm_output
            ctx.return_layernorm_output_gathered = return_layernorm_output_gathered
            ctx.bwd_ln_sm_margin = bwd_ln_sm_margin
            ctx.zero_centered_gamma = zero_centered_gamma
            ctx.ub_overlap_ag = ub_overlap_ag_dgrad
            ctx.ub_overlap_rs_dgrad = ub_overlap_rs_dgrad
            ctx.ub_bulk_wgrad = ub_bulk_wgrad
            ctx.ub_bulk_dgrad = ub_bulk_dgrad
            ctx.ub_name = ub_name
            ctx.requires_dgrad = inp_requires_grad
            ctx.normalization = normalization
            ctx.reduce_and_update_bwd_fp8_tensors = False
            if ctx.fp8 and requires_grad(inp, ln_weight, ln_bias, weight, bias):
                _first_fp8_module = FP8GlobalStateManager.IS_FIRST_FP8_MODULE
                ctx.reduce_and_update_bwd_fp8_tensors = FP8GlobalStateManager.is_first_fp8_module()
                if in_fp8_activation_recompute_phase():
                    FP8GlobalStateManager.IS_FIRST_FP8_MODULE = _first_fp8_module
            ctx.wgrad_store = wgrad_store
            ctx.debug = debug

        if return_layernorm_output:
            if return_layernorm_output_gathered:
                shape = list(inp_shape)
                shape[0] *= tp_size if with_input_all_gather else 1
                return out, ln_out_return.view(shape)
            return out, ln_out_return.view(inp_shape)
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Reuse TE's backward exactly — import the original
        from transformer_engine.pytorch.module.layernorm_linear import _LayerNormLinear

        return _LayerNormLinear.backward(ctx, *grad_outputs) + (None,)  # extra None for adapter_fn


class TEOFTLayerNormColumnParallelLinear(nn.Module):
    """Drop-in replacement for TELayerNormColumnParallelLinear with OFT support.

    Wraps the original module and overrides forward() to use
    _OFTLayerNormLinear which inserts OFT rotation between LN and GEMM.

    Args:
        orig_module: The original TELayerNormColumnParallelLinear.
        adapter: The OFTRotationModule to apply between LN and Linear.
    """

    def __init__(self, orig_module: nn.Module, adapter: nn.Module) -> None:
        super().__init__()
        assert HAVE_TE, "Transformer Engine required"
        self._orig_module = orig_module
        self.adapter = adapter
        self._adapter_enabled = True

    def enable_adapter_layers(self) -> None:
        self._adapter_enabled = True

    def disable_adapter_layers(self) -> None:
        self._adapter_enabled = False

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        if not self._adapter_enabled:
            result = self._orig_module(x, *args, **kwargs)
            # Normalize return value to (output, bias) 2-tuple.
            # The raw TE module may return nested tuples like ((out, ln_out), bias)
            # or 3-tuples like (out, bias, ln_out) which break callers expecting
            # a simple (tensor, bias) pair.
            if isinstance(result, tuple):
                if len(result) == 3:
                    # (out, bias, ln_out) -> (out, bias)
                    return result[0], result[1]
                if len(result) == 2 and isinstance(result[0], tuple):
                    # ((out, ln_out), bias) -> (out, bias)
                    return result[0][0], result[1]
            return result

        # Use the modified autograd function with OFT rotation hook
        _is_first_microbatch = (
            None if self._orig_module.disable_parameter_transpose_cache else self._orig_module.is_first_microbatch
        )
        quant_context = _get_fp8_autocast_for_quant_params(
            self._orig_module.te_quant_params, self._orig_module.training
        )

        # We need to call the TE module's prepare_forward / end_forward lifecycle.
        # The simplest way: replicate TELayerNormColumnParallelLinear.forward but
        # swap _LayerNormLinear with _OFTLayerNormLinear.
        #
        # However, the parent LayerNormLinear.forward (TE) handles prepare_forward,
        # quantizer setup, etc. We need to replicate that too.
        #
        # For now, we directly call the modified autograd function by replicating
        # the parent's forward() logic with our adapter inserted.
        mod = self._orig_module

        is_grad_enabled = torch.is_grad_enabled()
        debug = mod.is_debug_iter()

        if FP8GlobalStateManager.fp8_graph_capturing():
            skip_fp8_weight_update = FP8GlobalStateManager.get_skip_fp8_weight_update_tensor()
        else:
            skip_fp8_weight_update = None
        if skip_fp8_weight_update is not None:
            _is_first_microbatch = False

        if mod.ub_overlap_rs_fprop:
            if get_ub(mod.ub_name + "_fprop", FP8GlobalStateManager.is_fp8_enabled()).is_fp8_ubuf():
                fp8_output = True
            else:
                fp8_output = False
        else:
            fp8_output = False
        if mod.ub_overlap_rs_dgrad:
            if get_ub(mod.ub_name + "_dgrad", FP8GlobalStateManager.is_fp8_enabled()).is_fp8_ubuf():
                fp8_grad = True
            else:
                fp8_grad = False
        else:
            fp8_grad = False

        inp = mod.prepare_forward(x, allow_non_contiguous=False)

        try:
            weight_tensor, bias_tensor = mod._get_weight_and_bias_tensors()

            quantizers = (
                mod._get_quantizers(fp8_output, fp8_grad, is_grad_enabled)
                if not debug
                else mod._get_debug_quantizers(fp8_output, fp8_grad, is_grad_enabled)
            )
            if debug:
                if mod.no_debug_features_active(quantizers):
                    debug = False
                    quantizers = mod._get_quantizers(fp8_output, fp8_grad, is_grad_enabled)

            (
                input_quantizer,
                weight_quantizer,
                output_quantizer,
                grad_input_quantizer,
                grad_weight_quantizer,
                grad_output_quantizer,
            ) = quantizers

            if is_grad_enabled:
                fwd_fn = _OFTLayerNormLinear.apply
                autograd_ctx = []
            else:
                fwd_fn = _OFTLayerNormLinear.forward
                autograd_ctx = [None]

            non_tensor_args = (
                mod.eps,
                _is_first_microbatch,
                mod.fp8,
                mod.fp8_calibration,
                mod.wgrad_store,
                mod.fuse_wgrad_accumulation,
                input_quantizer,
                weight_quantizer,
                output_quantizer,
                grad_input_quantizer,
                grad_weight_quantizer,
                grad_output_quantizer,
                is_cpu_offload_enabled(),
                mod.tp_group,
                mod.tp_size,
                mod.sequence_parallel,
                mod.tp_size > 1,
                mod.activation_dtype,
                mod.parallel_mode,
                mod.return_layernorm_output,
                mod.return_layernorm_output_gathered,
                is_grad_enabled,
                mod.fwd_ln_sm_margin if is_grad_enabled else mod.inf_ln_sm_margin,
                mod.bwd_ln_sm_margin,
                mod.zero_centered_gamma,
                mod.normalization,
                mod.ub_overlap_ag_fprop,
                mod.ub_overlap_rs_fprop,
                mod.ub_overlap_ag_dgrad,
                mod.ub_overlap_rs_dgrad,
                mod.ub_bulk_wgrad,
                mod.ub_bulk_dgrad,
                mod.ub_name,
                mod.fsdp_group,
                mod,
                skip_fp8_weight_update,
                mod.symmetric_ar_type,
                debug,
            )

            with quant_context:
                out = fwd_fn(
                    *autograd_ctx,
                    inp,
                    mod.layer_norm_weight,
                    mod.layer_norm_bias,
                    weight_tensor,
                    bias_tensor if mod.apply_bias and not mod.gemm_bias_unfused_add else None,
                    non_tensor_args,
                    self.adapter,  # OFT adapter passed as extra arg
                )
        finally:
            mod.end_forward()

        mod.is_first_microbatch = False

        if mod.return_layernorm_output:
            out, ln_out = out

        if mod.gemm_bias_unfused_add:
            out = out + cast_if_needed(bias_tensor, mod.activation_dtype)

        if mod.te_return_bias:
            if mod.return_layernorm_output:
                return out, cast_if_needed(bias_tensor, mod.activation_dtype), ln_out
            return out, cast_if_needed(bias_tensor, mod.activation_dtype)
        if mod.return_layernorm_output:
            return out, ln_out
        # Always return 2-tuple (Megatron convention)
        return out, None

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._orig_module, name)

    def state_dict(
        self, destination: Optional[Dict[str, Any]] = None, prefix: str = "", keep_vars: bool = False
    ) -> Dict[str, Any]:
        if destination is None:
            destination = {}
        self._orig_module.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        self.adapter.state_dict(destination=destination, prefix=f"{prefix}adapter.", keep_vars=keep_vars)
        return destination

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ShardedStateDict":
        sharded_state_dict = {}
        sharded_state_dict.update(self._orig_module.sharded_state_dict(prefix, sharded_offsets, metadata))
        sharded_state_dict.update(self.adapter.sharded_state_dict(f"{prefix}adapter.", sharded_offsets, metadata))
        return sharded_state_dict
