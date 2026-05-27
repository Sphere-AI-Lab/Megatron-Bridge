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

"""OFT (Orthogonal Fine-Tuning) layer implementations for megatron-bridge.

OFT applies a learned block-diagonal orthogonal rotation to the input before the
base linear layer:  y = W @ (R @ x)
where R is parameterized via the Cayley transform of skew-symmetric matrices.

Reference: https://arxiv.org/abs/2306.07280
"""

import logging
import math
import os
import re
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state

from megatron.bridge.models.common.unimodal import to_empty_if_meta_device


logger = logging.getLogger(__name__)

_INT4_ACTIVE_EXPERT_CHUNK_SIZE = 4
# Default intentionally stays on dense INT4 dequantization to BF16 followed by
# BF16 GEMM because the real Kimi grouped-routing benchmark favors this path.
_INT4_DENSE_BF16_GROUPED_CHUNK_BACKEND = "python"
_INT4_GROUPED_CHUNK_BACKEND = _INT4_DENSE_BF16_GROUPED_CHUNK_BACKEND
_FP8_WEIGHT_BLOCK_SIZE = 128
_FP8_ACTIVATION_GROUP_SIZE = 128
_OFT_NVFP4_DEBUG_ENV = "MBRIDGE_OFT_NVFP4_DEBUG"
_OFT_NVFP4_DEBUG_LIMIT_ENV = "MBRIDGE_OFT_NVFP4_DEBUG_LIMIT"
_OFT_FP8_DEBUG_ENV = "MBRIDGE_OFT_FP8_DEBUG"
_OFT_FP8_DEBUG_LIMIT_ENV = "MBRIDGE_OFT_FP8_DEBUG_LIMIT"
_OFT_APPLIED_TRAP_ENV = "MEGATRON_OFT_APPLIED_TRAP"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


def _oft_applied_trap_enabled() -> bool:
    return _env_flag(_OFT_APPLIED_TRAP_ENV)


def _trap_oft_applied_bridge(wrapper: nn.Module, layer_kind: str) -> None:
    """Structural check: an OFT-wrapped Bridge layer is being forwarded with
    ``_adapter_enabled=False``. Under ``MEGATRON_OFT_APPLIED_TRAP=1``, this
    means the OFT rotation is structurally not running on a layer that the
    user listed in ``target_modules`` — the train-side analogue of
    SGLang's ``SGLANG_OFT_APPLIED_TRAP``. The trap is opt-in so callers that
    legitimately disable adapters (e.g. base-model eval) are not impacted.
    """
    if not _oft_applied_trap_enabled():
        return
    if getattr(wrapper, "_adapter_enabled", False):
        return
    raise RuntimeError(
        f"MEGATRON_OFT_APPLIED_TRAP triggered: OFT-wrapped {layer_kind} "
        f"({type(wrapper).__name__}) forwarded with _adapter_enabled=False. "
        "The R rotation is structurally not being applied. Either the OFT "
        "adapter is disabled for this layer, or the wrong forward branch is "
        "running. If you intentionally disabled adapters, clear "
        f"{_OFT_APPLIED_TRAP_ENV}; otherwise the rotation is silently "
        "missing on a target module."
    )


def _debug_rank_values() -> tuple[str, str]:
    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get("LOCAL_RANK", "?")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            rank = str(torch.distributed.get_rank())
        except Exception:
            pass
    return rank, local_rank


def _debug_rank_prefix() -> str:
    rank, local_rank = _debug_rank_values()
    return f"rank={rank} local_rank={local_rank}"


def _oft_nvfp4_debug_enabled() -> bool:
    return _env_flag(_OFT_NVFP4_DEBUG_ENV)


def _oft_nvfp4_debug_limit() -> int:
    try:
        return int(os.environ.get(_OFT_NVFP4_DEBUG_LIMIT_ENV, "16"))
    except ValueError:
        return 16


def _oft_fp8_debug_enabled() -> bool:
    return _env_flag(_OFT_FP8_DEBUG_ENV)


def _oft_fp8_debug_limit() -> int:
    try:
        return int(os.environ.get(_OFT_FP8_DEBUG_LIMIT_ENV, "32"))
    except ValueError:
        return 32


def _oft_fp8_debug_log(event: str, **fields: Any) -> None:
    if not _oft_fp8_debug_enabled():
        return

    count = getattr(_oft_fp8_debug_log, "_count", 0)
    if count >= _oft_fp8_debug_limit():
        return
    setattr(_oft_fp8_debug_log, "_count", count + 1)

    parts = [f"[OFT FP8 debug] event={event}", _debug_rank_prefix()]
    for key, value in fields.items():
        if isinstance(value, torch.Tensor):
            value = (
                f"shape={tuple(value.shape)} dtype={value.dtype} "
                f"device={value.device} contig={value.is_contiguous()}"
            )
        parts.append(f"{key}={value}")
    print(" ".join(parts), flush=True)


def _cuda_memory_summary(device: torch.device) -> str:
    if device.type != "cuda":
        return "mem=non_cuda"
    gib = 1024**3
    try:
        free, total = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        peak = torch.cuda.max_memory_allocated(device)
        return (
            f"mem_free={free / gib:.2f}GiB mem_total={total / gib:.2f}GiB "
            f"allocated={allocated / gib:.2f}GiB reserved={reserved / gib:.2f}GiB "
            f"peak_allocated={peak / gib:.2f}GiB"
        )
    except Exception as exc:
        return f"mem_error={type(exc).__name__}:{exc}"


def _tensor_debug_summary(name: str, tensor: torch.Tensor) -> str:
    return (
        f"{name}_shape={tuple(tensor.shape)} {name}_stride={tuple(tensor.stride())} "
        f"{name}_dtype={tensor.dtype} {name}_device={tensor.device} "
        f"{name}_contig={tensor.is_contiguous()}"
    )


def _get_oft_fp8_activation_quant_mode() -> str:
    mode = os.environ.get("MEGATRON_OFT_FP8_ACTIVATION_QUANT", "none").strip().lower()
    if mode in ("", "0", "false", "none", "off"):
        return "none"
    if mode in ("w8a8", "fp8", "dynamic"):
        return "w8a8"
    raise ValueError(
        "Unsupported MEGATRON_OFT_FP8_ACTIVATION_QUANT="
        f"{mode!r}. Expected one of: none, w8a8."
    )


def _is_direct_fp8_runtime_weight(weight: Any, scale_inv: Any) -> bool:
    """Detect direct-FP8 checkpoint weights after Megatron DCP materialization.

    Native FP8 parameters stay as ``torch.float8_e4m3fn``.  When a direct-FP8
    DCP is loaded into the standard BF16 Megatron modules, the FP8 payload is
    materialized into the BF16 parameter tensor while the sibling
    ``weight_scale_inv`` buffer preserves the block scales.  Both cases must
    use the explicit FP8 OFT path.
    """

    if getattr(weight, "dtype", None) == torch.float8_e4m3fn:
        return True
    if not isinstance(weight, torch.Tensor) or not isinstance(scale_inv, torch.Tensor):
        return False
    if not weight.is_floating_point() or weight.ndim != 2 or scale_inv.numel() == 0:
        return False
    if scale_inv.numel() == 1:
        return True
    if scale_inv.ndim != 2:
        return False

    expected_rows = max(1, math.ceil(int(weight.shape[0]) / _FP8_WEIGHT_BLOCK_SIZE))
    expected_cols = max(1, math.ceil(int(weight.shape[1]) / _FP8_WEIGHT_BLOCK_SIZE))
    return tuple(scale_inv.shape) == (expected_rows, expected_cols)


def _fp8_activation_qdq_per_token_group_ste(
    x: torch.Tensor,
    group_size: int = _FP8_ACTIVATION_GROUP_SIZE,
) -> torch.Tensor:
    """Quantize/dequantize activations like SGLang block-FP8 W8A8, with STE backward."""

    if x.shape[-1] % group_size != 0:
        raise ValueError(
            f"FP8 W8A8 activation quantization requires hidden size divisible by {group_size}, "
            f"got {x.shape[-1]}."
        )

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    original_shape = x.shape
    x_2d = x.contiguous().view(-1, x.shape[-1])
    x_grouped = x_2d.float().reshape(
        x_2d.shape[0],
        x_2d.shape[1] // group_size,
        group_size,
    )
    scale = x_grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / fp8_info.max
    x_q = torch.clamp(x_grouped / scale, fp8_info.min, fp8_info.max).to(torch.float8_e4m3fn)
    x_qdq = (x_q.float() * scale).to(x.dtype).reshape_as(x_2d).view(original_shape)
    return x + (x_qdq - x).detach()


def set_int4_active_expert_chunk_size(chunk_size: int) -> None:
    """Set the number of active experts to process per INT4 chunk.

    A value <= 0 disables the chunked sparse fallback and reverts to the
    original eager grouped path.
    """

    global _INT4_ACTIVE_EXPERT_CHUNK_SIZE
    _INT4_ACTIVE_EXPERT_CHUNK_SIZE = max(0, int(chunk_size))


def get_int4_active_expert_chunk_size() -> int:
    """Return the configured INT4 grouped-expert chunk size."""

    return _INT4_ACTIVE_EXPERT_CHUNK_SIZE


def set_int4_grouped_chunk_backend(backend: str) -> None:
    """Set the grouped INT4 chunk execution backend."""

    global _INT4_GROUPED_CHUNK_BACKEND
    normalized = str(backend).strip().lower()
    if normalized not in ("python", "te"):
        raise ValueError(
            f"Unsupported INT4 grouped chunk backend {backend!r}. Expected one of: python, te."
        )
    _INT4_GROUPED_CHUNK_BACKEND = normalized


def get_int4_grouped_chunk_backend() -> str:
    """Return the configured grouped INT4 chunk execution backend."""

    return _INT4_GROUPED_CHUNK_BACKEND

try:
    from megatron.bridge.peft.triton_oft import cayley_neumann as triton_cayley_neumann
    from megatron.bridge.peft.triton_oft import oft_r_single as triton_oft_r_single
    HAS_TRITON_OFT = True
except ImportError:
    HAS_TRITON_OFT = False
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.moe.router import apply_random_logits
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

from megatron.bridge.peft.adapter_wrapper import AdapterWrapper
from megatron.bridge.peft.int4_utils import _empty_storage_view
from megatron.bridge.peft.qwen3_fp8_gemm import (
    maybe_qwen3_native_block_fp8_linear,
    should_attempt_qwen3_native_fp8_gemm,
)
from megatron.bridge.utils.import_utils import safe_import_from


TEColumnParallelLinear, HAVE_TE_COL_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine", "TEColumnParallelLinear"
)
TERowParallelLinear, HAVE_TE_ROW_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine", "TERowParallelLinear"
)
TELinear, HAVE_TE_LINEAR = safe_import_from("megatron.core.extensions.transformer_engine", "TELinear")


def _module_bias_enabled(module: nn.Module) -> bool:
    """Return whether a wrapped linear/router should actively use bias.

    For Megatron/TE modules, explicit flags like ``use_bias`` or ``apply_bias``
    are the source of truth. Falling back to the presence of ``bias`` alone is
    unsafe under meta init because disabled-bias modules may still temporarily
    carry placeholder bias parameters.
    """
    if hasattr(module, "use_bias"):
        return bool(getattr(module, "use_bias"))
    if hasattr(module, "apply_bias"):
        return bool(getattr(module, "apply_bias"))

    bias = getattr(module, "bias", None)
    if bias is None:
        return False
    if isinstance(bias, torch.Tensor) and bias.numel() == 0:
        return False
    return True


def _get_active_bias_tensor(module: nn.Module, name: str = "bias") -> Optional[torch.Tensor]:
    """Return the active bias tensor for ``module`` or ``None`` when bias is disabled."""
    if not _module_bias_enabled(module):
        return None

    bias = getattr(module, name, None)
    if bias is None:
        return None
    if isinstance(bias, torch.Tensor) and bias.numel() == 0:
        return None
    return bias


def _clear_disabled_bias_parameters(module: nn.Module) -> None:
    """Normalize disabled bias placeholders so later save/load/runtime paths skip them."""
    if _module_bias_enabled(module):
        return

    for name, param in list(getattr(module, "_parameters", {}).items()):
        if param is None or not name.startswith("bias"):
            continue
        module._parameters[name] = None


class MultiplicativeDropoutLayer(nn.Module):
    """Applies multiplicative dropout to OFT blocks.

    During training, randomly replaces OFT blocks with identity matrices.
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        num_blocks = x.shape[0]
        block_size = x.shape[1]
        mask = torch.bernoulli(torch.ones(num_blocks, 1, 1, device=x.device) * (1 - self.p))
        eye = torch.eye(block_size, device=x.device, dtype=x.dtype).unsqueeze(0)
        return x * mask + eye * (1 - mask)


class OFTRotationModule(nn.Module):
    """Core OFT module that computes block-diagonal orthogonal transformations.

    Parameterizes orthogonal matrices via the Cayley transform of skew-symmetric matrices:
        R = (I - Q)(I + Q)^{-1}
    where Q is block-diagonal skew-symmetric.

    Supports Megatron tensor parallelism: when the base linear is RowParallel
    (input_is_parallel=True), the rotation operates on the TP-local shard and
    ``oft_r`` is checkpointed as TP-sharded along axis 0 (the blocks dimension).
    When the base linear is ColumnParallel (input_is_parallel=False), the rotation
    operates on the full input and ``oft_r`` is replicated across TP ranks.

    Args:
        in_features: Input dimension of the layer to adapt.
        r: Number of OFT blocks. Either r or block_size must be specified.
        block_size: Size of each orthogonal block. Either r or block_size must be specified.
        coft: Whether to use Constrained OFT (projects onto epsilon-ball).
        eps: Epsilon for COFT constraint strength.
        block_share: Whether all blocks share the same parameters.
        module_dropout: Multiplicative dropout probability for blocks.
        model_parallel_config: Megatron ModelParallelConfig for distributed checkpointing.
        input_is_parallel: Whether the input is TP-sharded (True for RowParallel base layers).
        is_expert: Whether this adapter is for expert layers in MoE.
        dtype: Parameter dtype.
        device: Parameter device.
    """

    def __init__(
        self,
        in_features: int,
        r: int = 0,
        block_size: int = 0,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        module_dropout: float = 0.0,
        model_parallel_config: Optional[ModelParallelConfig] = None,
        input_is_parallel: bool = False,
        is_expert: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if r > 0 and block_size > 0:
            raise ValueError("Only one of r or block_size should be specified, not both.")
        if r == 0 and block_size == 0:
            raise ValueError("One of r or block_size must be specified.")

        if r > 0:
            if in_features % r != 0:
                r = self._find_nearest_divisor(in_features, r)
            block_size = in_features // r
        else:
            if in_features % block_size != 0:
                block_size = self._find_nearest_divisor(in_features, block_size)
            r = in_features // block_size

        self.in_features = in_features
        self.r = r
        self.block_size = block_size
        self.coft = coft
        self.eps = eps
        self.block_share = block_share
        self.input_is_parallel = input_is_parallel
        self.is_expert = is_expert

        if model_parallel_config is None:
            model_parallel_config = ModelParallelConfig()
        self.config = model_parallel_config

        n_elements = block_size * (block_size - 1) // 2
        num_blocks = 1 if block_share else r
        if model_parallel_config.bf16:
            dtype = torch.bfloat16
        elif model_parallel_config.fp16:
            dtype = torch.float16

        if self.is_expert:
            self.tp_group = parallel_state.get_expert_tensor_parallel_group()
        else:
            self.tp_group = parallel_state.get_tensor_model_parallel_group()

        self.oft_r = nn.Parameter(torch.zeros(num_blocks, n_elements, dtype=dtype, device=device))
        if self.is_expert:
            self.oft_r.allreduce = False

        self.dropout = MultiplicativeDropoutLayer(p=module_dropout)

        # Pre-register upper triangle indices as buffers for efficient skew-symmetric construction
        rows, cols = torch.triu_indices(block_size, block_size, 1)
        rows = rows.to(torch.int32)
        cols = cols.to(torch.int32)
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("cols", cols, persistent=False)


    @staticmethod
    def _find_nearest_divisor(n: int, target: int) -> int:
        """Find the nearest divisor of n to target."""
        best = 1
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                for d in (i, n // i):
                    if abs(d - target) < abs(best - target):
                        best = d
        return best

    def _pytorch_skew_symmetric(self, vec: torch.Tensor, block_size: int) -> torch.Tensor:
        """Convert parameter vectors to skew-symmetric matrices.

        Args:
            vec: (num_blocks, n_elements) parameter vectors.
            block_size: Size of each orthogonal block.

        Returns:
            (num_blocks, block_size, block_size) skew-symmetric matrices.
        """
        batch_size = vec.shape[0]
        matrix = torch.zeros(batch_size, block_size, block_size, device=vec.device, dtype=vec.dtype)
        matrix[:, self.rows, self.cols] = vec
        matrix = matrix - matrix.transpose(-2, -1)
        return matrix

    def _cayley_batch(
        self, Q: torch.Tensor, block_size: int, use_cayley_neumann: bool = True, num_neumann_terms: int = 5
    ) -> torch.Tensor:
        """Perform the Cayley parametrization on a batch of skew-symmetric parameter vectors.

        Uses triton-accelerated kernel when available. Falls back to Neumann series or exact solve.

        Args:
            Q: A batch of parameter vectors of shape (b, n_elements).
            block_size: Size of each orthogonal block.
            use_cayley_neumann: Use Neumann series approximation (default True).
            num_neumann_terms: Number of terms in the Neumann series (default 5).
        """
        b, _ = Q.shape
        previous_dtype = Q.dtype

        Q_skew = self._pytorch_skew_symmetric(Q, block_size)

        if use_cayley_neumann:
            # Use triton kernel when available (fused fwd+bwd, autograd-compatible)
            if HAS_TRITON_OFT and Q_skew.is_cuda and num_neumann_terms == 5 and block_size >= 16:
                R = triton_cayley_neumann(Q_skew, num_neumann_terms)
            else:
                R = torch.eye(block_size, device=Q.device, dtype=Q.dtype).repeat(b, 1, 1)
                if num_neumann_terms > 1:
                    R.add_(Q_skew, alpha=2.0)
                    if num_neumann_terms > 2:
                        Q_squared = torch.bmm(Q_skew, Q_skew)
                        R.add_(Q_squared, alpha=2.0)

                        Q_power = Q_squared
                        for _ in range(3, num_neumann_terms - 1):
                            Q_power = torch.bmm(Q_power, Q_skew)
                            R.add_(Q_power, alpha=2.0)
                        Q_power = torch.bmm(Q_power, Q_skew)
                        R.add_(Q_power)
        else:
            id_mat = torch.eye(block_size, device=Q_skew.device).unsqueeze(0).expand(b, block_size, block_size)
            R = torch.linalg.solve(id_mat + Q_skew, id_mat - Q_skew, left=False)

        return R.to(previous_dtype)

    def _project_batch(self, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """Project weight parameters for Constrained OFT.

        Constructs the rotation from the weight, computes distance from identity,
        and scales the weight so the rotation stays within an epsilon-ball.

        Args:
            weight: (num_blocks, n_elements) parameter vectors.
            eps: Maximum Frobenius norm distance from identity.
        """
        Q_skew = self._pytorch_skew_symmetric(weight, self.block_size)
        bs = self.block_size
        eye = torch.eye(bs, dtype=weight.dtype, device=weight.device).unsqueeze(0)
        diff = Q_skew - eye
        norm = torch.norm(diff, p="fro", dim=(-2, -1), keepdim=True)
        scale = torch.clamp(norm, max=eps) / (norm + 1e-8)
        # Scale the original weight vectors
        return weight * (scale.squeeze(-1).squeeze(-1).unsqueeze(-1))

    def _compute_rotation(self) -> torch.Tensor:
        """Compute the block-diagonal orthogonal rotation blocks.

        Returns:
            (r, block_size, block_size) orthogonal blocks.
        """
        if self.input_is_parallel:
            # Row Tensor Parallel: each TP rank has a shard of the blocks, so use directly
            oft_r_parallel = self.oft_r
        else:
            # Column Tensor Parallel: each TP rank has the full set of blocks, so replicate across TP ranks
            oft_r_parallel = copy_to_tensor_model_parallel_region(self.oft_r, group=self.tp_group)

        if self.coft:
            with torch.no_grad():
                oft_r_parallel.copy_(self._project_batch(oft_r_parallel, eps=self.eps))

        R = self._cayley_batch(oft_r_parallel, self.block_size)

        R = self.dropout(R)

        return R

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply block-diagonal orthogonal rotation to input.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Rotated tensor of same shape.
        """
        required_dtype = x.dtype
        if required_dtype != self.oft_r.dtype:
            x = x.to(self.oft_r.dtype)

        R = self._compute_rotation()  # (r, block_size, block_size)

        orig_shape = x.shape
        rank = self.in_features // self.block_size if self.block_share else self.r

        if self.block_share:
            R = R.repeat(rank, 1, 1)

        # Use triton kernel when available: requires CUDA, block_size in [16, 256).
        # block_size >= 256 is slower in triton due to shared memory pressure and
        # fewer blocks to parallelize; einsum (cuBLAS batched matmul) wins there.
        if HAS_TRITON_OFT and x.is_cuda and 16 <= self.block_size < 256:
            x_2d = x.reshape(-1, orig_shape[-1])
            x_2d = triton_oft_r_single(x_2d, R)
            x = x_2d.reshape(orig_shape)
        else:
            x = x.reshape(*orig_shape[:-1], rank, self.block_size)
            x = torch.einsum("...rk, rkc -> ...rc", x, R)
            x = x.reshape(orig_shape)

        return x.to(required_dtype)

    def get_delta_weight(self) -> torch.Tensor:
        """Compute the full block-diagonal rotation matrix for weight merging.

        Returns:
            (in_features, in_features) block-diagonal orthogonal matrix.
        """
        R = self._compute_rotation()
        rank = self.r if not self.block_share else self.in_features // self.block_size
        if self.block_share:
            R = R.repeat(rank, 1, 1)
        return torch.block_diag(*[R[i] for i in range(rank)])

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple = (),
        metadata: Optional[Dict] = None,
    ) -> ShardedStateDict:
        """Create sharded state dictionary for distributed checkpointing.

        When input_is_parallel=True (RowParallel base layer), oft_r is TP-sharded
        along axis 0 (blocks dimension) since each TP rank holds rotation blocks
        for its local shard of features.

        When input_is_parallel=False (ColumnParallel base layer), oft_r is replicated
        across TP ranks since all ranks rotate the same full input.
        """
        sharded_state_dict = {}
        state_dict = self.state_dict(prefix="", keep_vars=True)

        # Determine TP sharding: when input_is_parallel, oft_r blocks are
        # partitioned across TP ranks (axis 0). Otherwise replicated.
        if self.input_is_parallel and not self.block_share:
            tp_axis_map = {"oft_r": 0}
        else:
            tp_axis_map = {}

        oft_r_sd = make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            tensor_parallel_layers_axis_map=tp_axis_map,
            sharded_offsets=sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata['dp_cp_group'],
        )

        if self.is_expert:
            ep_rank = parallel_state.get_expert_model_parallel_rank()
            edp_rank = parallel_state.get_expert_data_parallel_rank()
            dp_size = parallel_state.get_data_parallel_world_size()
            # TODO: This modification logic is in question and needs further verification.
            rank = (ep_rank + 1) * (edp_rank + 1) - 1 if dp_size == 1 else ep_rank
            for sd in [oft_r_sd]:
                for v in sd.values():
                    if hasattr(v, "replica_id"):
                        old_rid = v.replica_id
                        v.replica_id = (old_rid[0], rank, old_rid[2])

        sharded_state_dict.update(oft_r_sd)
        return sharded_state_dict


class OFTTopKRouter(AdapterWrapper):
    """Adapter wrapper that applies OFT rotation to MoE router gating logits.

    The router contains a small linear `gating: [hidden_dim -> num_experts]`.
    OFT rotates the input before the gating projection:
        logits = gating(R @ x)
    """

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        _trap_oft_applied_bridge(self, "TopKRouter")
        self.to_wrap._maintain_float32_expert_bias()
        jittered_input = self.to_wrap.apply_input_jitter(x)
        if self._adapter_enabled:
            jittered_input = self.adapter(jittered_input.contiguous())
        logits = self.to_wrap.gating(jittered_input)
        if self.to_wrap.config.moe_router_force_load_balancing:
            logits = apply_random_logits(logits)
        return self.to_wrap.routing(logits, *args, **kwargs)


class OFTLinear(AdapterWrapper):
    """Adapter wrapper that applies OFT rotation to non-fused linear layers.

    Rotates the input x via a block-diagonal orthogonal matrix R, then passes
    through the base linear: y = W @ (R @ x).

    When the base weight is FP8 (QOFT mode — set by the FP8 bridge), the
    weight is dequantised to BF16 on every forward using the block-wise
    ``weight_scale_inv`` buffer and injected into the base linear so that TE
    handles TP / SP / FP8-compute transparently.  A ``saved_tensors_hooks``
    context ensures autograd stores the compact ``(w_fp8, scale_inv)`` handle
    instead of the BF16 copy, and re-dequantises on the fly during backward.
    This gives ~50 % weight memory reduction without gradient checkpointing.

    Args:
        to_wrap: Base linear module.
        adapter: OFTRotationModule instance.
    """

    def __init__(self, to_wrap: nn.Module, adapter: OFTRotationModule):
        super().__init__(to_wrap, adapter)
        _clear_disabled_bias_parameters(self.to_wrap)
        # Detect quantised base weights (FP8, INT4, or NVFP4).
        # Standard linears have .weight; grouped expert linears (TEGroupedLinear)
        # have .weight0 ... .weight{num_gemms-1}, one per local expert on this rank.
        num_gemms = getattr(self.to_wrap, "num_gemms", 0)
        if num_gemms > 0:
            self._weight_names = [f"weight{i}" for i in range(num_gemms)]
            first_w = getattr(self.to_wrap, "weight0")
        elif hasattr(self.to_wrap, "weight"):
            self._weight_names = ["weight"]
            first_w = self.to_wrap.weight
        else:
            self._weight_names = []
            first_w = None

        # Direct FP8: native float8 parameters or BF16 materialized payload with
        # the sibling weight_scale_inv buffer from the distributed checkpoint.
        first_scale_inv = (
            getattr(self.to_wrap, f"{self._weight_names[0]}_scale_inv", None)
            if self._weight_names
            else None
        )
        self._base_fp8 = _is_direct_fp8_runtime_weight(first_w, first_scale_inv)

        # INT4: weight_packed buffer exists (set by DeepSeekV3INT4Bridge)
        first_packed = self._weight_names[0] + "_packed" if self._weight_names else ""
        self._base_int4 = hasattr(self.to_wrap, first_packed)
        self._base_nvfp4 = self._is_base_nvfp4()
        # NVFP4 grouped MoE buffers (registered by register_nvfp4_buffers_after_load):
        #   fc2: weight{N}_packed + weight_scale{N} + weight_double_scale{N}
        #   fc1: weight{N}_w_packed + weight{N}_v_packed + weight_scale{N} + weight_double_scale{N}
        # Distinguished from INT4 by presence of `weight_double_scale{N}` and absence of
        # `weight{N}_shape`. Detection takes precedence over `_base_int4` because NVFP4 fc2
        # collides with INT4 on the `weight{N}_packed` attribute name.
        self._base_nvfp4_grouped = self._detect_nvfp4_grouped_buffers()
        # If NVFP4 grouped buffers are present, suppress INT4 detection — they share the
        # `weight{N}_packed` attribute name and would otherwise be misrouted.
        if self._base_nvfp4_grouped:
            self._base_int4 = False
        self._int4_active_expert_chunk_size = get_int4_active_expert_chunk_size()
        self._int4_grouped_chunk_backend = get_int4_grouped_chunk_backend()
        self._int4_te_chunk_modules: Dict[
            Tuple[int, int, int, str, Optional[int], str], nn.Module
        ] = {}

    def _is_base_nvfp4(self) -> bool:
        """Return whether the wrapped module currently carries ModelOpt NVFP4 weight state."""
        if not self._weight_names:
            return False
        weight = getattr(self.to_wrap, self._weight_names[0], None)
        quantizer = getattr(self.to_wrap, "weight_quantizer", None)
        return (
            getattr(weight, "dtype", None) == torch.uint8
            and quantizer is not None
            and self._get_nvfp4_quantizer_buffer(quantizer, "_scale", self._weight_names[0]) is not None
            and self._get_nvfp4_scale_2(quantizer, self._weight_names[0]) is not None
        )

    def _detect_nvfp4_grouped_buffers(self) -> bool:
        """Return whether the wrapped grouped-MoE module carries NVFP4 per-expert buffers.

        Looks for the per-expert buffer layout from ``register_nvfp4_buffers_after_load``:
            - ``weight_scale{N}`` + ``weight_double_scale{N}`` for the first expert N
            - plus either ``weight{N}_packed`` (fc2) or ``weight{N}_w_packed`` (fc1 SwiGLU).
        Returns ``False`` for dense linears (no ``num_gemms`` / no ``weight{N}`` namespace).
        """
        # Grouped only: require numbered weight names (weight0, weight1, ...)
        if len(self._weight_names) < 1:
            return False
        first = self._weight_names[0]
        if first == "weight":
            return False
        suffix = first[len("weight") :]
        if not suffix.isdigit():
            return False
        has_packed = hasattr(self.to_wrap, f"{first}_packed") or hasattr(
            self.to_wrap, f"{first}_w_packed"
        )
        return has_packed and self._nvfp4_grouped_scale_suffix(suffix) is not None

    def _nvfp4_grouped_scale_suffix(self, local_suffix: str) -> Optional[str]:
        """Return the quantizer-buffer suffix for a local grouped expert.

        Packed expert buffers are keyed by local runtime names (``weight0``),
        while EP-sharded checkpoint quantizer buffers can retain global expert
        suffixes (for example ``weight_scale96``). Prefer an exact suffix match;
        otherwise map local expert order to sorted global scale suffixes.
        """
        if hasattr(self.to_wrap, f"weight_scale{local_suffix}") and hasattr(
            self.to_wrap, f"weight_double_scale{local_suffix}"
        ):
            return local_suffix

        scale_re = re.compile(r"^weight_scale(\d+)$")
        double_re = re.compile(r"^weight_double_scale(\d+)$")
        buffer_names = set(getattr(self.to_wrap, "_buffers", {}))
        scale_suffixes = {
            match.group(1)
            for name in buffer_names
            if (match := scale_re.match(name)) is not None
        }
        double_suffixes = {
            match.group(1)
            for name in buffer_names
            if (match := double_re.match(name)) is not None
        }
        common_suffixes = sorted(scale_suffixes & double_suffixes, key=int)
        if not common_suffixes:
            return None

        local_suffixes = [
            name[len("weight") :]
            for name in self._weight_names
            if name.startswith("weight") and name[len("weight") :].isdigit()
        ]
        try:
            position = local_suffixes.index(local_suffix)
        except ValueError:
            return None
        if position >= len(common_suffixes):
            return None
        return common_suffixes[position]

    def _is_base_fp8(self) -> bool:
        """Return whether the wrapped module currently carries direct FP8 weight state."""
        if not self._weight_names:
            return False
        weight = getattr(self.to_wrap, self._weight_names[0], None)
        scale_inv = getattr(self.to_wrap, f"{self._weight_names[0]}_scale_inv", None)
        return _is_direct_fp8_runtime_weight(weight, scale_inv)

    def _debug_fp8_detection_miss(self, x: torch.Tensor) -> None:
        if not _oft_fp8_debug_enabled() or not self._weight_names:
            return
        if self._base_fp8 or self._is_base_fp8():
            return

        count = getattr(OFTLinear, "_fp8_debug_miss_count", 0)
        if count >= _oft_fp8_debug_limit():
            return
        OFTLinear._fp8_debug_miss_count = count + 1

        first_name = self._weight_names[0]
        weight = getattr(self.to_wrap, first_name, None)
        scale_inv = getattr(self.to_wrap, f"{first_name}_scale_inv", None)
        _oft_fp8_debug_log(
            "detection_miss",
            wrapper=type(self).__name__,
            to_wrap=type(self.to_wrap).__name__,
            weight_name=first_name,
            cached_base_fp8=self._base_fp8,
            weight=weight,
            scale_inv=scale_inv,
            has_scale_inv=scale_inv is not None,
            input=x,
        )

    def _debug_nvfp4_detection_miss(self, x: torch.Tensor) -> None:
        if not self._weight_names:
            return
        weight = getattr(self.to_wrap, self._weight_names[0], None)
        if getattr(weight, "dtype", None) != torch.uint8:
            return

        count = getattr(OFTLinear, "_nvfp4_debug_miss_count", 0)
        limit = _oft_nvfp4_debug_limit() if _oft_nvfp4_debug_enabled() else 1
        if count >= limit:
            return
        OFTLinear._nvfp4_debug_miss_count = count + 1

        quantizer = getattr(self.to_wrap, "weight_quantizer", None)
        quantizer_attrs = []
        if quantizer is not None:
            quantizer_attrs = [
                name
                for name in sorted(dir(quantizer))
                if name.startswith(("_scale", "_double_scale", "_amax"))
            ][:32]
        buffer_names = list(getattr(quantizer, "_buffers", {}).keys()) if quantizer is not None else []
        param_names = list(getattr(quantizer, "_parameters", {}).keys()) if quantizer is not None else []

        first_name = self._weight_names[0]
        has_scale = self._get_nvfp4_quantizer_buffer(quantizer, "_scale", first_name) is not None if quantizer else False
        has_scale_2 = self._get_nvfp4_scale_2(quantizer, first_name) is not None if quantizer else False
        rank_prefix = _debug_rank_prefix()
        message = (
            "[OFT NVFP4 debug] detection miss "
            f"{rank_prefix} wrapper={type(self).__name__} to_wrap={type(self.to_wrap).__name__} "
            f"num_gemms={getattr(self.to_wrap, 'num_gemms', None)} "
            f"weight_names_sample={self._weight_names[:4]} first_weight={first_name} "
            f"first_weight_dtype={getattr(weight, 'dtype', None)} first_weight_shape={tuple(getattr(weight, 'shape', ())) if weight is not None else None} "
            f"x_dtype={x.dtype} x_shape={tuple(x.shape)} "
            f"has_quantizer={quantizer is not None} has_scale={has_scale} has_scale_2={has_scale_2} "
            f"quantizer_type={type(quantizer).__name__ if quantizer is not None else None} "
            f"quantizer_attrs={quantizer_attrs} quantizer_buffers={buffer_names[:32]} "
            f"quantizer_params={param_names[:32]}"
        )
        logger.warning("%s", message)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _trap_oft_applied_bridge(self, "Linear")
        if self._adapter_enabled:
            x = self.adapter(x.contiguous())

        if self._base_fp8 or self._is_base_fp8():
            return self._forward_fp8(x, *args, **kwargs)
        # NVFP4 grouped MoE: check BEFORE INT4 because fc2 NVFP4 also registers
        # `weight{N}_packed` and would otherwise be misrouted to INT4.
        if getattr(self, "_base_nvfp4_grouped", False) or self._detect_nvfp4_grouped_buffers():
            self._base_nvfp4_grouped = True
            # NVFP4 fc2 shares the `weight{N}_packed` attribute name with
            # INT4. If buffers appeared after wrapper construction, refresh
            # the cached dispatch state before checking the INT4 branch.
            self._base_int4 = False
            return self._forward_nvfp4_grouped_buffers(x, *args, **kwargs)
        if self._base_int4:
            return self._forward_int4(x, *args, **kwargs)
        if getattr(self, "_base_nvfp4", False) or self._is_base_nvfp4():
            return self._forward_nvfp4(x, *args, **kwargs)
        self._debug_fp8_detection_miss(x)
        self._debug_nvfp4_detection_miss(x)

        # --- Standard BF16 path ---
        if not hasattr(self.to_wrap, "config"):
            return self.to_wrap(x, *args, **kwargs), None
        linear_output, bias, _ = self.base_linear_forward(x, *args, **kwargs)
        return linear_output, bias

    @staticmethod
    def _nvfp4_weight_suffix(name: str) -> str:
        if name == "weight":
            return ""
        if name.startswith("weight"):
            return name[len("weight") :]
        return ""

    def _get_nvfp4_quantizer_buffer(self, quantizer: nn.Module, attr: str, name: str) -> Optional[torch.Tensor]:
        suffix = self._nvfp4_weight_suffix(name)
        if suffix:
            value = getattr(quantizer, f"{attr}{suffix}", None)
            if value is not None:
                return value
        value = getattr(quantizer, attr, None)
        if value is None:
            return None
        if suffix and isinstance(value, torch.Tensor) and suffix.isdigit():
            expert_idx = int(suffix)
            if value.ndim > 0 and value.shape[0] == len(self._weight_names):
                return value[expert_idx]
        return value

    def _get_nvfp4_scale_2(self, quantizer: nn.Module, name: str) -> Optional[torch.Tensor]:
        scale_2 = self._get_nvfp4_quantizer_buffer(quantizer, "_double_scale", name)
        if scale_2 is not None:
            return scale_2

        amax = self._get_nvfp4_quantizer_buffer(quantizer, "_amax", name)
        if amax is None:
            return None

        from megatron.bridge.models.conversion.low_precision.nvfp4 import NVFP4_AMAX_SCALE

        return amax.to(torch.float32) / NVFP4_AMAX_SCALE

    def _nvfp4_weight_shape(self, packed: torch.Tensor, scale: torch.Tensor, name: str = "weight") -> torch.Tensor:
        named_weight_shape = getattr(self.to_wrap, f"{name}_shape", None)
        if named_weight_shape is not None:
            return named_weight_shape
        weight_shape = getattr(self.to_wrap, "weight_shape", None)
        if weight_shape is not None:
            return weight_shape

        device = getattr(packed, "device", None) or getattr(scale, "device", torch.device("cpu"))
        if getattr(scale, "ndim", 0) >= 2:
            from megatron.bridge.models.conversion.low_precision.nvfp4 import NVFP4_GROUP_SIZE

            return torch.tensor(
                [int(scale.shape[-2]), int(scale.shape[-1]) * NVFP4_GROUP_SIZE],
                dtype=torch.int64,
                device=device,
            )

        return torch.tensor(
            [int(packed.shape[0]), int(packed.shape[1]) * 2],
            dtype=torch.int64,
            device=device,
        )

    def _forward_nvfp4_te_linear(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """NVFP4 forward for TE-backed single-weight linears."""
        from megatron.bridge.models.conversion.low_precision.nvfp4 import dequantize_nvfp4

        name = "weight"
        packed = getattr(self.to_wrap, name)
        quantizer = getattr(self.to_wrap, "weight_quantizer")
        scale = self._get_nvfp4_quantizer_buffer(quantizer, "_scale", name)
        scale_2 = self._get_nvfp4_scale_2(quantizer, name)
        shape = self._nvfp4_weight_shape(packed, scale, name)

        w_compute = dequantize_nvfp4(packed, scale, scale_2, shape, dtype=x.dtype, device=x.device)
        w_compute_ptr = w_compute.data_ptr()
        bias = _get_active_bias_tensor(self.to_wrap)
        has_bias = bias is not None
        tp_group = getattr(self.to_wrap, "_tp_group", None)

        def pack(tensor):
            if tensor.data_ptr() == w_compute_ptr:
                return (
                    packed,
                    scale,
                    scale_2,
                    shape,
                    tensor.dtype,
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    tensor.storage_offset(),
                )
            return tensor

        def unpack(packed_tuple):
            if isinstance(packed_tuple, tuple) and len(packed_tuple) == 8:
                p, s, s2, sh, dtype, saved_shape, saved_stride, saved_storage_offset = packed_tuple
                base = dequantize_nvfp4(p, s, s2, sh, dtype=dtype, device=p.device)
                return base.as_strided(saved_shape, saved_stride, saved_storage_offset)
            return packed_tuple

        try:
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                if HAVE_TE_ROW_LINEAR and isinstance(self.to_wrap, TERowParallelLinear):
                    output_parallel = F.linear(x.to(w_compute.dtype), w_compute, None)
                    if getattr(self.to_wrap, "explicit_expert_comm", False):
                        output_ = output_parallel
                    elif getattr(self.to_wrap, "sequence_parallel", False):
                        output_ = reduce_scatter_to_sequence_parallel_region(output_parallel, group=tp_group)
                    else:
                        output_ = reduce_from_tensor_model_parallel_region(output_parallel, group=tp_group)

                    if getattr(self.to_wrap, "te_return_bias", False):
                        return output_, bias
                    return output_ + bias if has_bias else output_, None

                if HAVE_TE_COL_LINEAR and isinstance(self.to_wrap, TEColumnParallelLinear):
                    if (
                        getattr(self.to_wrap, "allreduce_dgrad", False)
                        or getattr(self.to_wrap, "sequence_parallel", False)
                        or getattr(self.to_wrap, "explicit_expert_comm", False)
                        or getattr(self.to_wrap, "disable_grad_reduce", False)
                    ):
                        input_parallel = x
                    else:
                        input_parallel = copy_to_tensor_model_parallel_region(x, group=tp_group)

                    output_parallel = F.linear(
                        input_parallel.to(w_compute.dtype),
                        w_compute,
                        None if getattr(self.to_wrap, "te_return_bias", False) else (bias if has_bias else None),
                    )

                    if getattr(self.to_wrap, "gather_output", False):
                        output = gather_from_tensor_model_parallel_region(output_parallel, group=tp_group)
                    else:
                        output = output_parallel

                    if getattr(self.to_wrap, "te_return_bias", False):
                        return output, bias
                    return output, None

                if getattr(self.to_wrap, "te_return_bias", False) and has_bias:
                    out = F.linear(x.to(w_compute.dtype), w_compute, None)
                    return out, bias
                out = F.linear(x.to(w_compute.dtype), w_compute, bias if has_bias else None)
                return out, None
        finally:
            del w_compute

    def _forward_nvfp4_grouped_eager(
        self, x: torch.Tensor, tokens_per_expert: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Grouped expert NVFP4 forward that dequantizes active experts and uses F.linear."""
        from megatron.bridge.models.conversion.low_precision.nvfp4 import dequantize_nvfp4

        token_splits = self._normalize_tokens_per_expert(tokens_per_expert)
        if len(token_splits) != len(self._weight_names):
            raise ValueError(
                f"Expected {len(self._weight_names)} expert token splits for grouped NVFP4 path, "
                f"got {len(token_splits)}"
            )

        quantizer = getattr(self.to_wrap, "weight_quantizer")
        first_name = self._weight_names[0]
        first_packed = getattr(self.to_wrap, first_name)
        first_scale = self._get_nvfp4_quantizer_buffer(quantizer, "_scale", first_name)
        first_shape = self._nvfp4_weight_shape(first_packed, first_scale, first_name)
        out_features = int(first_shape.reshape(-1)[0].item())
        output_chunks = [
            x.new_empty((split, out_features), dtype=x.dtype) for split in token_splits
        ]
        active_experts = [idx for idx, split in enumerate(token_splits) if split > 0]
        if not active_experts:
            return x.new_empty((0, out_features), dtype=x.dtype), None

        input_chunks = list(torch.split(x, token_splits, dim=0))
        ptr_to_handle = {}
        expert_weights = {}

        for idx in active_experts:
            name = self._weight_names[idx]
            packed = getattr(self.to_wrap, name)
            scale = self._get_nvfp4_quantizer_buffer(quantizer, "_scale", name)
            scale_2 = self._get_nvfp4_scale_2(quantizer, name)
            shape = self._nvfp4_weight_shape(packed, scale, name)
            w_compute = dequantize_nvfp4(packed, scale, scale_2, shape, dtype=x.dtype, device=x.device)
            ptr_to_handle[w_compute.data_ptr()] = (packed, scale, scale_2, shape, w_compute.dtype)
            expert_weights[idx] = w_compute

        def pack(tensor):
            handle = ptr_to_handle.get(tensor.data_ptr())
            if handle is not None:
                return (
                    handle[0],
                    handle[1],
                    handle[2],
                    handle[3],
                    handle[4],
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    tensor.storage_offset(),
                )
            return tensor

        def unpack(packed_tuple):
            if isinstance(packed_tuple, tuple) and len(packed_tuple) == 8:
                p, s, s2, sh, dtype, saved_shape, saved_stride, saved_storage_offset = packed_tuple
                base = dequantize_nvfp4(p, s, s2, sh, dtype=dtype, device=p.device)
                return base.as_strided(saved_shape, saved_stride, saved_storage_offset)
            return packed_tuple

        try:
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                for idx in active_experts:
                    bias = _get_active_bias_tensor(self.to_wrap, f"bias{idx}")
                    output_chunks[idx] = F.linear(input_chunks[idx], expert_weights[idx], bias)
        finally:
            del expert_weights

        return torch.cat(output_chunks, dim=0), None

    def _forward_nvfp4(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if len(self._weight_names) != 1:
            if len(args) == 0:
                raise ValueError("Grouped OFTLinear NVFP4 fallback requires tokens_per_expert")
            return self._forward_nvfp4_grouped_eager(x, args[0])
        return self._forward_nvfp4_te_linear(x)

    def _forward_nvfp4_grouped_buffers(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Grouped MoE NVFP4 forward reading per-expert NVFP4 buffers.

        Reads ``weight{N}_packed`` (fc2) or ``weight{N}_w_packed`` + ``weight{N}_v_packed``
        (fc1 SwiGLU halves), together with ``weight_scale{N}`` and ``weight_double_scale{N}``.
        Per active expert, dequantizes to bf16 and runs ``F.linear`` on the token chunk.

        ``tokens_per_expert`` is expected as the first positional arg (matches the
        INT4 grouped contract used by ``_forward_int4_grouped_chunked``).
        """
        from megatron.bridge.models.conversion.low_precision.nvfp4 import (
            NVFP4_GROUP_SIZE,
            dequantize_nvfp4,
        )

        if len(args) == 0:
            raise ValueError(
                "Grouped OFTLinear NVFP4 forward requires tokens_per_expert as the first positional arg"
            )
        token_splits = self._normalize_tokens_per_expert(args[0])
        if len(token_splits) != len(self._weight_names):
            raise ValueError(
                f"Expected {len(self._weight_names)} expert token splits for grouped NVFP4 buffer path, "
                f"got {len(token_splits)}"
            )

        # Derive (out, in) from the first expert's scale tensor: scale is [out, in/16].
        first_name = self._weight_names[0]
        first_suffix = first_name[len("weight") :]
        first_scale_suffix = self._nvfp4_grouped_scale_suffix(first_suffix)
        first_scale = (
            getattr(self.to_wrap, f"weight_scale{first_scale_suffix}", None)
            if first_scale_suffix is not None
            else None
        )
        if first_scale is None or first_scale.ndim < 2:
            raise ValueError(
                f"NVFP4 grouped forward requires 2-D weight_scale for local expert {first_suffix}, got "
                f"{None if first_scale is None else tuple(first_scale.shape)}"
            )
        out_features = int(first_scale.shape[-2])
        in_features = int(first_scale.shape[-1]) * NVFP4_GROUP_SIZE

        # Megatron's TEGroupedLinear inputs are typically [total_tokens, hidden],
        # but when ``config.fp4`` or ``config.fp8`` is on, Megatron applies
        # ``Fp8Padding`` upstream which can pad the token axis with trailing
        # junk rows for kernel alignment without updating ``tokens_per_expert``
        # (the saved ``actual_tokens_per_expert`` is propagated through).
        # Handle the common cases: sum matches dim 0 (no padding), sum matches
        # dim 1 (transposed input), or total < x.shape[0] (trailing junk).
        total_tokens = int(sum(token_splits))
        trim_to = None  # if set, the valid prefix length after computation
        if total_tokens == x.shape[0]:
            split_dim = 0
            x_to_split = x
        elif x.ndim >= 2 and total_tokens == x.shape[1]:
            split_dim = 1
            x_to_split = x
        elif total_tokens < x.shape[0]:
            # Trailing-junk padding: operate on the first ``total_tokens`` rows
            # and pad the output back to the original x.shape[0] with zeros for
            # the junk rows (caller's downstream unpadding will discard them).
            split_dim = 0
            x_to_split = x[:total_tokens]
            trim_to = x.shape[0]
        else:
            # Sum doesn't match either dim. Try TE-aligned padding (multiples of 16).
            align = 16
            padded_splits = [
                ((s + align - 1) // align) * align if s > 0 else 0 for s in token_splits
            ]
            if sum(padded_splits) == x.shape[0]:
                token_splits = padded_splits
                split_dim = 0
                x_to_split = x
            elif x.ndim >= 2 and sum(padded_splits) == x.shape[1]:
                token_splits = padded_splits
                split_dim = 1
                x_to_split = x
            else:
                raise ValueError(
                    "NVFP4 grouped forward: cannot reconcile token_splits "
                    f"(sum={total_tokens}, padded_sum={sum(padded_splits)}) with x.shape "
                    f"{tuple(x.shape)}; expected sum to match either dim 0 or dim 1."
                )

        output_chunks = [
            x.new_empty((split, out_features), dtype=x.dtype) for split in token_splits
        ]
        active_experts = [idx for idx, split in enumerate(token_splits) if split > 0]
        if not active_experts:
            return x.new_empty((0, out_features), dtype=x.dtype), None

        input_chunks = list(torch.split(x_to_split, token_splits, dim=split_dim))

        for idx in active_experts:
            name = self._weight_names[idx]
            suffix = name[len("weight") :]
            scale_suffix = self._nvfp4_grouped_scale_suffix(suffix)
            if scale_suffix is None:
                raise ValueError(
                    f"NVFP4 grouped forward: missing weight_scale/weight_double_scale for local expert {suffix}"
                )
            scale = getattr(self.to_wrap, f"weight_scale{scale_suffix}")
            double_scale = getattr(self.to_wrap, f"weight_double_scale{scale_suffix}")

            packed_full = getattr(self.to_wrap, f"{name}_packed", None)
            if packed_full is None:
                # SwiGLU fc1: reassemble fused packed weight from gate (_w) + up (_v) halves.
                w_half = getattr(self.to_wrap, f"{name}_w_packed", None)
                v_half = getattr(self.to_wrap, f"{name}_v_packed", None)
                if w_half is None or v_half is None:
                    raise ValueError(
                        f"NVFP4 grouped forward: expert {idx} missing both `{name}_packed` and "
                        f"`{name}_w_packed`/`{name}_v_packed` buffers"
                    )
                packed = torch.cat([w_half, v_half], dim=0)
                local_out = int(w_half.shape[0]) + int(v_half.shape[0])
                local_in = int(w_half.shape[1]) * 2
            else:
                packed = packed_full
                local_out = int(packed.shape[0])
                local_in = int(packed.shape[1]) * 2

            w_compute = dequantize_nvfp4(
                packed,
                scale,
                double_scale,
                (local_out, local_in),
                device=x.device,
                dtype=x.dtype,
            )
            bias = _get_active_bias_tensor(self.to_wrap, f"bias{idx}")
            # F.linear expects [..., in_features]; for split_dim==1 the chunk is
            # [hidden, tokens] so transpose to [tokens, hidden] for the GEMM, then
            # transpose back when concatenating.
            chunk_input = input_chunks[idx]
            if split_dim == 1:
                chunk_input = chunk_input.transpose(0, 1).contiguous()
            chunk_output = F.linear(chunk_input, w_compute, bias)
            if split_dim == 1:
                chunk_output = chunk_output.transpose(0, 1).contiguous()
            output_chunks[idx] = chunk_output

        output = torch.cat(output_chunks, dim=split_dim)
        # Restore trailing-junk padding so the caller sees a tensor of the
        # original x.shape[0] (downstream unpadding will discard the pad rows).
        if trim_to is not None and output.shape[0] < trim_to:
            pad = output.new_zeros(
                (trim_to - output.shape[0], *output.shape[1:]),
                dtype=output.dtype,
            )
            output = torch.cat([output, pad], dim=0)
        return output, None

    @staticmethod
    def _normalize_tokens_per_expert(tokens_per_expert: Any) -> list[int]:
        if isinstance(tokens_per_expert, torch.Tensor):
            return [int(v) for v in tokens_per_expert.detach().cpu().tolist()]
        return [int(v) for v in tokens_per_expert]

    def _forward_int4_grouped_chunked(
        self, x: torch.Tensor, tokens_per_expert: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Grouped expert INT4 forward that dequantizes only active experts in chunks.

        This keeps the current low-memory behavior but prefers a chunk-local
        TE grouped GEMM when available. If TE cannot be used for a chunk, it
        falls back to the existing per-expert F.linear path.
        """
        from megatron.bridge.models.conversion.low_precision.int4 import dequantize_int4

        token_splits = self._normalize_tokens_per_expert(tokens_per_expert)
        if len(token_splits) != len(self._weight_names):
            raise ValueError(
                f"Expected {len(self._weight_names)} expert token splits for grouped INT4 path, "
                f"got {len(token_splits)}"
            )

        if getattr(self.to_wrap, "te_return_bias", False):
            logger.warning(
                "Falling back to eager INT4 grouped path for %s because return_bias=True is "
                "not supported by the chunked grouped INT4 fallback.",
                type(self.to_wrap).__name__,
            )
            return self._forward_int4_eager(x, tokens_per_expert)

        out_shape = getattr(self.to_wrap, f"{self._weight_names[0]}_shape")
        out_features = int(out_shape[0].item())
        output_chunks = [
            x.new_empty((split, out_features), dtype=x.dtype) for split in token_splits
        ]
        active_experts = [idx for idx, split in enumerate(token_splits) if split > 0]
        if not active_experts:
            return x.new_empty((0, out_features), dtype=x.dtype), None

        input_chunks = list(torch.split(x, token_splits, dim=0))
        chunk_size = max(1, self._int4_active_expert_chunk_size)

        for chunk_start in range(0, len(active_experts), chunk_size):
            chunk_experts = active_experts[chunk_start : chunk_start + chunk_size]
            ptr_to_handle = {}
            chunk_weights = {}
            chunk_expected_shapes = {}
            chunk_actual_shapes = {}

            for idx in chunk_experts:
                name = self._weight_names[idx]
                packed = getattr(self.to_wrap, f"{name}_packed")
                scale = getattr(self.to_wrap, f"{name}_scale")
                shape = getattr(self.to_wrap, f"{name}_shape")
                expected_input = input_chunks[idx].shape[-1]
                expected_output = int(shape.reshape(-1)[0].item())
                w_compute = dequantize_int4(packed, scale, shape, device=x.device).to(x.dtype)
                actual_shape = tuple(w_compute.shape)
                ptr_to_handle[w_compute.data_ptr()] = (packed, scale, shape, w_compute.dtype)
                chunk_weights[idx] = w_compute
                chunk_expected_shapes[idx] = (expected_output, expected_input)
                chunk_actual_shapes[idx] = actual_shape

            def pack(tensor):
                handle = ptr_to_handle.get(tensor.data_ptr())
                if handle is not None:
                    return (
                        handle[0],
                        handle[1],
                        handle[2],
                        handle[3],
                        tuple(tensor.shape),
                        tuple(tensor.stride()),
                        tensor.storage_offset(),
                    )
                return tensor

            def unpack(packed_tuple):
                if isinstance(packed_tuple, tuple) and len(packed_tuple) == 7:
                    p, s, sh, dtype, saved_shape, saved_stride, saved_storage_offset = packed_tuple
                    base = dequantize_int4(p, s, sh, device=p.device).to(dtype)
                    return base.as_strided(saved_shape, saved_stride, saved_storage_offset)
                if isinstance(packed_tuple, tuple) and len(packed_tuple) == 4:
                    p, s, sh, dtype = packed_tuple
                    return dequantize_int4(p, s, sh, device=p.device).to(dtype)
                return packed_tuple

            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                chunk_module = None
                if self._int4_grouped_chunk_backend == "te":
                    chunk_module = self._get_or_create_int4_te_chunk_module(
                        num_gemms=len(chunk_experts),
                        local_input_size=input_chunks[chunk_experts[0]].shape[-1],
                        local_output_size=out_features,
                        device=x.device,
                        dtype=x.dtype,
                    )
                te_chunk_succeeded = False
                if chunk_module is not None:
                    chunk_input = torch.cat([input_chunks[idx] for idx in chunk_experts], dim=0)
                    chunk_splits = [token_splits[idx] for idx in chunk_experts]
                    try:
                        for local_idx, expert_idx in enumerate(chunk_experts):
                            getattr(chunk_module, f"weight{local_idx}").data = chunk_weights[expert_idx]
                            if _module_bias_enabled(self.to_wrap):
                                bias_param = getattr(chunk_module, f"bias{local_idx}", None)
                                if bias_param is not None:
                                    source_bias = _get_active_bias_tensor(self.to_wrap, f"bias{expert_idx}")
                                    if source_bias is not None:
                                        bias_param.data = source_bias.data.to(
                                            device=bias_param.device, dtype=bias_param.dtype
                                        )
                                    else:
                                        bias_param.data.zero_()
                        chunk_output, _ = chunk_module(chunk_input, chunk_splits)
                        for expert_idx, expert_output in zip(
                            chunk_experts, torch.split(chunk_output, chunk_splits, dim=0)
                        ):
                            output_chunks[expert_idx] = expert_output
                        te_chunk_succeeded = True
                    except Exception as exc:
                        logger.warning(
                            "Falling back to Python INT4 chunked path for %s because TE chunk "
                            "execution failed for chunk size %d: %s",
                            type(self.to_wrap).__name__,
                            len(chunk_experts),
                            exc,
                        )
                    finally:
                        for local_idx in range(len(chunk_experts)):
                            weight_param = getattr(chunk_module, f"weight{local_idx}")
                            weight_param.data = _empty_storage_view(weight_param.data)

                if not te_chunk_succeeded:
                    for idx in chunk_experts:
                        name = self._weight_names[idx]
                        expected_output, expected_input = chunk_expected_shapes[idx]
                        actual_shape = chunk_actual_shapes[idx]
                        if actual_shape != (expected_output, expected_input):
                            logger.warning(
                                "Falling back to eager INT4 grouped path for %s because %s dequantized "
                                "to shape %s but the chunked F.linear fallback expected (%d, %d). "
                                "This usually means the grouped expert weight layout for this module is "
                                "not a plain F.linear-compatible [out, in] tensor.",
                                type(self.to_wrap).__name__,
                                name,
                                actual_shape,
                                expected_output,
                                expected_input,
                            )
                            return self._forward_int4_eager(x, tokens_per_expert)
                        bias = _get_active_bias_tensor(self.to_wrap, f"bias{idx}")
                        output_chunks[idx] = F.linear(input_chunks[idx], chunk_weights[idx], bias)

            del chunk_weights

        return torch.cat(output_chunks, dim=0), None

    def _get_int4_te_grouped_chunk_cls_and_mode(self):
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

    def _get_int4_te_grouped_tp_size(self) -> int:
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

    def _get_or_create_int4_te_chunk_module(
        self,
        *,
        num_gemms: int,
        local_input_size: int,
        local_output_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[nn.Module]:
        chunk_cls, parallel_mode = self._get_int4_te_grouped_chunk_cls_and_mode()
        if chunk_cls is None or parallel_mode is None:
            return None

        cache_key = (
            num_gemms,
            local_input_size,
            local_output_size,
            device.type,
            device.index,
            str(dtype),
        )
        cached = self._int4_te_chunk_modules.get(cache_key)
        if cached is not None:
            cached.train(self.training)
            return cached

        tp_size = self._get_int4_te_grouped_tp_size()
        input_size = local_input_size
        output_size = local_output_size
        if getattr(self.to_wrap, "explicit_expert_comm", False) and tp_size > 1:
            if parallel_mode == "column":
                output_size *= tp_size
            elif parallel_mode == "row":
                input_size *= tp_size

        chunk_module = chunk_cls(
            num_gemms=num_gemms,
            input_size=input_size,
            output_size=output_size,
            config=self.to_wrap.config,
            init_method=lambda w: None,
            bias=_module_bias_enabled(self.to_wrap),
            skip_bias_add=False,
            is_expert=True,
            pg_collection=getattr(self.to_wrap, "_pg_collection", None),
        )
        _clear_disabled_bias_parameters(chunk_module)
        has_meta_tensors = any(param.device.type == "meta" for param in chunk_module.parameters())
        if not has_meta_tensors:
            has_meta_tensors = any(buffer.device.type == "meta" for buffer in chunk_module.buffers())

        if has_meta_tensors:
            to_empty_if_meta_device(chunk_module, device=device)
            chunk_module.to(dtype=dtype)
        else:
            chunk_module.to(device=device, dtype=dtype)
        chunk_module.train(self.training)
        if hasattr(chunk_module, "disable_parameter_transpose_cache"):
            chunk_module.disable_parameter_transpose_cache = True
        if hasattr(chunk_module, "is_first_microbatch"):
            chunk_module.is_first_microbatch = False
        for param in chunk_module.parameters():
            param.requires_grad_(False)

        logger.debug(
            "Enabled TE chunked INT4 grouped path for %s with num_gemms=%d, mode=%s, tp_size=%d, "
            "device=%s, dtype=%s",
            type(self.to_wrap).__name__,
            num_gemms,
            parallel_mode,
            tp_size,
            device,
            dtype,
        )
        self._int4_te_chunk_modules[cache_key] = chunk_module
        return chunk_module

    def _forward_int4_eager(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Original eager INT4 path that materializes all local weights for the wrapped module."""
        from megatron.bridge.models.conversion.low_precision.int4 import dequantize_int4

        weight_param = getattr(self.to_wrap, self._weight_names[0], None) if self._weight_names else None
        if len(self._weight_names) == 1 and (
            (HAVE_TE_LINEAR and isinstance(self.to_wrap, TELinear))
            or (weight_param is not None and type(weight_param) is not nn.Parameter)
        ):
            return self._forward_int4_te_linear(x)

        int4_originals = {}  # name → (packed, scale, shape) to restore
        ptr_to_handle = {}   # data_ptr → (packed, scale, shape, dtype) for hook

        for name in self._weight_names:
            packed = getattr(self.to_wrap, f"{name}_packed")
            scale = getattr(self.to_wrap, f"{name}_scale")
            shape = getattr(self.to_wrap, f"{name}_shape")

            w_compute = dequantize_int4(packed, scale, shape, device=x.device).to(x.dtype)
            int4_originals[name] = (packed, scale, shape)
            ptr_to_handle[w_compute.data_ptr()] = (packed, scale, shape, w_compute.dtype)
            getattr(self.to_wrap, name).data = w_compute

        def pack(tensor):
            handle = ptr_to_handle.get(tensor.data_ptr())
            if handle is not None:
                return handle
            return tensor

        def unpack(packed_tuple):
            if isinstance(packed_tuple, tuple) and len(packed_tuple) == 4:
                p, s, sh, dtype = packed_tuple
                return dequantize_int4(p, s, sh, device=p.device).to(dtype)
            return packed_tuple

        try:
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                if not hasattr(self.to_wrap, "config"):
                    return self.to_wrap(x, *args, **kwargs), None
                linear_output, bias, _ = self.base_linear_forward(x, *args, **kwargs)
                return linear_output, bias
        finally:
            # Restore empty weight — INT4 buffers remain.
            for name in int4_originals:
                weight_param = getattr(self.to_wrap, name)
                weight_param.data = _empty_storage_view(weight_param.data)

    def _forward_int4_te_linear(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """INT4 forward for TE-backed single-weight linears.

        TE linear weights are tensor subclasses, so rebinding ``weight.data`` to a
        plain dense tensor is not safe. For INT4 we instead dequantize the local
        shard and mirror the row/column/duplicated linear semantics explicitly.
        """
        from megatron.bridge.models.conversion.low_precision.int4 import dequantize_int4

        packed = getattr(self.to_wrap, "weight_packed")
        scale = getattr(self.to_wrap, "weight_scale")
        shape = getattr(self.to_wrap, "weight_shape")

        w_compute = dequantize_int4(packed, scale, shape, device=x.device).to(x.dtype)
        w_compute_ptr = w_compute.data_ptr()
        bias = _get_active_bias_tensor(self.to_wrap)
        has_bias = bias is not None
        tp_group = getattr(self.to_wrap, "_tp_group", None)

        def pack(tensor):
            if tensor.data_ptr() == w_compute_ptr:
                return (
                    packed,
                    scale,
                    shape,
                    tensor.dtype,
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    tensor.storage_offset(),
                )
            return tensor

        def unpack(packed_tuple):
            if isinstance(packed_tuple, tuple) and len(packed_tuple) == 7:
                p, s, sh, dtype, saved_shape, saved_stride, saved_storage_offset = packed_tuple
                base = dequantize_int4(p, s, sh, device=p.device).to(dtype)
                return base.as_strided(saved_shape, saved_stride, saved_storage_offset)
            if isinstance(packed_tuple, tuple) and len(packed_tuple) == 4:
                p, s, sh, dtype = packed_tuple
                return dequantize_int4(p, s, sh, device=p.device).to(dtype)
            return packed_tuple

        try:
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                if HAVE_TE_ROW_LINEAR and isinstance(self.to_wrap, TERowParallelLinear):
                    output_parallel = F.linear(x.to(w_compute.dtype), w_compute, None)
                    if getattr(self.to_wrap, "explicit_expert_comm", False):
                        output_ = output_parallel
                    elif getattr(self.to_wrap, "sequence_parallel", False):
                        output_ = reduce_scatter_to_sequence_parallel_region(output_parallel, group=tp_group)
                    else:
                        output_ = reduce_from_tensor_model_parallel_region(output_parallel, group=tp_group)

                    if getattr(self.to_wrap, "te_return_bias", False):
                        return output_, bias
                    return output_ + bias if has_bias else output_, None

                if HAVE_TE_COL_LINEAR and isinstance(self.to_wrap, TEColumnParallelLinear):
                    if (
                        getattr(self.to_wrap, "allreduce_dgrad", False)
                        or getattr(self.to_wrap, "sequence_parallel", False)
                        or getattr(self.to_wrap, "explicit_expert_comm", False)
                        or getattr(self.to_wrap, "disable_grad_reduce", False)
                    ):
                        input_parallel = x
                    else:
                        input_parallel = copy_to_tensor_model_parallel_region(x, group=tp_group)

                    output_parallel = F.linear(
                        input_parallel.to(w_compute.dtype),
                        w_compute,
                        None if getattr(self.to_wrap, "te_return_bias", False) else (bias if has_bias else None),
                    )

                    if getattr(self.to_wrap, "gather_output", False):
                        output = gather_from_tensor_model_parallel_region(output_parallel, group=tp_group)
                    else:
                        output = output_parallel

                    if getattr(self.to_wrap, "te_return_bias", False):
                        return output, bias
                    return output, None

                if getattr(self.to_wrap, "te_return_bias", False) and has_bias:
                    out = F.linear(x.to(w_compute.dtype), w_compute, None)
                    return out, bias
                out = F.linear(x.to(w_compute.dtype), w_compute, bias if has_bias else None)
                return out, None
        finally:
            del w_compute

    def _forward_int4(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward with INT4 base weight — no BF16 copy held by autograd.

        Same ``saved_tensors_hooks`` pattern as ``_forward_fp8`` but dequants
        from INT4 packed format (weight_packed + weight_scale + weight_shape).
        """
        if (
            getattr(self.to_wrap, "num_gemms", 0) > 0
            and self._int4_active_expert_chunk_size > 0
            and len(args) > 0
        ):
            return self._forward_int4_grouped_chunked(x, args[0])

        return self._forward_int4_eager(x, *args, **kwargs)

    def _is_te_single_weight_linear(self) -> bool:
        return (
            (HAVE_TE_LINEAR and isinstance(self.to_wrap, TELinear))
            or (HAVE_TE_ROW_LINEAR and isinstance(self.to_wrap, TERowParallelLinear))
            or (HAVE_TE_COL_LINEAR and isinstance(self.to_wrap, TEColumnParallelLinear))
        )

    def _forward_fp8_te_linear(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """FP8 forward for TE-backed single-weight linears.

        TE linear weights can be tensor subclasses/cached by TE internals, so
        rebinding ``weight.data`` to a dense BF16 tensor is unsafe. Mirror the
        local row/column parallel semantics explicitly, matching the INT4 and
        NVFP4 fallbacks.
        """
        from megatron.bridge.peft.fp8_utils import dequant_fp8

        name = "weight"
        w_fp8 = getattr(self.to_wrap, name).data
        scale_inv = getattr(self.to_wrap, f"{name}_scale_inv", None)
        has_scale_inv = scale_inv is not None
        if scale_inv is None:
            scale_inv = torch.ones(1, device=w_fp8.device, dtype=torch.float32)

        _oft_fp8_debug_log(
            "single_te",
            wrapper=type(self).__name__,
            to_wrap=type(self.to_wrap).__name__,
            weight=w_fp8,
            scale_inv=scale_inv,
            has_scale_inv=has_scale_inv,
            input=x,
            te_return_bias=getattr(self.to_wrap, "te_return_bias", False),
            sequence_parallel=getattr(self.to_wrap, "sequence_parallel", False),
        )

        bias = _get_active_bias_tensor(self.to_wrap)
        has_bias = bias is not None
        tp_group = getattr(self.to_wrap, "_tp_group", None)
        w_compute = None
        w_compute_ptr = None

        def get_w_compute() -> torch.Tensor:
            nonlocal w_compute, w_compute_ptr
            if w_compute is None:
                w_compute = dequant_fp8(w_fp8, scale_inv, out_dtype=x.dtype)
                w_compute_ptr = w_compute.data_ptr()
            return w_compute

        def maybe_apply_activation_qdq(tensor: torch.Tensor) -> torch.Tensor:
            if _get_oft_fp8_activation_quant_mode() == "w8a8":
                return _fp8_activation_qdq_per_token_group_ste(tensor)
            return tensor

        def native_linear_or_none(
            input_tensor: torch.Tensor,
            *,
            native_bias: torch.Tensor | None,
            module_name: str,
        ) -> torch.Tensor | None:
            if not should_attempt_qwen3_native_fp8_gemm():
                return None
            try:
                return maybe_qwen3_native_block_fp8_linear(
                    input_tensor,
                    w_fp8,
                    scale_inv,
                    bias=native_bias,
                    module_name=module_name,
                )
            except ValueError:
                backend = os.environ.get("MEGATRON_QWEN3_FP8_GEMM_BACKEND", "auto").strip().lower()
                if backend == "auto":
                    return None
                raise

        def pack(tensor):
            if w_compute_ptr is not None and tensor.data_ptr() == w_compute_ptr:
                return (
                    w_fp8,
                    scale_inv,
                    tensor.dtype,
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    tensor.storage_offset(),
                )
            return tensor

        def unpack(packed):
            if isinstance(packed, tuple) and len(packed) == 6:
                fp8, sinv, dtype, saved_shape, saved_stride, saved_storage_offset = packed
                base = dequant_fp8(fp8, sinv, out_dtype=dtype)
                return base.as_strided(saved_shape, saved_stride, saved_storage_offset)
            if isinstance(packed, tuple) and len(packed) == 3:
                fp8, sinv, dtype = packed
                return dequant_fp8(fp8, sinv, out_dtype=dtype)
            return packed

        try:
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                if HAVE_TE_ROW_LINEAR and isinstance(self.to_wrap, TERowParallelLinear):
                    output_parallel = native_linear_or_none(
                        x,
                        native_bias=None,
                        module_name=f"{type(self.to_wrap).__name__}.row",
                    )
                    if output_parallel is None:
                        w_compute = get_w_compute()
                        x_qdq = maybe_apply_activation_qdq(x)
                        output_parallel = F.linear(x_qdq.to(w_compute.dtype), w_compute, None)
                    if getattr(self.to_wrap, "explicit_expert_comm", False):
                        output_ = output_parallel
                    elif getattr(self.to_wrap, "sequence_parallel", False):
                        output_ = reduce_scatter_to_sequence_parallel_region(output_parallel, group=tp_group)
                    else:
                        output_ = reduce_from_tensor_model_parallel_region(output_parallel, group=tp_group)

                    if getattr(self.to_wrap, "te_return_bias", False):
                        return output_, bias
                    return output_ + bias if has_bias else output_, None

                if HAVE_TE_COL_LINEAR and isinstance(self.to_wrap, TEColumnParallelLinear):
                    if (
                        getattr(self.to_wrap, "allreduce_dgrad", False)
                        or getattr(self.to_wrap, "sequence_parallel", False)
                        or getattr(self.to_wrap, "explicit_expert_comm", False)
                        or getattr(self.to_wrap, "disable_grad_reduce", False)
                    ):
                        input_parallel = x
                    else:
                        input_parallel = copy_to_tensor_model_parallel_region(x, group=tp_group)

                    native_bias = None if getattr(self.to_wrap, "te_return_bias", False) else (bias if has_bias else None)
                    output_parallel = native_linear_or_none(
                        input_parallel,
                        native_bias=native_bias,
                        module_name=f"{type(self.to_wrap).__name__}.column",
                    )
                    if output_parallel is None:
                        w_compute = get_w_compute()
                        input_qdq = maybe_apply_activation_qdq(input_parallel)
                        output_parallel = F.linear(
                            input_qdq.to(w_compute.dtype),
                            w_compute,
                            native_bias,
                        )

                    if getattr(self.to_wrap, "gather_output", False):
                        output = gather_from_tensor_model_parallel_region(output_parallel, group=tp_group)
                    else:
                        output = output_parallel

                    if getattr(self.to_wrap, "te_return_bias", False):
                        return output, bias
                    return output, None

                native_bias = None if getattr(self.to_wrap, "te_return_bias", False) else (bias if has_bias else None)
                output = native_linear_or_none(
                    x,
                    native_bias=native_bias,
                    module_name=f"{type(self.to_wrap).__name__}.plain",
                )
                if output is None:
                    w_compute = get_w_compute()
                    x_qdq = maybe_apply_activation_qdq(x)
                    output = F.linear(x_qdq.to(w_compute.dtype), w_compute, native_bias)
                if getattr(self.to_wrap, "te_return_bias", False):
                    return output, bias
                return output, None
        finally:
            if w_compute is not None:
                del w_compute

    def _forward_fp8_grouped_eager(
        self, x: torch.Tensor, tokens_per_expert: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Grouped expert FP8 forward with native dispatch and F.linear fallback."""
        from megatron.bridge.peft.fp8_utils import dequant_fp8

        token_splits = self._normalize_tokens_per_expert(tokens_per_expert)
        if len(token_splits) != len(self._weight_names):
            raise ValueError(
                f"Expected {len(self._weight_names)} expert token splits for grouped FP8 path, "
                f"got {len(token_splits)}"
            )
        if getattr(self.to_wrap, "te_return_bias", False):
            raise ValueError("Grouped OFTLinear FP8 fallback does not support te_return_bias=True")

        first_w = getattr(self.to_wrap, self._weight_names[0]).data
        out_features = int(first_w.shape[0])
        output_chunks = [
            x.new_empty((split, out_features), dtype=x.dtype) for split in token_splits
        ]
        active_experts = [idx for idx, split in enumerate(token_splits) if split > 0]
        if not active_experts:
            return x.new_empty((0, out_features), dtype=x.dtype), None

        input_chunks = list(torch.split(x, token_splits, dim=0))
        use_native = should_attempt_qwen3_native_fp8_gemm()
        fallback_chunks = None

        def get_fallback_input_chunks():
            nonlocal fallback_chunks
            if fallback_chunks is None:
                fallback_x = x
                if _get_oft_fp8_activation_quant_mode() == "w8a8":
                    fallback_x = _fp8_activation_qdq_per_token_group_ste(x)
                fallback_chunks = list(torch.split(fallback_x, token_splits, dim=0))
            return fallback_chunks

        ptr_to_handle = {}
        expert_weights = {}

        for idx in active_experts:
            name = self._weight_names[idx]
            w_fp8 = getattr(self.to_wrap, name).data
            scale_inv = getattr(self.to_wrap, f"{name}_scale_inv", None)
            has_scale_inv = scale_inv is not None
            if scale_inv is None:
                scale_inv = torch.ones(1, device=w_fp8.device, dtype=torch.float32)
            _oft_fp8_debug_log(
                "grouped_expert",
                wrapper=type(self).__name__,
                to_wrap=type(self.to_wrap).__name__,
                expert_idx=idx,
                token_count=token_splits[idx],
                weight=w_fp8,
                scale_inv=scale_inv,
                has_scale_inv=has_scale_inv,
                input=input_chunks[idx],
            )
            if use_native:
                try:
                    native_output = maybe_qwen3_native_block_fp8_linear(
                        input_chunks[idx],
                        w_fp8,
                        scale_inv,
                        bias=_get_active_bias_tensor(self.to_wrap, f"bias{idx}"),
                        module_name=f"{type(self.to_wrap).__name__}.expert{idx}",
                    )
                except ValueError:
                    backend = os.environ.get("MEGATRON_QWEN3_FP8_GEMM_BACKEND", "auto").strip().lower()
                    if backend != "auto":
                        raise
                    native_output = None
                if native_output is not None:
                    output_chunks[idx] = native_output
                    continue

            fallback_input = get_fallback_input_chunks()[idx]
            input_chunks[idx] = fallback_input
            w_compute = dequant_fp8(w_fp8, scale_inv, out_dtype=x.dtype)
            ptr_to_handle[w_compute.data_ptr()] = (w_fp8, scale_inv, w_compute.dtype)
            expert_weights[idx] = w_compute

        def pack(tensor):
            handle = ptr_to_handle.get(tensor.data_ptr())
            if handle is not None:
                return (
                    handle[0],
                    handle[1],
                    handle[2],
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    tensor.storage_offset(),
                )
            return tensor

        def unpack(packed):
            if isinstance(packed, tuple) and len(packed) == 6:
                fp8, sinv, dtype, saved_shape, saved_stride, saved_storage_offset = packed
                base = dequant_fp8(fp8, sinv, out_dtype=dtype)
                return base.as_strided(saved_shape, saved_stride, saved_storage_offset)
            if isinstance(packed, tuple) and len(packed) == 3:
                fp8, sinv, dtype = packed
                return dequant_fp8(fp8, sinv, out_dtype=dtype)
            return packed

        try:
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                for idx in active_experts:
                    if idx not in expert_weights:
                        continue
                    bias = _get_active_bias_tensor(self.to_wrap, f"bias{idx}")
                    output_chunks[idx] = F.linear(
                        input_chunks[idx].to(expert_weights[idx].dtype),
                        expert_weights[idx],
                        bias,
                    )
        finally:
            del expert_weights

        return torch.cat(output_chunks, dim=0), None

    def _forward_fp8(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward with FP8 base weight — no BF16 copy held by autograd.

        When ``MEGATRON_OFT_FP8_ACTIVATION_QUANT=w8a8`` is set, the post-OFT
        activation is dynamically FP8-quantized per token per 128-wide group
        before the base linear, matching SGLang's native block-FP8 checkpoint
        path while keeping straight-through gradients for the OFT adapter.

        Uses ``saved_tensors_hooks`` so that when TE's autograd Function calls
        ``ctx.save_for_backward(weight, ...)``, each BF16 weight is replaced by
        a lightweight ``(w_fp8, scale_inv)`` handle.  During backward the hook
        re-dequantises on the fly.  Result: autograd never holds BF16 copies
        of the weights, regardless of gradient checkpointing.

        Handles both standard linears (.weight) and grouped expert linears
        (.weight0, .weight1, ..., .weight{num_gemms-1}).
        """
        from megatron.bridge.peft.fp8_utils import dequant_fp8

        def maybe_apply_activation_qdq(tensor: torch.Tensor) -> torch.Tensor:
            if _get_oft_fp8_activation_quant_mode() == "w8a8":
                return _fp8_activation_qdq_per_token_group_ste(tensor)
            return tensor

        _oft_fp8_debug_log(
            "dispatch",
            wrapper=type(self).__name__,
            to_wrap=type(self.to_wrap).__name__,
            weight_names=len(self._weight_names),
            te_single=self._is_te_single_weight_linear(),
            activation_quant=_get_oft_fp8_activation_quant_mode(),
            input=x,
        )

        if len(self._weight_names) != 1:
            if len(args) == 0:
                raise ValueError("Grouped OFTLinear FP8 fallback requires tokens_per_expert")
            return self._forward_fp8_grouped_eager(x, args[0])

        if self._is_te_single_weight_linear():
            return self._forward_fp8_te_linear(x)

        # Dequant all FP8 weights, track data_ptr → (fp8, scale) for the hook.
        fp8_originals = {}   # name → fp8 tensor (to restore in finally)
        ptr_to_handle = {}   # data_ptr → (fp8, scale_inv, dtype) (for pack hook)

        for name in self._weight_names:
            w_fp8 = getattr(self.to_wrap, name).data
            scale_inv = getattr(self.to_wrap, f"{name}_scale_inv", None)
            if scale_inv is None:
                scale_inv = torch.ones(1, device=w_fp8.device, dtype=torch.float32)

            w_compute = dequant_fp8(w_fp8, scale_inv, out_dtype=x.dtype)
            fp8_originals[name] = w_fp8
            ptr_to_handle[w_compute.data_ptr()] = (w_fp8, scale_inv, w_compute.dtype)
            getattr(self.to_wrap, name).data = w_compute

        def pack(tensor):
            handle = ptr_to_handle.get(tensor.data_ptr())
            if handle is not None:
                return handle
            return tensor

        def unpack(packed):
            if isinstance(packed, tuple) and len(packed) == 3:
                fp8, sinv, dtype = packed
                return dequant_fp8(fp8, sinv, out_dtype=dtype)
            return packed

        try:
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                x_qdq = maybe_apply_activation_qdq(x)
                if not hasattr(self.to_wrap, "config"):
                    return self.to_wrap(x_qdq, *args, **kwargs), None
                linear_output, bias, _ = self.base_linear_forward(x_qdq, *args, **kwargs)
                return linear_output, bias
        finally:
            # Restore all FP8 weights on the module.
            for name, w_fp8 in fp8_originals.items():
                getattr(self.to_wrap, name).data = w_fp8


class OFTVocabParallelEmbedding(AdapterWrapper):
    """Adapter wrapper that applies OFT rotation to the OUTPUT of a vocab-parallel
    embedding lookup.

    Unlike the linear case (rotation on input), the HF PEFT reference (see
    huggingface/peft src/peft/tuners/oft/layer.py — class Embedding) applies the
    rotation AFTER the lookup. The lookup itself is a gather over the vocab
    dimension; the rotation lives on the hidden dimension which is replicated
    across TP ranks, so the rotation parameter is treated like ColumnParallel
    OFT (``input_is_parallel=False``).

    The wrapper preserves the base module's full forward signature — for
    ``VocabParallelEmbedding`` the API is ``forward(input_ids)``, but downstream
    Megatron-Core may add positional / keyword args. Forward everything
    verbatim, then rotate the result on its last dim.

    Initialisation note: ``OFTRotationModule.oft_r`` starts at zero, so
    ``Cayley(0) = I`` and step 0 is bit-identical to the unwrapped embedding.
    """

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        _trap_oft_applied_bridge(self, "VocabParallelEmbedding")
        result = self.to_wrap(*args, **kwargs)
        if not self._adapter_enabled:
            return result
        # The embedding output's last dim is hidden_size (replicated across TP);
        # OFTRotationModule rotates the last dim only. Handles both
        # (b, s, h) and (s_local, b, h) layouts.
        return self.adapter(result.contiguous())


class TEOFTLayerNormLinear(nn.Module):
    """OFT adapter for TE fused LayerNorm+Linear (TELayerNormColumnParallelLinear).

    Inherits from the original fused module but de-fuses the forward pass to insert
    OFT rotation between LayerNorm and Linear:
        1. Run LayerNorm using TE's normalization with existing LN weights
        2. Apply OFT rotation R to the LN output
        3. Run the Linear part

    This preserves checkpoint FQNs (weight, bias, layer_norm_weight stay at
    the same keys) and avoids the 2x compute cost of the decomposition approach.
    The only cost is losing kernel fusion between LN and Linear.

    Args:
        orig_module: The original TELayerNormColumnParallelLinear to adapt.
        adapter: OFTRotationModule instance.
    """

    def __init__(self, orig_module: nn.Module, adapter: OFTRotationModule):
        super().__init__()
        # Store reference to original module and adapter
        self._orig_module = orig_module
        _clear_disabled_bias_parameters(self._orig_module)
        self.adapter = adapter
        self._adapter_enabled = True

    def enable_adapter_layers(self) -> None:
        self._adapter_enabled = True

    def disable_adapter_layers(self) -> None:
        self._adapter_enabled = False

    def _run_layernorm(self, x: torch.Tensor) -> torch.Tensor:
        """Run just the LayerNorm part using the original module's LN weights."""

        normalization = getattr(self._orig_module, "normalization", "LayerNorm")
        eps = self._orig_module.eps
        ln_weight = self._orig_module.layer_norm_weight
        ln_bias = getattr(self._orig_module, "layer_norm_bias", None)
        zero_centered_gamma = getattr(self._orig_module, "zero_centered_gamma", False)

        if normalization == "RMSNorm":
            # TE RMSNorm: y = x * rsqrt(mean(x^2) + eps) * (1 + gamma) or * gamma
            if zero_centered_gamma:
                ln_out = F.rms_norm(x, (x.shape[-1],), ln_weight + 1.0, eps)
            else:
                ln_out = F.rms_norm(x, (x.shape[-1],), ln_weight, eps)
        else:
            # Standard LayerNorm
            if zero_centered_gamma:
                ln_out = F.layer_norm(x, (x.shape[-1],), ln_weight + 1.0, ln_bias, eps)
            else:
                ln_out = F.layer_norm(x, (x.shape[-1],), ln_weight, ln_bias, eps)

        return ln_out

    def _run_linear(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run just the Linear part using the original module's weight/bias.

        For column-parallel with sequence parallelism, the fused TE kernel
        internally does an all-gather on the input before GEMM and a
        reduce-scatter on the output gradient. Since we de-fused the kernel,
        we must replicate this communication pattern explicitly.
        """
        weight = self._orig_module.weight
        bias = _get_active_bias_tensor(self._orig_module)
        has_bias = bias is not None

        # Handle sequence parallelism: all-gather input along seq dimension
        # For ColumnParallel + SP, input is [seq/tp, batch, hidden] and needs
        # to be gathered to [seq, batch, hidden] before GEMM.
        if getattr(self._orig_module, "sequence_parallel", False):
            from megatron.core.tensor_parallel.mappings import (
                gather_from_sequence_parallel_region,
            )

            x = gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=True)

        x = x.to(weight.dtype)

        te_return_bias = getattr(self._orig_module, "te_return_bias", False)
        if te_return_bias and has_bias:
            out = F.linear(x, weight, None)
            return out, bias
        out = F.linear(x, weight, bias if has_bias else None)
        return out, None

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        """De-fused forward: LN -> OFT rotation -> Linear."""
        _trap_oft_applied_bridge(self, "LayerNormLinear")
        if not self._adapter_enabled:
            # Fall back to original fused forward
            return self._orig_module(x, *args, **kwargs)

        # Step 1: LayerNorm
        ln_out = self._run_layernorm(x)

        # Step 2: OFT rotation
        rotated = self.adapter(ln_out.contiguous())

        # Step 3: Linear
        return self._run_linear(rotated)

    # --- Delegate attribute access to original module for checkpoint compatibility ---

    def __getattr__(self, name: str):
        """Delegate attribute access to the original module for everything
        except our own attributes (adapter, _adapter_enabled, _orig_module)."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._orig_module, name)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Return state dict with original module's keys plus adapter keys."""
        if destination is None:
            destination = {}
        self._orig_module.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        self.adapter.state_dict(destination=destination, prefix=f"{prefix}adapter.", keep_vars=keep_vars)
        return destination

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Combine original module's sharded state dict with adapter's."""
        sharded_state_dict = {}
        sharded_state_dict.update(self._orig_module.sharded_state_dict(prefix, sharded_offsets, metadata))
        sharded_state_dict.update(self.adapter.sharded_state_dict(f"{prefix}adapter.", sharded_offsets, metadata))
        return sharded_state_dict
