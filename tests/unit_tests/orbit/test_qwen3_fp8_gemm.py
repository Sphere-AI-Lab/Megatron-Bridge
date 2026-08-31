# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import pytest
import torch

from megatron.bridge.orbit.quant import qwen3_fp8_gemm


pytestmark = pytest.mark.unit


def _valid_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_tensor = torch.ones((2, 128), dtype=torch.bfloat16)
    weight = torch.ones((1, 128), dtype=torch.float8_e4m3fn)
    scale_inv = torch.ones((1, 1), dtype=torch.float32)
    return input_tensor, weight, scale_inv


def test_auto_backend_propagates_native_kernel_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto fallback must not hide a backend that imported but failed while executing."""

    def failing_native_linear(**_kwargs) -> torch.Tensor:
        raise RuntimeError("native kernel execution failed")

    monkeypatch.setenv("MEGATRON_QWEN3_FP8_GEMM_BACKEND", "auto")
    monkeypatch.setattr(qwen3_fp8_gemm, "_load_sglang_native_linear", lambda: failing_native_linear)
    input_tensor, weight, scale_inv = _valid_inputs()

    with pytest.raises(RuntimeError, match="native kernel execution failed"):
        qwen3_fp8_gemm.maybe_qwen3_native_block_fp8_linear(input_tensor, weight, scale_inv)


def test_auto_backend_falls_back_when_sglang_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEGATRON_QWEN3_FP8_GEMM_BACKEND", "auto")
    monkeypatch.setattr(qwen3_fp8_gemm, "_load_sglang_native_linear", lambda: None)
    input_tensor, weight, scale_inv = _valid_inputs()

    result = qwen3_fp8_gemm.maybe_qwen3_native_block_fp8_linear(input_tensor, weight, scale_inv)

    assert result is None


def test_explicit_native_backend_rejects_missing_sglang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEGATRON_QWEN3_FP8_GEMM_BACKEND", "sglang_native")
    monkeypatch.setattr(qwen3_fp8_gemm, "_load_sglang_native_linear", lambda: None)
    input_tensor, weight, scale_inv = _valid_inputs()

    with pytest.raises(RuntimeError, match="sglang_native requested"):
        qwen3_fp8_gemm.maybe_qwen3_native_block_fp8_linear(input_tensor, weight, scale_inv)
