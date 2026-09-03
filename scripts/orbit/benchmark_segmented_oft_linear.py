# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Benchmark routed BF16 OFT projection against the eager segmented baseline."""

from __future__ import annotations

import json
import logging
import platform
import sys

import torch
import triton

from megatron.bridge.orbit.oft.triton_oft.segmented_oft_linear import (
    segmented_oft_linear,
)


LAYOUTS = {
    "rank0": ([0, 4096, 4352], [0, 1]),
    "rank1": ([0, 3840, 4096, 4352], [1, 2, 3]),
}
TOKEN_COUNTS = [1, 64, 256, 1024]
logger = logging.getLogger(__name__)


def _eager(
    x: torch.Tensor,
    weight: torch.Tensor,
    rotations: torch.Tensor,
    offsets: list[int],
    rotation_ids: list[int],
) -> torch.Tensor:
    leading_shape = x.shape[:-1]
    x_2d = x.reshape(-1, x.shape[-1])
    x_blocks = x_2d.reshape(x_2d.shape[0], rotations.shape[1], rotations.shape[-1])
    outputs = []
    for start, end, rotation_id in zip(offsets, offsets[1:], rotation_ids):
        rotated = torch.einsum("mbi,bij->mbj", x_blocks, rotations[rotation_id]).reshape_as(x_2d)
        outputs.append(torch.nn.functional.linear(rotated, weight[start:end]))
    return torch.cat(outputs, dim=-1).reshape(*leading_shape, weight.shape[0])


def _measure_memory(callable_) -> int:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    callable_()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - baseline


def _benchmark_case(layout: str, tokens: int) -> list[dict[str, object]]:
    device = torch.device("cuda")
    input_dim = 4096
    output_dim = 4352
    block_size = 32
    offsets, rotation_ids = LAYOUTS[layout]
    offsets_tensor = torch.tensor(offsets, device=device, dtype=torch.int32)
    ids_tensor = torch.tensor(rotation_ids, device=device, dtype=torch.int32)
    torch.manual_seed(2026 + tokens)
    x = torch.randn(tokens, input_dim, device=device, dtype=torch.bfloat16)
    weight = (
        torch.randn(output_dim, input_dim, device=device, dtype=torch.bfloat16)
        / input_dim**0.5
    )
    skew = torch.randn(
        4,
        input_dim // block_size,
        block_size,
        block_size,
        device=device,
        dtype=torch.float32,
    ) * 0.01
    skew = skew - skew.transpose(-1, -2)
    identity = torch.eye(block_size, device=device).expand_as(skew)
    rotations = torch.linalg.solve(identity + skew, identity - skew).to(torch.bfloat16)

    eager_out = _eager(x, weight, rotations, offsets, rotation_ids)
    fused_out = segmented_oft_linear(x, weight, rotations, offsets_tensor, ids_tensor)
    max_diff = (eager_out.float() - fused_out.float()).abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(
        eager_out.float().flatten(), fused_out.float().flatten(), dim=0
    ).item()
    try:
        torch.testing.assert_close(fused_out.float(), eager_out.float(), rtol=1e-2, atol=1e-2)
    except AssertionError as error:
        raise AssertionError(f"{layout=} {tokens=}: max_diff={max_diff}, cosine={cosine}") from error
    if cosine <= 0.9999:
        raise AssertionError(f"{layout=} {tokens=}: max_diff={max_diff}, cosine={cosine}")

    rows: list[dict[str, object]] = []
    for mode in ("forward", "forward_backward"):
        x_eager = x.detach().clone().requires_grad_(mode == "forward_backward")
        r_eager = rotations.detach().clone().requires_grad_(mode == "forward_backward")
        x_fused = x.detach().clone().requires_grad_(mode == "forward_backward")
        r_fused = rotations.detach().clone().requires_grad_(mode == "forward_backward")
        upstream = torch.randn(tokens, output_dim, device=device, dtype=torch.bfloat16)

        def eager_call() -> None:
            x_eager.grad = None
            r_eager.grad = None
            out = _eager(x_eager, weight, r_eager, offsets, rotation_ids)
            if mode == "forward_backward":
                out.backward(upstream)

        def fused_call() -> None:
            x_fused.grad = None
            r_fused.grad = None
            out = segmented_oft_linear(x_fused, weight, r_fused, offsets_tensor, ids_tensor)
            if mode == "forward_backward":
                out.backward(upstream)

        eager_ms = triton.testing.do_bench(eager_call, warmup=25, rep=100)
        fused_ms = triton.testing.do_bench(fused_call, warmup=25, rep=100)
        rows.append(
            {
                "layout": layout,
                "tokens": tokens,
                "mode": mode,
                "eager_ms": eager_ms,
                "fused_ms": fused_ms,
                "speedup": eager_ms / fused_ms,
                "eager_peak_bytes": _measure_memory(eager_call),
                "fused_peak_bytes": _measure_memory(fused_call),
                "max_abs_diff": max_diff,
                "cosine": cosine,
            }
        )
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    metadata = {
        "device": torch.cuda.get_device_name(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
    }
    logger.info(json.dumps({"environment": metadata}, sort_keys=True))
    rows = [
        row
        for layout in LAYOUTS
        for tokens in TOKEN_COUNTS
        for row in _benchmark_case(layout, tokens)
    ]
    for row in rows:
        logger.info(
            f"{row['layout']:5s} M={row['tokens']:4d} {row['mode']:16s} "
            f"eager={row['eager_ms']:.4f}ms fused={row['fused_ms']:.4f}ms "
            f"speedup={row['speedup']:.3f}x"
        )
    logger.info(json.dumps({"results": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
