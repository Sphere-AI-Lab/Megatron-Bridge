#!/usr/bin/env python3
"""Dump the sharded_state_dict keys of a modelopt-NVFP4-wrapped Megatron meta model.

This is a diagnostic helper: we want to know what keys modelopt expects for the
per-expert weight_quantizer / input_quantizer state on a grouped MoE linear, so
the direct-save converter can write the scale tensors under the right names.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

from megatron.core import dist_checkpointing as _dc

_orig_dc_save = _dc.save


def _dc_save_with_mcore(*args, **kwargs):
    kwargs.setdefault("async_strategy", "mcore")
    return _orig_dc_save(*args, **kwargs)


_dc.save = _dc_save_with_mcore

from megatron.bridge.sphere.low_precision.common import (
    build_single_rank_meta_provider,
    patch_meta_init_for_te_modules,
)
from megatron.bridge.sphere.low_precision.nvfp4 import (
    apply_modelopt_nvfp4_to_meta_model,
    collect_nvfp4_target_module_names,
    is_nvfp4_source,
)
from megatron.bridge.training.model_load_save import temporary_distributed_context
from megatron.bridge.training.utils.pg_utils import get_pg_collection


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model-path", required=False, default=None)
    p.add_argument(
        "--probe-compress",
        action="store_true",
        help="Standalone probe: build a tiny CPU Linear, apply mtq.quantize + "
             "mtq.compress with NVFP4_DEFAULT_CFG, and dump the post-compress "
             "quantizer buffers. Does not require --hf-model-path.",
    )
    p.add_argument(
        "--probe-roundtrip",
        action="store_true",
        help="Standalone probe: verify that manually registering the "
             "_amax/_scale/_double_scale buffers on a quantized (no-compress) "
             "Linear produces the same state_dict and forward output as "
             "mtq.compress. Does not require --hf-model-path.",
    )
    p.add_argument(
        "--layer-match",
        default=r"decoder\.layers\.1\.",
        help="Regex of which Megatron keys to dump (default: layer 1).",
    )
    p.add_argument(
        "--max-keys",
        type=int,
        default=0,
        help="Cap number of matching keys dumped (0 = no cap).",
    )
    p.add_argument(
        "--exclude",
        default=None,
        help="Optional regex: keys matching this are skipped.",
    )
    p.add_argument(
        "--include",
        default=None,
        help="Optional regex: only keys matching this are dumped.",
    )
    p.add_argument(
        "--module",
        default=None,
        help="Optional substring: only show keys whose module prefix contains it.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Write the full dump to this file (default: /tmp/nvfp4_meta_keys_<model>_<tag>.log).",
    )
    p.add_argument(
        "--repr-factories",
        action="store_true",
        help="Also print repr() of ShardedTensorFactory entries to inspect their fields.",
    )
    p.add_argument(
        "--dump-quantizers",
        action="store_true",
        help="Also walk named_modules() and dump weight_quantizer/input_quantizer buffers for "
             "the first module matching --layer-match (filtered by --module if provided).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.probe_compress:
        return _probe_compress_standalone()

    if args.probe_roundtrip:
        return _probe_roundtrip_standalone()

    if args.hf_model_path is None:
        raise SystemExit(
            "--hf-model-path is required unless --probe-compress or --probe-roundtrip is set"
        )

    auto_bridge, provider = build_single_rank_meta_provider(
        args.hf_model_path, trust_remote_code=True,
    )
    if not is_nvfp4_source(auto_bridge.hf_pretrained.config):
        raise ValueError("Source is not an NVFP4 HuggingFace checkpoint")

    if hasattr(provider, "finalize"):
        provider.finalize()

    patch_meta_init_for_te_modules()

    with temporary_distributed_context(backend="gloo"):
        megatron_model = provider.provide_distributed_model(
            wrap_with_ddp=False,
            use_cpu_initialization=True,
            init_model_with_meta_device=True,
            mixed_precision_wrapper=None,
        )

        conversion_tasks = auto_bridge._model_bridge.build_conversion_tasks(
            auto_bridge.hf_pretrained, megatron_model,
        )
        module_names = collect_nvfp4_target_module_names(
            conversion_tasks, auto_bridge.hf_pretrained.state, show_progress=False,
        )
        apply_modelopt_nvfp4_to_meta_model(megatron_model[0], module_names=module_names)

        pg_collection = get_pg_collection(megatron_model)
        ssd = megatron_model[0].sharded_state_dict(
            metadata={"dp_cp_group": pg_collection.dp_cp},
        )

        layer_re = re.compile(args.layer_match)
        exclude_re = re.compile(args.exclude) if args.exclude else None
        include_re = re.compile(args.include) if args.include else None

        def _keep(key: str) -> bool:
            if not layer_re.search(key):
                return False
            if exclude_re is not None and exclude_re.search(key):
                return False
            if include_re is not None and not include_re.search(key):
                return False
            if args.module is not None and args.module not in key:
                return False
            return True

        matches = sorted(k for k in ssd.keys() if _keep(k))

        grouped = defaultdict(list)
        for k in matches:
            module = k.rsplit(".", 1)[0]
            grouped[module].append(k)

        output_path = args.output
        if output_path is None:
            model_tag = Path(args.hf_model_path).name or "model"
            filter_tag = re.sub(r"[^a-zA-Z0-9]+", "_", args.layer_match).strip("_") or "all"
            repo_root = Path(__file__).resolve().parents[2]
            output_path = str(repo_root / f"nvfp4_meta_keys_{model_tag}_{filter_tag}.log")
        output_path = str(Path(output_path).resolve())
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        header_lines = [
            f"HF model:                  {args.hf_model_path}",
            f"Layer match regex:         {args.layer_match}",
            f"Include regex:             {args.include}",
            f"Exclude regex:             {args.exclude}",
            f"Module substring:          {args.module}",
            f"Total sharded_state_dict keys: {len(ssd)}",
            f"Matching (after filters):   {len(matches)}",
            "=" * 70,
        ]

        log = open(output_path, "w")

        def _emit(line: str) -> None:
            print(line)
            log.write(line + "\n")

        for line in header_lines:
            _emit(line)

        count = 0
        cap = args.max_keys if args.max_keys and args.max_keys > 0 else None
        for module, keys in grouped.items():
            _emit(f"\n[module prefix] {module}")
            for k in keys:
                entry = ssd[k]
                dtype = getattr(entry, "dtype", None)
                shape = (
                    getattr(entry, "local_shape", None)
                    or getattr(entry, "global_shape", None)
                    or getattr(entry, "shape", None)
                )
                _emit(
                    f"    {k}   type={type(entry).__name__}  dtype={dtype}  shape={shape}"
                )
                if args.repr_factories:
                    _emit(f"      repr: {entry!r}")
                count += 1
                if cap is not None and count >= cap:
                    _emit(f"\n(truncated after {cap} keys)")
                    log.flush()
                    log.close()
                    print(f"\nFull dump written to: {output_path}")
                    return 0

        if args.dump_quantizers:
            _emit("\n" + "=" * 70)
            _emit("Quantizer module buffer probe")
            _emit("=" * 70)
            _dump_quantizer_modules(
                megatron_model[0],
                layer_re=layer_re,
                module_substring=args.module,
                emit=_emit,
            )

        log.flush()
        log.close()
        print(f"\nFull dump written to: {output_path}")

    return 0


def _describe_tensor(tensor) -> str:
    """Compact one-line description of a buffer/parameter tensor."""
    if tensor is None:
        return "None"
    try:
        device = tensor.device
    except AttributeError:
        return f"<non-tensor {type(tensor).__name__}>"
    if device.type == "meta":
        return f"meta dtype={tensor.dtype} shape={tuple(tensor.shape)}"
    numel = tensor.numel()
    try:
        scalar = tensor.item() if numel == 1 else None
    except Exception:
        scalar = None
    scalar_str = f" value={scalar:.6g}" if scalar is not None else ""
    return (
        f"device={device} dtype={tensor.dtype} shape={tuple(tensor.shape)}"
        f" numel={numel}{scalar_str}"
    )


def _dump_quantizer_modules(
    model_chunk,
    *,
    layer_re,
    module_substring: str | None,
    emit,
) -> None:
    """Walk ``model_chunk.named_modules()`` and dump the first few quantizer modules.

    For each parent module matching the layer filter that owns a
    ``weight_quantizer`` / ``input_quantizer`` / ``output_quantizer`` submodule,
    print the submodule's parameters and buffers with shapes/dtypes/values.
    """
    quantizer_attrs = ("weight_quantizer", "input_quantizer", "output_quantizer")
    dumped = 0
    for module_name, module in model_chunk.named_modules():
        if not layer_re.search(module_name):
            continue
        if module_substring is not None and module_substring not in module_name:
            continue
        if not any(hasattr(module, attr) for attr in quantizer_attrs):
            continue

        emit(f"\n[parent module] {module_name}  ({type(module).__name__})")
        for attr in quantizer_attrs:
            q = getattr(module, attr, None)
            if q is None:
                continue
            emit(f"  [{attr}]  ({type(q).__name__})")
            try:
                emit(f"    repr: {q!r}")
            except Exception as exc:  # noqa: BLE001
                emit(f"    repr: <failed: {exc}>")

            params = list(q.named_parameters(recurse=False))
            buffers = list(q.named_buffers(recurse=False))
            emit(f"    parameters ({len(params)}): {[n for n, _ in params] or '—'}")
            for pname, param in params:
                emit(f"      param  {pname:<24} {_describe_tensor(param)}")
            emit(f"    buffers    ({len(buffers)}): {[n for n, _ in buffers] or '—'}")
            for bname, buf in buffers:
                emit(f"      buffer {bname:<24} {_describe_tensor(buf)}")

            # Dump every attribute on the quantizer so we can see what modelopt
            # has pre-registered (even None-valued placeholders for _amax,
            # _scale, _bias, _pre_quant_scale, etc.).
            state = vars(q)
            keys = sorted(state.keys())
            emit(f"    __dict__ keys ({len(keys)}):")
            for k in keys:
                v = state[k]
                if torch.is_tensor(v):
                    summary = _describe_tensor(v)
                elif v is None:
                    summary = "None"
                elif isinstance(v, (int, float, bool, str)):
                    summary = f"{type(v).__name__}={v!r}"
                elif isinstance(v, (list, tuple, dict, set)):
                    summary = f"{type(v).__name__} len={len(v)}  value={v!r}"[:140]
                else:
                    summary = f"<{type(v).__name__}>"
                emit(f"      attr   {k:<24} {summary}")

        dumped += 1
        if dumped >= 3:
            emit("\n(stopped after 3 parent modules — use --module to target others)")
            break


def _probe_compress_standalone() -> int:
    """Build a tiny CPU Linear, quantize + compress with NVFP4, dump buffers.

    Goal: find out what buffers modelopt's ``TensorQuantizer`` registers once
    ``mtq.compress`` is actually run on a real (non-meta) weight. The results
    tell the converter which buffer names / shapes it must populate from the
    HF NVFP4 bundle so that ``save_sharded_modelopt_state`` can serialize them.
    """
    import modelopt.torch.quantization as mtq

    repo_root = Path(__file__).resolve().parents[2]
    output_path = str(repo_root / "nvfp4_probe_compress.log")
    log = open(output_path, "w")

    def emit(line: str) -> None:
        print(line)
        log.write(line + "\n")

    class TinyBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(128, 64, bias=False)

        def forward(self, x):
            return self.linear(x)

    torch.manual_seed(0)
    model = TinyBlock().to(torch.float32)
    calib_x = torch.randn(4, 128)

    def _forward_loop(m):
        with torch.no_grad():
            m(calib_x)

    emit("=" * 70)
    emit("Stage 1: mtq.quantize (no compress) — inspect registered state")
    emit("=" * 70)

    mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, _forward_loop)
    _report_tensorquantizer(model.linear, emit, label="after quantize")

    emit("\n" + "=" * 70)
    emit("Stage 2: mtq.compress — inspect registered state after compression")
    emit("=" * 70)

    try:
        mtq.compress(model)
    except Exception as exc:  # noqa: BLE001
        emit(f"mtq.compress raised: {type(exc).__name__}: {exc}")
        log.flush()
        log.close()
        return 0

    _report_tensorquantizer(model.linear, emit, label="after compress")

    emit("\n" + "=" * 70)
    emit("module state_dict() keys after compress")
    emit("=" * 70)
    sd = model.state_dict()
    for k, v in sd.items():
        emit(f"  {k:<50} {_describe_tensor(v)}")

    emit(f"\n(written to {output_path})")
    log.flush()
    log.close()
    print(f"\nFull probe written to: {output_path}")
    return 0


def _probe_roundtrip_standalone() -> int:
    """Compare mtq.compress vs manually registered buffers on a dense Linear.

    Verifies that writing the HF NVFP4 bundle fields directly onto the
    TensorQuantizer as ``register_buffer`` calls produces the same state_dict
    and forward output as running ``mtq.compress``. If so, the direct-save
    converter can skip mtq.compress (which doesn't work cleanly on meta
    tensors) and populate buffers from HF data instead.
    """
    import modelopt.torch.quantization as mtq

    repo_root = Path(__file__).resolve().parents[2]
    output_path = str(repo_root / "nvfp4_probe_roundtrip.log")
    log = open(output_path, "w")

    def emit(line: str) -> None:
        print(line)
        log.write(line + "\n")

    torch.manual_seed(0)
    ref_weight = torch.randn(64, 128, dtype=torch.float32)
    calib_x = torch.randn(4, 128, dtype=torch.float32)

    def _build_linear_with_weight(w: torch.Tensor) -> torch.nn.Module:
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(128, 64, bias=False)
                with torch.no_grad():
                    self.linear.weight.copy_(w)

            def forward(self, x):
                return self.linear(x)

        return Block()

    def _forward_loop(m):
        with torch.no_grad():
            m(calib_x)

    emit("=" * 70)
    emit("Path A: mtq.quantize + mtq.compress  (ground truth)")
    emit("=" * 70)
    model_a = _build_linear_with_weight(ref_weight)
    mtq.quantize(model_a, mtq.NVFP4_DEFAULT_CFG, _forward_loop)
    mtq.compress(model_a)
    sd_a = model_a.state_dict()
    for k, v in sd_a.items():
        emit(f"  A  {k:<50} {_describe_tensor(v)}")

    emit("\n" + "=" * 70)
    emit("Path B: mtq.quantize + manual register_buffer (no mtq.compress)")
    emit("=" * 70)
    model_b = _build_linear_with_weight(ref_weight)
    mtq.quantize(model_b, mtq.NVFP4_DEFAULT_CFG, _forward_loop)

    # Copy ground-truth buffer values from path A onto path B.
    linear_b = model_b.linear
    packed_weight = sd_a["linear.weight"]
    wq_amax = sd_a["linear.weight_quantizer._amax"]
    wq_scale = sd_a["linear.weight_quantizer._scale"]
    wq_double = sd_a["linear.weight_quantizer._double_scale"]
    iq_amax = sd_a["linear.input_quantizer._amax"]

    # Replace linear.weight with the packed uint8 tensor (shape + dtype must
    # match what mtq.compress produced).
    linear_b.weight = torch.nn.Parameter(packed_weight.clone(), requires_grad=False)

    # Register buffers on the quantizer modules.
    linear_b.weight_quantizer.register_buffer("_amax", wq_amax.clone())
    linear_b.weight_quantizer.register_buffer("_scale", wq_scale.clone())
    linear_b.weight_quantizer.register_buffer("_double_scale", wq_double.clone())
    linear_b.input_quantizer.register_buffer("_amax", iq_amax.clone())

    sd_b = model_b.state_dict()
    for k, v in sd_b.items():
        emit(f"  B  {k:<50} {_describe_tensor(v)}")

    emit("\n" + "=" * 70)
    emit("Comparison")
    emit("=" * 70)
    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())
    emit(f"  keys-only-in-A: {sorted(keys_a - keys_b) or '—'}")
    emit(f"  keys-only-in-B: {sorted(keys_b - keys_a) or '—'}")
    common = sorted(keys_a & keys_b)
    for k in common:
        va, vb = sd_a[k], sd_b[k]
        same_meta = va.dtype == vb.dtype and tuple(va.shape) == tuple(vb.shape)
        if not same_meta:
            emit(f"  mismatch {k}: A={_describe_tensor(va)}  B={_describe_tensor(vb)}")
            continue
        try:
            allclose = torch.equal(va, vb)
        except Exception:
            allclose = torch.allclose(va.float(), vb.float(), atol=0, rtol=0)
        emit(f"  {'OK ' if allclose else 'DIFF'} {k:<50} dtype={va.dtype} shape={tuple(va.shape)}")

    emit("\n(forward-pass comparison skipped: modelopt NVFP4 dynamic_block_quant "
         "is CUDA-only; state_dict match is the equivalence test we can run on CPU)")

    emit(f"\n(written to {output_path})")
    log.flush()
    log.close()
    print(f"\nRound-trip probe written to: {output_path}")
    return 0


def _report_tensorquantizer(parent_module, emit, *, label: str) -> None:
    """Dump weight/input/output quantizer state on a single parent module."""
    for attr in ("weight_quantizer", "input_quantizer", "output_quantizer"):
        q = getattr(parent_module, attr, None)
        if q is None:
            continue
        emit(f"\n[{attr}] {label}  ({type(q).__name__})")
        try:
            emit(f"  repr: {q!r}")
        except Exception as exc:  # noqa: BLE001
            emit(f"  repr: <failed: {exc}>")
        params = list(q.named_parameters(recurse=False))
        buffers = list(q.named_buffers(recurse=False))
        emit(f"  parameters ({len(params)}): {[n for n, _ in params] or '—'}")
        for pname, param in params:
            emit(f"    param  {pname:<24} {_describe_tensor(param)}")
        emit(f"  buffers    ({len(buffers)}): {[n for n, _ in buffers] or '—'}")
        for bname, buf in buffers:
            emit(f"    buffer {bname:<24} {_describe_tensor(buf)}")
        sd = q.state_dict(keep_vars=True)
        emit(f"  state_dict ({len(sd)}): {list(sd.keys()) or '—'}")
        for k, v in sd.items():
            emit(f"    sd     {k:<24} {_describe_tensor(v)}")

    # Also show the parent linear's weight after compress (dtype/shape changes).
    if hasattr(parent_module, "weight"):
        emit(f"\n[{type(parent_module).__name__}.weight] {_describe_tensor(parent_module.weight)}")


if __name__ == "__main__":
    raise SystemExit(main())
