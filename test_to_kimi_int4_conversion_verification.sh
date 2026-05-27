#!/bin/bash
# Verify a BF16 HF -> Kimi-style INT4 HF conversion.
#
# Checks:
#   - expert MLP weights were replaced by weight_packed/weight_scale/weight_shape
#   - non-expert tensors were preserved
#   - sampled INT4 tensors dequantize back close to the original BF16 values
#
# Environment overrides:
#   REPO            - Megatron-Bridge repo path
#   IN              - original BF16 HF checkpoint
#   OUT             - converted INT4 HF checkpoint
#   GROUP_SIZE      - quantization group size (default: 32)
#   MAX_QUANT_KEYS  - sample this many expert weights; use "all" for full pass

set -euo pipefail

# CUDA setup
if command -v module >/dev/null 2>&1; then
    module load "${CUDA_MODULE:-cuda/13.2}" || true
fi
if command -v module >/dev/null 2>&1; then
    module load "${NCCL_MODULE:-nccl}" || true
fi
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH

# Library paths
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -n "${CUDNN_HOME:-}" ]]; then
    export LD_LIBRARY_PATH="${CUDNN_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

REPO="${REPO:-${MEGATRON_BRIDGE_ROOT:-${PWD}}}"
IN="${IN:-${HF_MODEL_ROOT:-${HOME}/hf_models}/Moonlight-16B-A3B}"
OUT="${OUT:-${HF_MODEL_ROOT:-${HOME}/hf_models}/Moonlight-16B-A3B-INT4}"
GROUP_SIZE="${GROUP_SIZE:-32}"
MAX_QUANT_KEYS="${MAX_QUANT_KEYS:-8}"

export REPO IN OUT GROUP_SIZE MAX_QUANT_KEYS

python - <<'PY'
from pathlib import Path
import importlib.util
import json
import math
import os
import sys
from safetensors import safe_open
import torch

REPO = Path(os.environ["REPO"])
IN = Path(os.environ["IN"])
OUT = Path(os.environ["OUT"])
GROUP_SIZE = int(os.environ["GROUP_SIZE"])
MAX_QUANT_KEYS_RAW = os.environ["MAX_QUANT_KEYS"].strip().lower()


def parse_limit(raw: str):
    if raw in {"", "all", "none", "null"}:
        return None
    return int(raw)


MAX_QUANT_KEYS = parse_limit(MAX_QUANT_KEYS_RAW)

spec = importlib.util.spec_from_file_location(
    "kimi_utils",
    REPO / "src/megatron/bridge/models/kimi_vl/utils.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def should_quantize(key: str) -> bool:
    return (
        key.endswith(".weight")
        and "experts." in key
        and "shared_expert" not in key
        and any(p in key for p in ("gate_proj", "up_proj", "down_proj"))
    )


def load_index(root: Path) -> dict:
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())

    shard_files = sorted(root.glob("*.safetensors"))
    if len(shard_files) != 1:
        raise FileNotFoundError(
            f"Could not find model.safetensors.index.json under {root}, "
            f"and found {len(shard_files)} standalone .safetensors files"
        )

    shard_name = shard_files[0].name
    with safe_open(str(shard_files[0]), framework="pt") as f:
        weight_map = {key: shard_name for key in f.keys()}
    return {"metadata": {}, "weight_map": weight_map}


def get_tensor(root: Path, index: dict, key: str):
    shard = index["weight_map"][key]
    with safe_open(str(root / shard), framework="pt") as f:
        return f.get_tensor(key)

if not IN.exists():
    raise FileNotFoundError(f"Input checkpoint path not found: {IN}")
if not OUT.exists():
    raise FileNotFoundError(f"Output checkpoint path not found: {OUT}")

orig_index = load_index(IN)
out_index = load_index(OUT)

missing = []
bad = []
maes = []
max_abses = []
quant_checked = 0
kept_checked = 0
quant_total = 0

for key in orig_index["weight_map"]:
    if should_quantize(key):
        quant_total += 1
        if MAX_QUANT_KEYS is not None and quant_checked >= MAX_QUANT_KEYS:
            continue

        base = key[:-len(".weight")]
        packed_key = f"{base}.weight_packed"
        scale_key = f"{base}.weight_scale"
        shape_key = f"{base}.weight_shape"

        for new_key in (packed_key, scale_key, shape_key):
            if new_key not in out_index["weight_map"]:
                missing.append(new_key)

        if key in out_index["weight_map"]:
            bad.append(f"dense expert weight was not removed: {key}")
            continue
        if any(k not in out_index["weight_map"] for k in (packed_key, scale_key, shape_key)):
            continue

        orig = get_tensor(IN, orig_index, key)
        packed = get_tensor(OUT, out_index, packed_key)
        scale = get_tensor(OUT, out_index, scale_key)
        shape = get_tensor(OUT, out_index, shape_key)

        if packed.dtype != torch.int32:
            bad.append(f"{packed_key}: dtype {packed.dtype}, expected torch.int32")
        if scale.dtype != torch.float16:
            bad.append(f"{scale_key}: dtype {scale.dtype}, expected torch.float16")
        if shape.dtype != torch.int64:
            bad.append(f"{shape_key}: dtype {shape.dtype}, expected torch.int64")

        if shape.tolist() != list(orig.shape):
            bad.append(f"{shape_key}: stores {shape.tolist()}, expected {list(orig.shape)}")

        expected_packed_shape = [orig.shape[0], orig.shape[1] // 8]
        expected_scale_shape = [orig.shape[0], math.ceil(orig.shape[1] / GROUP_SIZE)]

        if list(packed.shape) != expected_packed_shape:
            bad.append(f"{packed_key}: shape {list(packed.shape)}, expected {expected_packed_shape}")
        if list(scale.shape) != expected_scale_shape:
            bad.append(f"{scale_key}: shape {list(scale.shape)}, expected {expected_scale_shape}")

        recon = mod.dequantize_int4(packed, scale, shape, group_size=GROUP_SIZE)
        diff = (recon.float() - orig.float()).abs()
        maes.append(float(diff.mean()))
        max_abses.append(float(diff.max()))
        quant_checked += 1
    else:
        if key not in out_index["weight_map"]:
            missing.append(key)
            continue
        orig = get_tensor(IN, orig_index, key)
        new = get_tensor(OUT, out_index, key)
        if not torch.equal(orig, new):
            bad.append(f"non-quantized tensor changed: {key}")
        kept_checked += 1

if quant_total == 0:
    bad.append("found zero quantizable expert weights in the input checkpoint")

print({
    "input_quantizable_expert_weights": quant_total,
    "quantized_keys_checked": quant_checked,
    "kept_keys_checked": kept_checked,
    "sampled_quantized_check": MAX_QUANT_KEYS is not None,
    "max_quant_keys": MAX_QUANT_KEYS,
    "missing_count": len(missing),
    "bad_count": len(bad),
    "avg_mean_abs_error": sum(maes) / len(maes) if maes else None,
    "worst_mean_abs_error": max(maes) if maes else None,
    "worst_max_abs_error": max(max_abses) if max_abses else None,
    "reported_total_size_unverified": out_index.get("metadata", {}).get("total_size"),
})

if missing:
    print("MISSING:")
    for x in missing[:20]:
        print(" ", x)

if bad:
    print("BAD:")
    for x in bad[:20]:
        print(" ", x)

if missing or bad:
    sys.exit(1)
PY
