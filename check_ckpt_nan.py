"""Check megatron torch_dist checkpoint for NaN/Inf/unusual values."""
import os
import sys
import torch
from pathlib import Path

DEFAULT_CKPT = (
    Path(os.environ.get("MEGATRON_CKPT_ROOT", Path.cwd() / "checkpoints"))
    / "Moonlight-16B-A3B-INT4"
    / "iter_0000000"
)

# Accept path via CLI arg. If a checkpoint root is given (containing
# `latest_checkpointed_iteration.txt`), auto-resolve to its iter_* subdir.
arg = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_CKPT)
ckpt_dir = Path(arg)
latest_file = ckpt_dir / "latest_checkpointed_iteration.txt"
if latest_file.exists():
    it = latest_file.read_text().strip()
    try:
        iter_dir = ckpt_dir / f"iter_{int(it):07d}"
    except ValueError:
        iter_dir = ckpt_dir / it
    if iter_dir.exists():
        ckpt_dir = iter_dir
print(f"Checking checkpoint: {ckpt_dir}")

# Use torch.distributed.checkpoint low-level reader to load all tensors
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
import torch.distributed.checkpoint as dcp

reader = FileSystemReader(str(ckpt_dir))
metadata: Metadata = reader.read_metadata()

# Build an empty state dict from metadata (load full tensors to CPU)
state_dict = {}
for key, md in metadata.state_dict_metadata.items():
    if isinstance(md, TensorStorageMetadata):
        state_dict[key] = torch.empty(md.size, dtype=md.properties.dtype)
    else:
        # BytesStorageMetadata etc
        state_dict[key] = None

# Load into the state dict
dcp.load(
    state_dict=state_dict,
    storage_reader=reader,
    planner=DefaultLoadPlanner(allow_partial_load=True),
)

print(f"Loaded {len(state_dict)} entries from checkpoint")
print("=" * 80)

problems = []
stats = []
for key, t in state_dict.items():
    if not isinstance(t, torch.Tensor):
        continue
    if t.numel() == 0:
        continue
    # Cast to float32 for stats (some may be int / fp8 / bf16)
    try:
        if t.dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
            tf = t.float()
        elif t.dtype == torch.uint8 or t.is_floating_point() is False:
            # skip integer / quantized raw bytes for nan check but still report magnitude
            tf = None
        else:
            tf = t.float()
    except Exception as e:
        print(f"[skip cast] {key} dtype={t.dtype}: {e}")
        continue

    if tf is None:
        # integer tensor — just report range
        mn = int(t.min().item())
        mx = int(t.max().item())
        stats.append((key, str(t.dtype), tuple(t.shape), mn, mx, 0, 0, 0.0, 0.0))
        continue

    n_nan = int(torch.isnan(tf).sum().item())
    n_inf = int(torch.isinf(tf).sum().item())
    mn = float(tf.min().item())
    mx = float(tf.max().item())
    mean = float(tf.mean().item())
    std = float(tf.std().item()) if tf.numel() > 1 else 0.0
    absmax = float(tf.abs().max().item())
    stats.append((key, str(t.dtype), tuple(t.shape), mn, mx, n_nan, n_inf, mean, std))
    if n_nan > 0 or n_inf > 0:
        problems.append((key, n_nan, n_inf))
    elif absmax > 1e4 or (absmax > 0 and absmax < 1e-20):
        problems.append((key, f"unusual absmax={absmax}"))

# Print full per-tensor stats
print(f"{'key':<80} {'dtype':<16} {'min':>12} {'max':>12} {'mean':>12} {'std':>12} {'nan':>6} {'inf':>6}")
for (key, dt, shape, mn, mx, nn, ni, mean, std) in stats:
    short = key if len(key) <= 80 else "…" + key[-79:]
    print(f"{short:<80} {dt:<16} {mn:>12.4g} {mx:>12.4g} {mean:>12.4g} {std:>12.4g} {nn:>6} {ni:>6}")

print("=" * 80)
if problems:
    print(f"PROBLEMS FOUND: {len(problems)}")
    for p in problems:
        print(" ", p)
else:
    print("No NaN/Inf detected. No extreme magnitude outliers.")
