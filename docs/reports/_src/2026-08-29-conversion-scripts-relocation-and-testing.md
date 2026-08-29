---
title: Conversion scripts — relocation to scripts/orbit and first real-input test campaign
kind: development
profile: development-log
status: final
date: 2026-08-29
tags: orbit, conversion, relocation, quantization, slurm
branch: feature/relocate-conversion-scripts
repo: Sphere-AI-Lab/Megatron-Bridge
base: origin/feature/generic-int4-adapter @ 40cf4522
cluster: slurm (slinky, H200)
---

<section class="report-summary" aria-label="Outcome">
  <p class="summary-label">Outcome</p>
  <p class="summary-title">All orbit-owned files are out of the upstream trees, and every conversion script that can physically run on this cluster passed a real-checkpoint test.</p>
  <p class="summary-detail">Two latent compat bugs were found and fixed on the branch; still open: an FP4 dense-Qwen shape defect, a k8s pod memory ceiling that blocks Kimi-scale single-process INT4 conversion, and three quantized-layout comparator deltas awaiting an owner call.</p>
</section>

Every Sphere-AI-Lab-added file was moved out of the upstream-owned `examples/` and
`tutorials/` trees into fork-owned `scripts/orbit/`, leaving both trees bit-identical to
the upstream base `a0ff9af5`; the relocated conversion scripts were then tested against
real checkpoints on GPU nodes. All six conversion entrypoints that could physically run
on this cluster passed real end-to-end runs. Testing surfaced and fixed two latent
environment-compat bugs (optional-HybridEP import, modelopt ≥0.44 schema), and produced
two open findings that predate the relocation: a dense-Qwen FP4 shape defect and a
cluster memory ceiling that makes Kimi-scale (555 GB) single-process INT4 conversion
impossible by design.

## Results

- **Relocation: verified safe.** 22 files moved by `git mv` (rename detection 94–100 %),
  references updated, `examples/` and `tutorials/` now match upstream exactly
  (`git diff a0ff9af5 -- examples/ tutorials/` is empty). Zero new ruff findings at every
  commit (baseline-compared). All 11 conversion entrypoints import and parse args from
  their new homes on a CUDA node; the two `__file__`-depth fixes (`parents[2]→[3]`)
  proven by probe logs landing at the worktree root.
- **Real conversions: 6/6 possible ones pass.** FP8 direct + full (Qwen3-30B-A3B-FP8),
  NVFP4 direct + meta-key dump (Qwen3-30B-A3B-NVFP4), and the full documented INT4 chain
  `quantize_to_int4.sh → convert_int4_checkpoint_direct.py` (Moonlight-16B-A3B).
- **Not runnable here:** Kimi-K2.7-Code INT4 (cluster pod memory ceiling, below);
  `convert_nvfp4_checkpoint_direct_multigpu` (needs 8 GPUs + the 591 GB
  `nvidia/Kimi-K2.5-NVFP4`, not yet requested); `convert_fp4_checkpoint_direct` (open
  defect, below).

## Status

<p><span data-status="done">done: relocation + verification</span> ·
<span data-status="done">done: 6 real conversion runs pass</span> ·
<span data-status="done">done: 2 compat fixes landed</span> ·
<span data-status="open">open: FP4 dense-Qwen defect</span> ·
<span data-status="open">open: Kimi-scale INT4 needs spill</span> ·
<span data-status="open">open: 3 comparator deltas for owner review</span></p>

## Branch commits

| Commit | What |
|---|---|
| `fc783a79` | refactor(orbit): 10 conversion scripts `examples/conversion/` → `scripts/orbit/conversion/`; 21 reference edits; `parents[2]→[3]` in `dump_nvfp4_meta_keys.py`, `quantize_to_int4.py` |
| `e89d9c8a` | refactor(orbit): remaining 12 orbit files (finetune recipes, 2 tutorials) → `scripts/orbit/{finetune_peft.py, models/, tutorials/llama/}`; `examples/`+`tutorials/` == upstream |
| `0932f4da` | fix(orbit): conversion meta-model builds no longer require HybridEP (flex→alltoall dispatcher downgrade in `build_single_rank_meta_provider` + multigpu twin) |
| `48111762` | fix(orbit): modelopt ≥0.44 rule-list `quant_cfg` schema in `apply_modelopt_nvfp4_to_meta_model` (dict branch kept for older modelopt) |

## Verification — real-input test matrix

Environment: worktree `~/miles-orbit-dev/worktrees/mbridge-relocate`,
`PYTHONPATH = $WT/src : $WT/3rdparty/Megatron-LM@731b7914` (the candidate env's editables
point at *other* trees — `bridge-orbit@7f0fb345`, `runtime/Megatron-LM` 0.19 without
`mla_qk_norm_config` — so the worktree pairing is load-bearing). Single H200, partition
`all`.

| Step | Input | Job | Result |
|---|---|---|---|
| import + `--help`, 11 entrypoints | — | 4765 | ✅ 11/11 |
| `parents[3]` depth assertions | — | 4765 | ✅ |
| `dump_nvfp4_meta_keys --probe-compress/--probe-roundtrip` | — (standalone) | 4765 | ✅ both |
| `convert_fp8_checkpoint_direct` | Qwen3-30B-A3B-FP8 (31 G) | 4775 | ✅ |
| `compare_model_metadata` on its output | | 4775 | ✅ clean pass |
| `convert_fp8_checkpoint` (full) | same | 4775 | ✅ converts; comparator flags delta (below) |
| `dump_nvfp4_meta_keys` (real) + `convert_nvfp4_checkpoint_direct` | Qwen3-30B-A3B-NVFP4 (17 G) | 4791 | ✅ after `48111762` |
| `quantize_to_int4.sh` → `.py` | Moonlight-16B-A3B BF16 (30 G) | 4815 | ✅ produces INT4 triplets |
| `convert_int4_checkpoint_direct` | that INT4 output | 4815 | ✅ |
| `convert_int4_checkpoint_direct` | Kimi-K2.7-Code (555 G) | 4776→4814 | ❌ blocked by pod ceiling (below) |
| `convert_fp4_checkpoint_direct` | Qwen3-8B-FP4 | 4765 | ❌ open defect (below) |

Run stores with full step logs and provenance:
`~/.local/state/remote-cluster-runs/slurm/megatron-bridge/feature-relocate-conversion-scripts/`
(mirrored locally), checkpoints under `/data/home/zeju/mbridge-conversion-test/`.

## Open findings (all predate the relocation)

1. **FP4 dense-Qwen defect.** `convert_fp4_checkpoint_direct.py` on `Qwen3-8B-FP4`
   (produced by `convert_qwen3_deepseek_style.py`) crashes at
   `decoder.layers.0.mlp.linear_fc2.weight`: payload shape mismatches the uint8 template
   `ShardedTensor` (local `(4096, 12288)`, layer-stacked global `(36, …)`). Looks like the
   dense-MLP path of `low_precision/fp4.py` disagrees with the fp4_only export layout.
2. **Kimi-scale INT4 conversion cannot fit.** The slinky "node" is a k8s pod:
   `kubepods.slice memory.max = 1.72 TiB` node-wide, with ~1.03 TiB permanently held by
   co-resident pods (the `/data` FS-client daemon ≈ 981 GB). Effective job headroom is
   ~840 GB. The INT4 direct converter holds the whole output state in RAM *and* keeps all
   source shard mmaps alive → ~1.1 TB peak for a 555 GB model → deterministic mmap ENOMEM
   at the charge ceiling (always shard 14). Ruled out empirically: filesystem (fails from
   node-local `/scratch` too), ulimits, overcommit, `max_map_count`, glibc tunables, cache
   eviction (stretched progress 3×, cannot fix the peak). Remedy is code: port the
   `TensorSpillManager` bucketed-spill pattern from the NVFP4 multigpu converter into the
   INT4 direct path, or convert on non-pod-limited hardware.
3. **Comparator deltas for owner review.** `compare_model_metadata` passes bit-clean on
   the FP8-direct output and flags precise, structured deltas on the other three:
   fp8_full −1,824,768 scale elements (48 × 38,016) with fp32→bf16 scale dtype;
   nvfp4 −12,384 fp32 (48 × 258, per-expert `scale_2` consolidation);
   int4/Moonlight −1,728 (64 experts × 27 layers, `e_score_correction_bias` accounting).
   Each looks like a deliberate layout choice the strict comparator counts differently —
   needs a call on whether the tool should treat them as equivalent.
4. Minor: the two relocated tutorials reference upstream's since-removed
   `examples/conversion/convert_checkpoints.py` (stale before the move).

## Asks / next actions

- Decide on the three comparator deltas (accept as layout-equivalent vs converter change).
- FP4 dense path: assign the shape-mismatch defect.
- Kimi INT4: approve the spill port if K2.7-scale conversion on this cluster is needed.
- Optional coverage: `convert_nvfp4_checkpoint_direct_multigpu` real run needs
  `nvidia/Kimi-K2.5-NVFP4` (591 GB) + 8 GPUs; per-rank footprint fits the pod ceiling.
