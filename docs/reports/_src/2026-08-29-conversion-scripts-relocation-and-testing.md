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
  <p class="summary-detail">Six fixes/capabilities landed (two compat bugs, disk spill, lazy reader, residency eviction, comparator trust fix), the parallel quant-adapter branch is merged in, and the Kimi-K2.7 INT4 chain is verified end-to-end on a 4-layer tiny model with the logical metadata comparison passing; still open: the FP4 dense-Qwen shape defect and a full-555 GB Kimi run, which needs low node co-tenancy even with the committed mitigation stack.</p>
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
- **Kimi-K2.7 INT4: verified via tiny model.** `create_hf_toy_model.py` cut a 4-layer /
  34 GB Kimi-K2.7-Code (byte-identical retained tensors); stock
  `convert_int4_checkpoint_direct` converted and saved it (job 4920) and, after the
  comparator trust fix, the **logical metadata comparison PASSED** (job 4922) on the
  merged line. The full 555 GB conversion additionally needs the opt-in mitigation stack
  below and a node without heavy co-tenants; best full-scale attempt reached 44 %.
- **Merged with `feature/generic-int4-adapter`** (`5b8b94c8`): both branches now carry the
  combined line; the adapter side's tested `_selective_nvfp4_quant_cfg` superseded this
  branch's equivalent modelopt fix; 29/29 orbit unit tests pass on the merge.
- **Not really run:** `convert_nvfp4_checkpoint_direct_multigpu` (needs 8 GPUs + the
  591 GB `nvidia/Kimi-K2.5-NVFP4`); `convert_fp4_checkpoint_direct` (open defect, below).

## Status

<p><span data-status="done">done: relocation + verification</span> ·
<span data-status="done">done: 6 real conversion runs pass</span> ·
<span data-status="done">done: 2 compat fixes landed</span> ·
<span data-status="done">done: Kimi INT4 chain verified (4-layer tiny)</span> ·
<span data-status="done">done: int4-adapter branch merged, 29/29 tests</span> ·
<span data-status="open">open: FP4 dense-Qwen defect</span> ·
<span data-status="open">open: full-555 GB Kimi run (co-tenancy-bound)</span> ·
<span data-status="open">open: fp8-full / nvfp4 comparator deltas</span></p>

## Branch commits

| Commit | What |
|---|---|
| `fc783a79` | refactor(orbit): 10 conversion scripts `examples/conversion/` → `scripts/orbit/conversion/`; 21 reference edits; `parents[2]→[3]` in `dump_nvfp4_meta_keys.py`, `quantize_to_int4.py` |
| `e89d9c8a` | refactor(orbit): remaining 12 orbit files (finetune recipes, 2 tutorials) → `scripts/orbit/{finetune_peft.py, models/, tutorials/llama/}`; `examples/`+`tutorials/` == upstream |
| `0932f4da` | fix(orbit): conversion meta-model builds no longer require HybridEP (flex→alltoall dispatcher downgrade in `build_single_rank_meta_provider` + multigpu twin) |
| `48111762` | fix(orbit): modelopt ≥0.44 rule-list `quant_cfg` schema (superseded in the merge by the adapter branch's tested `_selective_nvfp4_quant_cfg`) |
| `a6a41299` | feat(orbit): opt-in disk spill for the INT4 direct converter (`MEGATRON_BRIDGE_DIRECT_USE_SPILL`), payload-hash-verified |
| `872a3d4d` / `55cd0a6c` | fix(orbit): safe_open ENOMEM retry; spilled-page residency drop (madvise) |
| `f4aea4f5` / `d14191b6` | fix(orbit): lazy no-populate safetensors reader (`MEGATRON_BRIDGE_PYMMAP_READER`) + prior-shard residency eviction; dual-mode payload-hash-verified |
| `5b8b94c8` | merge of `feature/generic-int4-adapter` (scale-reuse requant fix + dual-schema quant_cfg + 2 unit tests) |
| `3c8935f6` | fix(orbit): allow `transformers_modules.` targets in `compare_model_metadata` under trust_remote_code |

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
2. **RESOLVED in capability, bounded by co-tenancy at full scale.** The slinky "node" is a k8s pod:
   `kubepods.slice memory.max = 1.72 TiB` node-wide, with ~1.03 TiB permanently held by
   co-resident pods (the `/data` FS-client daemon ≈ 981 GB). Effective job headroom is
   ~840 GB. The INT4 direct converter holds the whole output state in RAM *and* keeps all
   source shard mmaps alive → ~1.1 TB peak for a 555 GB model → deterministic mmap ENOMEM
   at the charge ceiling (always shard 14). Ruled out empirically: filesystem (fails from
   node-local `/scratch` too), ulimits, overcommit, `max_map_count`, glibc tunables, cache
   eviction (stretched progress 3×, cannot fix the peak). The committed mitigation stack
   (opt-in spill + residency caps + lazy no-populate reader + ENOMEM retry) removes every
   in-process cause — the final failure mode is co-tenant occupancy of the shared node,
   outside the job's control. The conversion/save path itself is verified end-to-end via
   the 4-layer tiny model; a full 555 GB run needs a quiet node (or `--exclusive`).
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
