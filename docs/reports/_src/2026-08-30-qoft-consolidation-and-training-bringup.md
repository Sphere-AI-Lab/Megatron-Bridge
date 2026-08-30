---
title: QOFT consolidation and quantized-base training bring-up
kind: development
profile: development-log
status: draft
date: 2026-08-30
tags: orbit, qoft, peft, int4, fp8, nvfp4, consolidation, slurm
branch: feature/relocate-conversion-scripts (= feature/generic-int4-adapter)
tip: 6d61cc04
repo: Sphere-AI-Lab/Megatron-Bridge
cluster: slurm (slinky, H200/B200)
reporting_period: 2026-08-29 to 2026-08-30
---

<section class="report-summary" aria-label="Outcome">
  <p class="summary-label">Outcome</p>
  <p class="summary-title">The consolidated QOFT entrypoint now loads the NVFP4 Qwen3 checkpoint end-to-end and has produced the first loss value of the whole campaign (4.79) — the remaining blocker is uninitialized token-row garbage entering the backward pass at the MoE boundary, with per-rank gradient magnitudes spanning fifteen orders of magnitude.</p>
  <p class="summary-detail">Wave 1 (2026-08-29) consolidated five per-model entrypoints into one and hit two environmental blockers. Wave 2 (2026-08-30) unified the NVFP4 checkpoint contract into one source of truth shared by the trainer and the RL loader, fixed the load end-to-end in six three-minute iterations, and localized the remaining defect precisely enough to hand to a MoE/dispatcher owner: forward is proven healthy; backward receives junk gradient rows for padded tokens — the same fingerprint as the still-open Moonlight forward NaN.</p>
</section>

## Status

<p><span data-status="done">done: consolidation written and pushed</span> ·
<span data-status="done">done: config-equivalence verified</span> ·
<span data-status="done">done: NVFP4 load contract unified + verified</span> ·
<span data-status="done">done: first loss produced (4.79, tp1/ep4)</span> ·
<span data-status="open">open: backward pad-row garbage (MoE boundary)</span> ·
<span data-status="open">open: tp≥2 scalar sharding validation</span> ·
<span data-status="open">open: Moonlight INT4 NaN (same fingerprint?)</span> ·
<span data-status="open">open: DeepEP / CUDA wheel skew</span> ·
<span data-status="open">open: Kimi padding_mask forward signature</span> ·
<span data-status="open">open: retire old entrypoints</span></p>

## Wave 1 (2026-08-29): consolidation

The intended end state is two launchers: `run_peft_finetune.sh` for high-precision bases and
`run_qoft_finetune.sh` for quantized bases that require a converted checkpoint. The five per-model
QOFT entrypoints were replaced behind the second one.

| File | Role |
|---|---|
| `scripts/orbit/finetune_qoft.py` | Generic entrypoint. Detects the architecture from the HF config; `ARCH_SPECS` is the single place a new architecture is added. |
| `scripts/orbit/models/_qoft_common.py` | Shared machinery: the INT4/FP8 (and now NVFP4) checkpoint-load patch stacks, the Moonshot tokenizer vocab clamp, and the memory-profile / NaN-trace diagnostics. |
| `scripts/orbit/run_qoft_finetune.sh` | Env-var launcher mirroring `run_peft_finetune.sh`. |

Supported: `Qwen3MoeForCausalLM` (fp8/int4/nvfp4), `KimiK25ForConditionalGeneration` (int4/nvfp4),
`DeepseekV3ForCausalLM` (int4). Verified by a config-equivalence harness (Qwen3 reduces to six
deliberate differences) and paired old-vs-new runtime runs; seven inherited defects fixed. Full
detail in the 2026-08-29 sections below and the prior git history of this report.

## Wave 2 (2026-08-30): one checkpoint contract, first loss

### Root cause of the NVFP4 load failure

Job 5162 (the first NVFP4 training attempt ever) died in the distributed-checkpoint planner:
the model requested per-expert keys (`mlp.experts.experts.96.linear_fc1.weight_w`) while the
converter stored fused-indexed families (`mlp.experts.linear_fc1.weight96_w/_v` +
`weight_quantizer._scale96`). Diagnosis:

1. **The converter's names are canonical.** `convert_nvfp4_checkpoint_direct.py` builds a plain
   Megatron meta model (single rank), applies ModelOpt NVFP4 to that skeleton, and uses its
   sharded state dict as the on-disk schema — matching, name for name, the contract documented in
   `orbit/quant/nvfp4_utils.py`.
2. **The trainer's shape was an accident.** The fork's `orbit-seam(modelopt)` in
   `bridge/training/checkpointing.py` restores a checkpoint's `modelopt_state/` directory
   **unconditionally** (no config gate) before building the load schema. The restored per-expert
   module layout emits keys that exist nowhere on disk. `restore_modelopt_state=True` in the
   entrypoint was a red herring — the seam fires on checkpoint content alone.
3. **The intended loader existed but had no caller.** `transform_sharded_state_dict_for_nvfp4` +
   `register_nvfp4_buffers_after_load` (grouped experts) and the `_dense` pair implement exactly
   this translation; the only caller in the entire workspace was orbit's RL
   `low_precision_bootstrap.py` — the production-proven path. No finetune entrypoint, current or
   retired, ever wired them. This cell had never worked anywhere.

### Design decision: single source of truth

<aside class="decision">
  <p class="block-label">Decision</p>
  <p>The checkpoint contract lives in exactly one place per quant format — the
  <code>orbit.quant</code> and <code>orbit.low_precision</code> contract modules — and every
  consumer (finetune trainer, RL loader, and in a follow-up pass the converters) imports it.
  ModelOpt and OFT wrapping stay out of the on-disk contract: ModelOpt may compute values in the
  converter but never names anything at runtime; adapters wrap after the base load. Checkpoints
  are regenerable, so no backward compatibility is owed. Rejected: renaming the converter output
  to match the trainer's accidental layout (chains the format to ModelOpt internals and breaks the
  proven RL consumer), and a standalone declarative schema (a second source of truth).</p>
</aside>

Implementation (commits `52b54eb6` wiring, `70bf69ed` library fix, `6d61cc04` diagnostics —
**all three exist only in the cluster worktree** `~/miles-orbit-dev/worktrees/mbridge-relocate`
on the slurm login; a login-pod outage blocked the push at the time of writing, so they are not
on origin. The next section summarizes their content so this report is reviewable without
repository access):

- `install_nvfp4_checkpoint_load_patches()` in `_qoft_common.py`, shaped identically to the INT4
  installer: plain bf16 meta build, dense+expert transforms applied in `_generate_model_state_dict`,
  packed expert buffers + dense bf16 dequantization in `_load_model_state_dict`, the ModelOpt
  auto-restore seam no-op'd, `_extra_state` requests dropped (they use a naming family the grouped
  runtime cannot reproduce; TE extra state is not needed for plain-bf16 + packed-buffer compute),
  and explicit-layer schema (`non_homogeneous_layers`) detected from the checkpoint's own keys.
- **Preflight**: after transformation, the set of requested tensor keys is compared against the
  checkpoint index; a mismatch fails with a sorted 8-key diff instead of a multi-megabyte planner
  `KeyError`. Every remaining contract gap in the ladder below surfaced as one readable line.
- Library fix in `nvfp4_utils.py`: quantizer **runtime dict keys** must use the *local* expert
  index (the on-disk `key` fields keep the global index). Mixing the spaces left every EP rank
  except rank 0 registering expert weights and scales in different groups — evidenced by
  6144 half-groups instead of 3072 on EP ranks 1–3 in job 5191. The RL loader shares this
  function and had the same latent EP>1 defect.
- The entrypoint's nvfp4 branch no longer sets `restore_modelopt_state` or the TE-FP4 compute
  preset `bf16_with_nvfp4_mixed` (that preset enables fp4 GEMMs — a different mechanism than the
  packed-buffer runtime); plain bf16 mixed precision instead, and the Kimi big-block bf16 config
  now applies to nvfp4 too.

### The three commits, reviewable without repository access

**`52b54eb6` — fix(orbit): wire NVFP4 checkpoint loading into the generic QOFT entrypoint**
(`scripts/orbit/models/_qoft_common.py` +220, `scripts/orbit/finetune_qoft.py` ~20 changed).
Idea: the trainer must speak the converter's on-disk contract, obtained from the shared contract
modules — never from ModelOpt's runtime shapes. The new
`install_nvfp4_checkpoint_load_patches(pretrained_checkpoint, arch_label, …)` monkeypatches the
same three bridge hooks the proven INT4 installer patches:

```text
_generate_model_state_dict  (request side)
    ├─ detect explicit-layer schema from checkpoint keys → metadata["non_homogeneous_layers"]=True
    ├─ drop *._extra_state requests (object naming the grouped runtime cannot reproduce)
    ├─ transform_sharded_state_dict_for_nvfp4_dense(state, checkpoint_keys)   # presence-driven
    ├─ transform_sharded_state_dict_for_nvfp4(state)                          # grouped experts
    └─ PREFLIGHT: requested tensor keys ⊆ checkpoint index, else readable 8-key diff
mcore_to_pyt_state_dict     (planner side)  → materialize meta tensors on CPU
_load_model_state_dict      (install side)
    ├─ register_nvfp4_buffers_after_load_dense(model, loaded)   # dense → bf16 Parameters
    ├─ register_nvfp4_buffers_after_load(model, loaded)         # experts → packed buffers
    ├─ delete NVFP4 entries, assign-load the rest, validate missing/unexpected
    └─ to_empty(meta→cuda) + re-zero meta-materialized oft_r
_maybe_restore_modelopt_state_for_sharded_load → no-op        # the root-cause seam
```

Entrypoint side: the nvfp4 branch drops `restore_modelopt_state=True` and the TE-FP4 compute
preset `bf16_with_nvfp4_mixed` (plain `bf16_mixed` for Qwen3; the Kimi big-block bf16 config now
applies to nvfp4 too), and `main()` installs the patches for `--quant nvfp4`.
Result: verified by the r2→r7 ladder below — load completes, `lm loss 4.790725` at iteration 1.

**`70bf69ed` — fix(orbit): keep NVFP4 quantizer runtime keys in the local expert index space**
(`src/megatron/bridge/orbit/quant/nvfp4_utils.py`, ~8 lines). Idea: the transform↔register pair
communicates through runtime dict keys; those must live in one index space (local), while only
the on-disk `key` fields carry global expert indices. The essential hunk:

```python
# before: dict keys mixed spaces (weight halves local, quantizer entries global)
new_sd[f"{prefix}.weight_quantizer._scale{expert_global_idx}"] = scale_st
# after: runtime dict keys local; ShardedTensor(key=...) keeps expert_global_idx
expert_local_idx = int(m.group(2))
new_sd[f"{prefix}.weight_quantizer._scale{expert_local_idx}"] = scale_st   # same for _double_scale/_amax
```

Result: expert buffer groups per rank went from 6144 half-groups (EP ranks 1–3, job 5191) to the
correct 3072 complete groups on all ranks (job 5195). Loss unchanged (forward had been rescued by
a fallback suffix-mapping in the OFT wrapper), but placeholders now empty correctly and the RL
loader — which shares this function — loses the same latent EP>1 defect.

**`6d61cc04` — chore(orbit): attribute grad norms per parameter in the NaN-trace callback**
(`_qoft_common.py`, ~25 lines). Idea: a finite-but-huge global grad norm is unactionable until it
names a parameter and a rank; the global norm is a cross-rank reduction, so rank 0 alone proves
nothing. The `--debug-nan` callback now prints, on every rank at each traced step end, the local
param-grad total and the top parameters by gradient norm (reads `main_grad`, falls back to
`.grad`). Result: the r10 per-rank table in the evidence section — which converted "grad norm is
2.7e17" into "rank-varying garbage spanning fifteen orders of magnitude on the same parameter
families", the pivotal fact behind the pad-row hypothesis.

### The run ladder

Six three-minute iterations, each clearing exactly one gate (all tp1/ep4 on 4×H200 unless noted;
run store execution `20260829T235425Z-r2281915244`):

| Run | Job | Gate reached | Result |
|---|---|---|---|
| r2 | 5185 | checkpoint-key read | my reader got the checkpoint root; torch-dist lives in `iter_0000000/` → resolver added |
| r3 | 5188 | preflight | readable 5-key diff: homogeneous vs explicit layer schema → detection-driven `non_homogeneous_layers` |
| r4 | 5189 | distributed load | all ~56k tensor requests match; `_extra_state` **objects** mismatch (per-expert vs offset naming) → requests dropped |
| r5 | 5190 | sharding validation | tp2: scalar quantizer entries (`_double_scaleN`, rank-0 tensors) fail mcore access-pattern validation → open item, tp1 probe |
| r6 | 5191 | **full training step** | **first loss ever: `lm loss 4.790725`**, but grad norm 2.7e17 at step 0, NaN at step 1; register counts expose the EP index bug |
| r7 | 5195 | same, index bug fixed | 3072 groups on all ranks; loss **bit-identical** 4.790725 — forward provably unaffected; blowup persists |

Two discriminator runs followed:

- **r8 (5197, LoRA)**: crashed in the *base TE grouped forward* (`reshape 0 elements`) — LoRA's
  wrapper calls the real TE forward, which hit the deliberately-emptied weights. Two conclusions:
  QLoRA on grouped NVFP4 is **not implemented** (only the OFT layer has the packed-buffer branch),
  and — more valuable — the OFT runs never touched the base TE forward (they would have crashed
  identically), so the packed-buffer branch was active and the 4.79 forward is real.
- **r9/r10 (5198/5199, gradient attribution)**: the NaN-trace callback now prints per-rank,
  per-parameter gradient norms (commit `6d61cc04`).

### Evidence package: the remaining backward defect

Step 0 of every OFT run: healthy loss, catastrophic gradient norm. Per-rank attribution (r10,
job 5199, identical loss 4.79073, global grad norm 2.73e17):

```text
rank0  local param-grad total-norm 272.541    top: 142.3   layers.0.self_attention.linear_proj.adapter.oft_r
rank1  local param-grad total-norm 5.80620e5  top: 5.80e5  layers.0.self_attention.linear_qkv.adapter.oft_r
rank3  local param-grad total-norm 2.80126e10 top: 2.13e10 layers.12.self_attention.linear_qkv.adapter.oft_r
rank2  local param-grad total-norm 1.09287e18 top: 6.70e17 layers.0.self_attention.linear_proj.adapter.oft_r
```

Same parameter families, magnitudes spanning **fifteen orders of magnitude across ranks** — the
signature of uninitialized memory, not of a deterministic scaling error. At step 1 the NaN-trace
hook reports the first non-finite value arriving as **grad_output** at
`decoder.layers.4.mlp.experts.linear_fc2.adapter` (OFTRotationModule): `nan=1536` in a
`(17168, 768)` bf16 tensor — **exactly two complete token rows** of a *padded* token buffer
(17168 = padded count). The affected layer varies between runs.

What is proven about the forward direction:

- loss is sane and *bit-identical* between r6 (three ranks with broken buffer groups) and r7
  (fixed) — the OFT wrapper's local→global scale-suffix fallback found the correct scale data in
  both runs, so forward numerics never changed;
- the NaN tracer sees no non-finite activation in any traced forward module at any step;
- the LoRA crash proves the packed-buffer branch (not the emptied base TE path) computes the
  forward.

The OFT NVFP4 grouped forward (`orbit/oft/oft_layers.py`, `_forward_nvfp4_grouped`) dequantizes
per expert and uses plain `F.linear` — **no custom backward exists to mis-scale**. It splits the
incoming buffer by `padded_splits` and zero-pads its own output tail; the pad-token *rows* still
flow through the surrounding permute/unpermute machinery.

<aside class="risk">
  <p class="block-label">Unified hypothesis</p>
  <p>The alltoall dispatch/permutation path on this stack leaves pad-row slots unwritten. In the
  forward direction that is the still-open Moonlight INT4 signature (complete token rows of NaN,
  arriving as module <code>input</code>, layer/rank-varying, 114 of 14,280 rows). In the backward
  direction it is what the NVFP4 runs now show: junk gradient rows for padded token positions
  entering at the expert boundary, contaminating every adapter gradient below it (bottom layers
  worst — layers 0–2 top the per-rank tables), finite-but-huge at step 0 and NaN by step 1. One
  mechanism, two symptoms, two architectures. Owner needed: orbit's MoE dispatcher/permute path.</p>
</aside>

Probes an analyst can run next (each ~3 min on 4 GPUs; commands under Reproduction):

1. Zero-fill or mask pad rows: zero `grad_output` rows beyond each expert's real token count
   inside the OFT adapter backward (or zero the dispatcher's buffers at allocation) — if step-0
   grad norms collapse to the rank-0 magnitudes (~1e2), the hypothesis is confirmed.
2. `--moe-permute-fusion false` equivalent (config `moe_permute_fusion=False`) — for Moonlight
   this *moved* the NaN, consistent with buffer-layout dependence.
3. Compare against the RL trainer's dispatcher configuration (orbit repo,
   `backends/megatron_utils/`) — RL trains OFT-on-NVFP4 stably; a difference in its
   pad/capacity handling would be load-bearing.
4. Single-expert-parallel run (`EP=1`, `NUM_GPUS=2`): removes alltoall entirely; a clean run
   pins the defect to the dispatch path rather than the OFT/dequant math.

### Also captured this wave

- **tp≥2 open item**: mcore sharding validation rejects the transform's scalar quantizer entries
  (`Invalid access pattern for ShardedTensor(key='…weight_quantizer._double_scale32', …
  local_shape=(), global_shape=())`) at tp2 — the `_replica_id_with_current_tp_rank` semantics
  need rework for tp>1 loads; tp1 is unaffected (job 5190 error text in the run store).
- **QLoRA on grouped NVFP4 does not exist**: the LoRA wrapper has no packed-buffer branch and
  calls the base TE forward. Implementing it means giving `lora_layers.py` the same dispatch the
  OFT layer has, or dequantizing grouped experts to bf16 at load like the dense path.

## Defects fixed in wave 1

Three inherited breakages (signature-agnostic `apply_swiglu_sharded_factory` wrapper,
`default_peft_config` import moved upstream, bare `squad` dataset id) plus four fidelity fixes —
see git history `6ff91c56`/`98eca8d2` for detail.

## Open: Moonlight INT4 NaN (wave 1, re-framed by wave 2)

Training executes, then the first forward pass emits non-finite activations: 114 complete
token-rows of 14,280 are NaN, all other values sane, the layer varies between identical runs
(1, 0, 18), and the NaN arrives already formed as module *input*. Ten hypotheses were eliminated
by measurement (LR schedule, chunking, backends, stored weights — all 4,992 triplets scanned
clean, adapters exactly zero, rotation math, fused permutation, shared-expert overlap, attention
backend). Wave 2's backward-direction evidence on Qwen3 NVFP4 makes the leading explanation —
a dispatch/permutation buffer whose padding rows are never written — considerably stronger:
see the unified hypothesis above. `NVTE_FUSED_ATTN=0` remains required (fused attention hits a
separate cuDNN error on this build).

## Open: DeepEP and the CUDA wheel skew (wave 1)

Qwen3 FP8 needs the recipe's native flex/DeepEP dispatcher. DeepEP is unbuildable because the
environment's pip-CUDA wheels drifted across three minor versions (`nvidia-cuda-nvcc` 13.4.46rc1,
`nvidia-cuda-cccl` 13.3.4.1.2rc1, `nvidia-cuda-runtime` 13.0.96). Fixable, not fundamental:
align the wheels or set `CCCL_DISABLE_CTK_COMPATIBILITY_CHECK` scoped to the build (untested).
The PyPI `deep_ep` 1.0.0 sdist is separately broken (ships `.cu` without `configs.cuh`); use the
GitHub tree. Do **not** force `moe_token_dispatcher_type=alltoall` for FP8 — it breaks the FP8
sharded-state-dict shapes.

## Open: Kimi training-path forward signature (wave 2, round 1)

Kimi INT4 (tiny, tp1/ep2) loads and enters the training loop, then the first forward call fails:
the packed-SFT step passes `padding_mask`, the fork's `GPTModel.forward` accepts it
(gpt_model.py:332 — why Moonlight/Qwen3 pass this point), but `KimiK25VLModel.forward`
(`models/kimi_vl/modeling_kimi_k25_vl.py:406`) does not declare it. Fix: accept and pass through
to the language model. The load-only smoke could not have caught this. (Job 5163.)

## Environment work (wave 1)

`diffusers` 0.40.0, `megatron-energon` 7.4.1, `pybind11`, `ninja`; Megatron's dataset C++ helpers
built; SQuAD cached. Notably `nvidia-resiliency-ext` 0.6.0 built from GitHub source — the 0.6.0
tag exists on neither public PyPI nor `pypi.nvidia.com`; build recipe:
`GIT_SSH_COMMAND="env LD_LIBRARY_PATH= ssh"`, `CUDA_HOME` at the env's pip-CUDA tree,
`STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD=1`. `transformers` must stay 5.12.1 despite the fork's
`<=5.6` metadata pin.

## Actions

| action | owner | status | evidence or trigger |
|:--|:--|:--|:--|
| Backward pad-row garbage: run probes 1–4, fix dispatcher or mask | MoE/dispatcher owner (or next agent) | Open | this report's evidence package |
| tp≥2 scalar sharding validation for quantized loads | next agent | Open | job 5190 error |
| Phase 2 SSoT: converters consume the contract modules; regenerate checkpoints; unify the three installers behind one generic one; toy round-trip matrix per (arch, quant) | next agent | Open | nvfp4 cell trains stably |
| Kimi `padding_mask` pass-through + rerun (answers the MLA discriminator for Moonlight) | next agent | Open | small fix, job 5163 evidence |
| DeepEP build retry with `CCCL_DISABLE_CTK_COMPATIBILITY_CHECK` | next agent | Open | wave-1 analysis |
| Retire the five old entrypoints | after training verified | Open | deliberately retained as baseline |

## Reproduction

```bash
# worktree ~/miles-orbit-dev/worktrees/mbridge-relocate, branch tip 6d61cc04
export PYTHONPATH="$WT/src:$WT/3rdparty/Megatron-LM"
export NVTE_FUSED_ATTN=0                               # else fused attention hits a cuDNN error

# The NVFP4 cell (produces the loss + the backward blowup; ~3 min on 4×H200):
env QUANT=nvfp4 \
    HF_MODEL_PATH=/data/home/zeju/hf_models/Qwen3-30B-A3B-NVFP4 \
    MEGATRON_CKPT=/data/home/zeju/mbridge-conversion-test/20260829T190607Z-r164228462/qwen3-nvfp4-mcore \
    NUM_GPUS=4 TP=1 EP=4 TRAIN_ITERS=3 GLOBAL_BATCH_SIZE=4 \
    EXTRA_ARGS="--skip-eval --log-interval 1 --debug-nan" \
    bash scripts/orbit/run_qoft_finetune.sh
```

`--debug-nan` prints the per-rank gradient attribution and the first-non-finite trace.
`--skip-train` exercises load only. `QOFT_ADAPTER_INIT_CHECK=1` reports adapter state at start.
Moonlight INT4 repro is unchanged from wave 1 (2 GPUs, `--quant int4`).

Run stores with full logs:
`~/.local/state/remote-cluster-runs/slurm/megatron-bridge/feature-relocate-conversion-scripts/`
— wave 1: jobs 5079–5152; wave 2 round 1: 5162/5163 (exec `20260829T223839Z-r29444327`);
wave 2 NVFP4 ladder: jobs 5185–5199 (exec `20260829T235425Z-r2281915244`, labels
`qoft-qwen3-nvfp4-train-r2` … `-r10-rankattr`). Mirrored on the workstation under the same
suffix.
