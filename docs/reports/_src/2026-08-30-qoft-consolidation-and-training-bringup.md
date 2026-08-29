---
title: QOFT consolidation and quantized-base training bring-up
kind: development
profile: development-log
status: draft
date: 2026-08-30
tags: orbit, qoft, peft, int4, fp8, consolidation, slurm
branch: feature/relocate-conversion-scripts (= feature/generic-int4-adapter)
tip: 98eca8d2
repo: Sphere-AI-Lab/Megatron-Bridge
cluster: slurm (slinky, H200/B200)
---

<section class="report-summary" aria-label="Outcome">
  <p class="summary-label">Outcome</p>
  <p class="summary-title">The five per-model QOFT entrypoints are consolidated into one generic entrypoint plus a second launcher, verified faithful by config-equivalence and strictly better at runtime — but quantized-base training still does not produce a loss value on this environment.</p>
  <p class="summary-detail">Two independent blockers remain, neither introduced by the consolidation: a NaN of unidentified origin in the Moonlight INT4 forward pass (ten hypotheses eliminated), and DeepEP being unbuildable for Qwen3 FP8 because the environment's pip-CUDA wheels carry three different minor versions. The consolidation itself is committed and pushed.</p>
</section>

The intended end state is two launchers: `run_peft_finetune.sh` for high-precision bases that need no
special handling, and `run_qoft_finetune.sh` for quantized bases that require a converted checkpoint.
The first already existed. This effort built the second and replaced the five per-model QOFT
entrypoints behind it. The consolidation is done and verified. Making it actually *train* on this
cluster is not, and the reasons are environmental rather than structural.

## Status

<p><span data-status="done">done: consolidation written and pushed</span> ·
<span data-status="done">done: config-equivalence verified</span> ·
<span data-status="done">done: paired runtime old-vs-new</span> ·
<span data-status="done">done: 7 defects fixed</span> ·
<span data-status="open">open: Moonlight INT4 NaN</span> ·
<span data-status="open">open: DeepEP / CUDA wheel skew</span> ·
<span data-status="open">open: retire old entrypoints</span></p>

## What was built

| File | Role |
|---|---|
| `scripts/orbit/finetune_qoft.py` | Generic entrypoint. Detects the architecture from the HF config and applies each one's retired-entrypoint settings verbatim. `ARCH_SPECS` is the single place a new architecture is added. |
| `scripts/orbit/models/_qoft_common.py` | Shared machinery: the INT4 checkpoint-load patch stack (`scope="experts"` for Kimi/Moonlight, `scope="all"` for Qwen3 dense+router triplets), the FP8 patch pair and `scale_inv` pre-wrap hook, the Moonshot tokenizer vocab clamp, and the memory-profile / NaN-trace diagnostics that previously existed in three near-identical copies. |
| `scripts/orbit/run_qoft_finetune.sh` | Env-var launcher mirroring `run_peft_finetune.sh`. |

Supported today: `Qwen3MoeForCausalLM` (fp8/int4/nvfp4), `KimiK25ForConditionalGeneration`
(int4/nvfp4), `DeepseekV3ForCausalLM` (int4).

## Verification

**Config equivalence.** A harness builds the config through the old entrypoint and the new one for
the same CLI intent, captures each `ConfigContainer` at the moment it would reach `finetune()`, and
diffs normalized JSON snapshots. Both Qwen3 paths reduce to **six differences, all deliberate**:
`save` / `load` / `save_interval` / `wandb_save_dir` (checkpoint saving is now opt-in via
`--save-checkpoints`) and `tensorboard_dir` / `wandb_exp_name` (unified output slug). The harness
also caught one *unintended* deviation — `log_interval` 10 → 1 on Qwen3 — which is fixed, so Qwen3
again inherits the recipe default while Kimi and Moonlight keep their explicit values.

Moonlight and Kimi report `INCOMPLETE` for an unambiguous reason: **the old entrypoints cannot run at
all** on the current Megatron pin, so no baseline exists.

**Paired runtime.** Each pair runs old and new back-to-back on the same node and compares exit codes
and normalized error signatures.

| Pair | old | new | Reading |
|---|---|---|---|
| `moonlight_int4_load` | ✗ `tp_group` TypeError | **✓ rc=0** | new works, old cannot |
| `kimi_tiny_int4_load` | ✗ dead `finetune_utils` import | **✓ rc=0** | new works, old cannot |
| `qwen3_fp8_train2` | ✗ `DeepEP is not installed` | ✗ identical | faithful; environment gap |
| `moonlight_int4_train2` | ✗ fails at load | ✗ NaN in first forward | new reaches further |

## Defects fixed

Three were inherited verbatim from the old entrypoints and are the reason the new path runs where the
old one cannot:

1. **`apply_swiglu_sharded_factory` signature** — `megatron.core` added a `tp_group` keyword; the
   copied three-argument wrapper raised `TypeError`. Now signature-agnostic.
2. **`default_peft_config` import** — upstream moved it out of `recipes.utils.finetune_utils`, which
   no longer exists. Now imported from `recipes.utils.dataset_utils`.
3. **Bare `squad` dataset id** — `huggingface_hub >= 1.0` requires `namespace/name`, so SFT via these
   recipes failed with `HfUriError` before training. Normalized to `rajpurkar/squad`.

Four more are fidelity or robustness fixes: `log_interval` (above), `lr_decay_iters` on the Qwen3
NVFP4 path, Moonlight's forced INT4 chunk defaults, and re-zeroing `oft_r` parameters that were still
on the meta device before `to_empty_if_meta_device` materialized them (defence in depth — measurement
later showed Moonlight's adapters were already zero).

## Open: Moonlight INT4 NaN

Training executes — checkpoint load, model build, forward — then the **first forward pass** emits
non-finite activations and the NaN-in-gradient guard aborts. No loss value has ever been produced.
The `iteration 2` in the error is Megatron's rerun-state-machine counter, not two completed steps.

Ten hypotheses eliminated, each by measurement:

| Hypothesis | Evidence against |
|---|---|
| Degenerate LR schedule | NaN at step 0; persists over 10 iterations |
| Chunked sparse INT4 fallback | `--int4-active-expert-chunk-size 0` — identical |
| INT4 chunk granularity | `--int4-active-expert-chunk-size 1` — identical |
| TE grouped-GEMM backend | `--int4-grouped-chunk-backend te` — identical |
| Corrupt stored weights | all 4,992 triplets scanned: 0 non-finite scales, **0 zero scales**, scales 0.0025–0.41, 80 dequantizations all finite |
| Uninitialized adapters | all 133 `oft_r` params: on CUDA, finite, **exactly zero** |
| OFT rotation math | tracer reports `source=input` — the NaN *arrives* already formed |
| Fused MoE permutation | `moe_permute_fusion=False` — NaN moves to layer 0 attention |
| Shared-expert overlap | `moe_shared_expert_overlap=False` — NaN at layer 18 |
| Attention backend | `flash` and `unfused` both identical; fused path separately hits a cuDNN error |

Signature: **114 complete token-rows of 14,280 are NaN** while all other values are sane
(`finite_abs_max ≈ 0.84`), at step 0, and **the layer varies between otherwise identical runs**
(1, 0, 18). Row-aligned, nondeterministic corruption arriving as module *input* points upstream of
the adapters — most plausibly a dispatch/permutation buffer whose padding or unassigned rows are
never written, or a TE/cuDNN incompatibility on this MLA build. Not localized further.

## Open: DeepEP and the CUDA wheel skew

Qwen3 FP8 needs the recipe's native flex/DeepEP dispatcher. DeepEP is not installed, and building it
fails on a CCCL assertion:

```text
error "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
```

The check requires compiler and header CUDA versions to match. This environment's "toolkit" is
assembled from independently versioned pip wheels that have drifted:

| Package | Version |
|---|---|
| `nvidia-cuda-nvcc` | 13.4.46rc1 |
| `nvidia-cuda-cccl` | 13.3.4.1.2rc1 |
| `nvidia-cuda-runtime` | 13.0.96 (`CUDART_VERSION 13000`) |

This is **fixable, not fundamental** — align the wheel versions, or set
`CCCL_DISABLE_CTK_COMPATIBILITY_CHECK` scoped to the build (the header comment sanctions exactly
this). The build was stopped before attempting it, so the fix is untested. Note also that the PyPI
`deep_ep` 1.0.0 sdist is separately broken: it ships `.cu` sources without `configs.cuh`; the GitHub
tree is complete.

Do **not** work around this by forcing `moe_token_dispatcher_type=alltoall`: that changes the MoE
layout and breaks the FP8 sharded-state-dict shapes.

## Environment work

This environment had never executed a finetune path, and each attempt peeled one layer. Added:
`diffusers` 0.40.0, `megatron-energon` 7.4.1, `pybind11`, `ninja`; Megatron's dataset C++ helpers
built; SQuAD cached.

The notable one: **`nvidia-resiliency-ext` 0.6.0 built from GitHub source.** `megatron.core` asserts
nvrx ≥ 0.6.0 when it is present, while `bridge/training/pretrain.py` imports it unconditionally — and
0.6.0 exists on neither public PyPI nor `pypi.nvidia.com` (both stop at 0.4.1). NVIDIA tags it on
GitHub without publishing. Build recipe: `GIT_SSH_COMMAND="env LD_LIBRARY_PATH= ssh"` (the gitconfig
rewrites https→ssh and conda's libssl breaks `/usr/bin/ssh`), `CUDA_HOME` pointed at the env's
pip-CUDA tree, and `STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD=1` to skip the optional CUPTI extension.

Two self-inflicted regressions were made and fully reverted: a `transformers` downgrade to 5.6 (the
fork's metadata pins `<=5.6` but its code needs 5.12.1), and an nvrx 0.4.1 install plus `__version__`
shim that broke `megatron.core` import for everything until removed.

## Next actions

1. **Run on a known-good runtime.** The NVIDIA container image this Megatron pin targets should carry
   consistent DeepEP, cuDNN/TE and nvrx. Repairing this env piecemeal has hit diminishing returns.
2. **DeepEP:** retry with `CCCL_DISABLE_CTK_COMPATIBILITY_CHECK` scoped to the build, or align the
   CUDA wheels. This is the shortest path to a training run that produces a loss.
3. **Moonlight NaN:** hand to the owner of orbit's MoE/attention path with the evidence above; the
   row-aligned, layer-varying signature is the key clue.
4. **Retire the old entrypoints** once training is verified — they are deliberately retained as the
   paired-verification baseline.

## Reproduction

```bash
# worktree ~/miles-orbit-dev/worktrees/mbridge-relocate, branch tip 98eca8d2
export PYTHONPATH="$WT/src:$WT/3rdparty/Megatron-LM"   # env editables point at other trees
export NVTE_FUSED_ATTN=0                               # else fused attention hits a cuDNN error

torchrun --nproc_per_node=2 scripts/orbit/finetune_qoft.py \
    --quant int4 --hf-model-path $MODELS/Moonlight-16B-A3B \
    --pretrained-checkpoint $CKPT/moonlight-int4-mcore \
    --tp 1 --ep 2 --train-iters 3 --global-batch-size 4 \
    --skip-eval --log-interval 1 --debug-nan
```

`--skip-train` exercises load only and succeeds. `QOFT_ADAPTER_INIT_CHECK=1` reports adapter
parameter state at train start. Run stores with full logs live under
`~/.local/state/remote-cluster-runs/slurm/megatron-bridge/feature-relocate-conversion-scripts/`
(Slurm jobs 5079–5152).
