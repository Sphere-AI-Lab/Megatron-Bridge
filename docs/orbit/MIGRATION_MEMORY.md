# Orbit → radixark migration: durable memory

Purpose: everything a fresh session needs to resume this migration without
re-deriving it. Facts here were verified by direct inspection, not assumed.
Read this together with `MIGRATION_PLAN.md` and `UPSTREAM_SEAMS.md`.

Last updated: 2026-08-30

## 1. Goal

Port the Sphere AI Lab "orbit" work (quantized-base PEFT + OFT: INT4 / FP8 /
NVFP4) onto the radixark fork of Megatron-Bridge.

Overriding constraint from the user: **orbit must stay extractable.** The
future workflow is "fetch upstream radixark, then re-apply our changes." So
orbit adapts to radixark; radixark does not get bent to fit orbit.

## 2. Local repos and remotes

| Path | Role |
|---|---|
| `/home/kerryliu/Megatron-Bridge-Spherelab` | **work happens here** |
| `/home/kerryliu/Megatron-Bridge-RadixArk` | target fork, plain clone |
| `/home/kerryliu/Megatron-Bridge` | NVIDIA upstream, plain clone |

Read-only remotes were added inside the Spherelab clone so all three trees are
reachable from one place (no network needed):

```
origin    https://github.com/Sphere-AI-Lab/Megatron-Bridge
radixark  /home/kerryliu/Megatron-Bridge-RadixArk   -> radixark/bridge
nvidia    /home/kerryliu/Megatron-Bridge            -> nvidia/main, tags v0.5.0 v0.5.1 v0.6.0
```

Useful: `git show radixark/bridge:<path>`, `git diff a0ff9af5..HEAD -- <path>`.

### Branches

- `feature/generic-int4-adapter` — the orbit source of truth (do not lose; `53b6e652`).
- `feature/orbit-on-radixark` — **the migration branch**, created off `radixark/bridge` (`bb61fcd0`).

Repo-local git identity is set (global config untouched):
`liulixinkerry <kerryliu1997@gmail.com>`

## 3. Fork topology (verified)

| Fork | NVIDIA base | Date | Version | mcore pin |
|---|---|---|---|---|
| radixark/bridge | `3b792c46` | 2026-05-07 | 0.5.0-dev | `5c7968af` |
| spherelab int4 branch | `a0ff9af5` | 2026-08-28 | 0.7.0-dev | `731b7914` |

- The two bases are **1063 upstream commits apart**.
- `merge-base(spherelab, radixark) == 3b792c46` — that is only radixark's own
  fork point, not a shared customization base.
- Spherelab is based on post-0.6.0 `main` (0.7.0-dev), **not** on the v0.6.0 tag.

### Two premises from the original request that turned out false

1. **"Remove the DeepSeek V4 fork changes."** There are none. DeepSeek V4 is
   entirely upstream NVIDIA code, already present in `a0ff9af5`. Spherelab
   never touched it. Nothing to remove.
2. **"Downgrade spherelab to the common base first."** Unnecessary.
   `git diff a0ff9af5..HEAD` is **34,528 insertions / 7 deletions over 107
   files**, and only **3 upstream files** are modified. There is no upstream
   drift to unwind, so we re-apply orbit directly onto radixark and skip the
   downgrade entirely.

## 4. What orbit actually consists of

`git diff --name-status -M a0ff9af5..HEAD` → 103 A, 3 M, 1 R (the R is a
false rename of an ~empty `__init__.py`; treat as an add).

| Area | Files |
|---|---|
| `src/megatron/bridge/orbit/` | 59 |
| `scripts/orbit/` | 35 |
| `tests/unit_tests/orbit/` | 5 |
| `docs/reports/` | 4 (dropping — see §6) |
| `NOTICE` | 1 |

Modified upstream files — only these three, each marked `orbit-seam(...)`:

- `src/megatron/bridge/training/checkpointing.py` (2 seams)
- `src/megatron/bridge/training/post_training/checkpointing.py` (1 seam)
- `src/megatron/bridge/training/setup.py` (1 seam, pure reorder)

Spherelab made **zero** changes to upstream `src/megatron/bridge/peft/`
(verified: `git diff a0ff9af5..HEAD -- src/megatron/bridge/peft` is empty) and
**zero** changes to `pyproject.toml`.

radixark made **zero** changes to the three seam files. All seam anchors exist
in radixark.

## 5. Verified blockers

Each of these was confirmed by reading both trees.

1. **`_base_returns_tuple` — hard break.**
   `orbit/peft_ext/int4_lora.py:35,39` reads `self._base_returns_tuple`.
   Spherelab's `peft/adapter_wrapper.py` sets it (lines 132/181/188);
   radixark's does not set it anywhere. → `AttributeError` on the first
   `Int4LoRALinear.forward`.
   **Fix (radixark-first):** drop the two branches in orbit. Orbit only ever
   wraps Megatron parallel linears, which always return `(out, bias)`. Do not
   add the attribute to radixark's `AdapterWrapper`.

2. **`models/conversion/quantization_utils.py` absent in radixark.**
   Added upstream in `39b79eb7` (PR #3778), after radixark's base. 582 lines.
   Orbit needs only `dequantize_int4` and `quantize_to_int4`.
   **Fix:** vendor just those two into `orbit/quant/`. Do not add the upstream
   file to radixark — that would be a non-extractable upstream edit.

3. **`default_peft_config` moved upstream.**
   radixark: `recipes/utils/finetune_utils.py:30`.
   spherelab: `recipes/utils/dataset_utils.py:48`.
   Only `scripts/orbit/finetune_qoft.py:458` uses the new path; two other orbit
   scripts already use the old one. **Fix:** one-line import change.

4. **Orbit monkeypatches mcore.** `orbit/low_precision/common.py:186`
   `patch_meta_init_for_te_modules()` reassigns
   `_initialize_affine_weight_cpu` in **both**
   `megatron.core.tensor_parallel.layers` and
   `megatron.core.extensions.transformer_engine`. It is idempotent (guards on
   `__name__ == "_patched_init_affine"`). This is the one genuine
   extractability wart; must be re-verified against mcore `5c7968af`.

5. **Two seam imports are unguarded**, on the checkpoint-resume path:
   `checkpointing.py:3008` and `post_training/checkpointing.py:162`. Deleting
   orbit would `ImportError` on every resume. Guard them so radixark stays
   runnable without orbit.

6. **`AdapterWrapper.sharded_state_dict`** in radixark lacks spherelab's
   `_plain_module_sharded_state_dict` fallback. Risk of dropping delta-only
   adapter weights. Needs a runtime check, not a static one.

7. **`LoRA` dataclass default flip:** radixark's `share_expert_adapters`
   defaults to `False` (spherelab `True`), and radixark has no
   `sequence_parallel_input_regather`. Orbit sets neither, so this is a silent
   behaviour change for MoE LoRA runs, not a break.

### Non-issues (checked, harmless)

- `_get_modelopt_checkpoint_path` / `has_modelopt_state` gained a `ckpt_step`
  kwarg upstream. Orbit calls both with a single positional arg → compatible.
- `AdapterAttributes` fields are identical in both trees.
- `get_adapter_attributes_from_linear` signature narrowed in radixark, but
  orbit's only call site passes `(m, is_expert=is_expert)` → compiles.
- `is_expert_linear`, the TE grouped-linear flags, `PEFT`, `ModuleMatcher`,
  `PEFT_RECOMPUTE_PATCHED` — all present and compatible.
- `register_allowed_target_prefix` exists in radixark
  (`utils/instantiate_utils.py:46`).
- radixark's multi-LoRA vs orbit's INT4-LoRA: **low collision risk.**
  `MultiLoRALinear` / `MultiLoRAGroupedExpertLinear` subclass `AdapterWrapper`
  directly, not `LoRALinear`; orbit's `Int4LoRALinear(LoRALinear)` only
  overrides `forward`; one PEFT object applies per model. Note `multi_lora.py`
  is **upstream** code, not radixark-invented.

## 6. Decisions taken (from the user)

- **mcore:** use radixark's pin `5c7968af`. Orbit must work against it.
- **TE:** use radixark's pin `f031cf87`. Therefore
  `orbit/oft/te_oft/te_oft_layernorm_linear.py` (a copy of TE's
  `LayerNormLinear`, currently pinned to "TE 2.13.0+28777046, updated for
  2.14.0") must be **adapted to f031cf87**, not gated off.
- **Seams:** stay as close to radixark as possible; minimal, clearly marked,
  and made optional so radixark runs without orbit.
- **Delivery:** curated commit series + a seam manifest.
- **Drop:** the 3 dead vendored `ref_*.py` files (~4,474 lines, imported by
  nothing) and `orbit/conversion/nccl_byte_view.py` (dead). Drop the 4
  `docs/reports/*.html|md` status reports.

## 7. Conventions

The user has **explicitly waived** `AGENTS.md` and all Megatron-Bridge repo
process constraints for this migration: no DCO sign-off, no mandated
commit-title taxonomy, no copyright-header requirement, no "ask first before
touching pyproject" gate, no CI/label process.

What we still keep, because it helps us rather than because a policy says so:

- Commit titles loosely follow radixark's `[{area}] {type}: {description}`
  shape, so `git log` stays readable next to radixark's own history.
- Do not edit `3rdparty/Megatron-LM/` — not policy, it is just a submodule and
  edits there would be lost.
- Keep orbit inside its own namespaces. Not style: this is the entire point of
  the migration (see §1).

## 8. Open verification gap

The `3rdparty/Megatron-LM` submodule is **not checked out** in either repo
(0 files), so orbit's ~29 `megatron.core` dependencies were never verified
against radixark's pin `5c7968af`. A blobless bare clone was started at
`/tmp/mlm.git`. Highest-risk symbols to check first:

- `megatron.core.quantization.quant_config` (whole module — newest, most likely absent)
- `megatron.core.tensor_parallel.layers._initialize_affine_weight_cpu` (monkeypatch target)
- `megatron.core.extensions.transformer_engine._initialize_affine_weight_cpu` (monkeypatch target)
- `megatron.core.extensions.transformer_engine._get_fp8_autocast_for_quant_params`
- `megatron.core.transformer.torch_norm`, `transformer.moe.router`,
  `process_groups_config`, `dist_checkpointing.strategies`

Nothing has been executed. No Python has been run. All findings are static.
