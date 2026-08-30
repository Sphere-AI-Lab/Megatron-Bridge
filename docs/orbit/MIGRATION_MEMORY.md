# Orbit → radixark migration: durable memory

Purpose: everything a fresh session needs to resume this migration without
re-deriving it. Facts here were verified by direct inspection, not assumed.
Read this together with `MIGRATION_PLAN.md` and `UPSTREAM_SEAMS.md`.

Last updated: 2026-08-30

## 1. Goal

Port the Sphere AI Lab "orbit" work (quantized-base PEFT + OFT: INT4 / FP8 /
NVFP4) onto the radixark fork of Megatron-Bridge.

Overriding constraint from the user: **orbit must stay extractable.** The
future workflow is "fetch upstream radixark, then re-apply our changes."

Two consequences, and it matters that they are distinct:

- **Containment.** Orbit code lives only in its own namespaces, so it rarely
  conflicts with radixark textually at all. The four seam hunks are the only
  upstream edits.
- **Deduplication.** Where radixark already provides a capability, orbit uses
  radixark's version instead of carrying its own. Do not vendor, fork, or
  reimplement something radixark already has — a duplicate is what actually
  breaks extractability later. Where orbit and radixark differ on a *shared*
  extension point, orbit adapts to radixark's shape rather than the reverse.

So this is mostly not a conflict-resolution exercise; it is a containment plus
dedup exercise. Blocker #2 below is a worked example of the dedup rule.

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

2. **~~`models/conversion/quantization_utils.py` absent in radixark.~~
   RETRACTED — not a blocker.**
   The module really is absent (added upstream in `39b79eb7` / PR #3778, after
   radixark's base), but orbit only needs `dequantize_int4` and
   `quantize_to_int4`, and **radixark already has both** at
   `models/kimi_vl/utils.py:19,89`. They are functionally identical to the
   upstream versions — the only diffs are comments and a lint-only
   `del weight_shape, group_size` of two params that neither version uses.
   Upstream later moved these into `quantization_utils.py` and left
   `kimi_vl/utils.py` as a re-export shim, so
   `megatron.bridge.models.kimi_vl.utils` resolves on **both** bases.
   **Fix:** import from `megatron.bridge.models.kimi_vl.utils`. Do not vendor a
   copy (an earlier attempt to do so was reverted), and do not add the upstream
   file to radixark.
   Note these are *not* interchangeable with orbit's own
   `low_precision.dequantize_int4`, which is a CUDA-only Triton kernel; the
   radixark/upstream one is pure PyTorch and runs on CPU. Five orbit call sites
   need the CPU version.

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
   extractability wart. Both patch targets were **verified present** in mcore
   `235952df` — see §8.

5. **Two seam imports are unguarded — RESOLVED, deliberately left unguarded.**
   `checkpointing.py` and `post_training/checkpointing.py` import orbit directly
   on the checkpoint save/resume paths. Guards were briefly added, then removed:
   extraction reverts each seam hunk *wholesale*, so a seam never executes with
   orbit absent, making the guard unreachable. All it could do was turn a real
   failure — a broken modelopt plugin, which `import_plugin` already downgrades to
   a warning — into a silent no-op on the main resume path. Seam design rule 3 in
   `UPSTREAM_SEAMS.md` now forbids the guard.

6. **`AdapterWrapper.sharded_state_dict`** in radixark lacks spherelab's
   `_plain_module_sharded_state_dict` fallback. Risk of dropping delta-only
   adapter weights. Needs a runtime check, not a static one.

7. **`LoRA` dataclass default flip — RESOLVED, keeping radixark's default.**
   radixark's `share_expert_adapters` defaults to `False` (spherelab `True`), and
   radixark has no `sequence_parallel_input_regather`. Orbit sets neither.

   Investigated: orbit's own OFT path is unaffected — `GroupedOFTRotation` is
   unconditionally per-expert and has no sharing option, so it never consults the
   flag. The only affected site is the `--peft lora` comparison baseline at
   `scripts/orbit/finetune_qoft.py:307`, which constructs `LoRA(**kwargs)` without
   the flag. radixark's `False` (per-expert) therefore *matches* orbit OFT's
   semantics, making that baseline more comparable than spherelab's `True`
   (shared) did. Decision: do not override; pass `share_expert_adapters=True`
   explicitly only if reproducing pre-migration LoRA parameter counts.

   Noted while checking: radixark's own `tests/unit_tests/peft/test_lora.py:207`
   and `test_canonical_lora.py:196` still assert the flag is `True`, so they fail
   against radixark's own default. Pre-existing upstream staleness, identical in
   both trees, left untouched — fixing it would add a 4th modified upstream file
   and break containment.

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

- **mcore:** use the user's checkout `/home/kerryliu/Megatron-LM-RadixArk`
  @ `235952df` (radixark's `miles-main`), **not** the `5c7968af` pyproject pin.
  Orbit must work against it — verified clean, §8.
- **TE:** use the user's checkout `/home/kerryliu/TransformerEngine`
  @ `f031cf87` (= tag v2.14), which matches radixark's pin. The user ruled out
  gating off the TE-OFT path, so `orbit/oft/te_oft/te_oft_layernorm_linear.py`
  had to work against f031cf87 — and §8 shows it already does, unchanged.
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

## 8. Dependency verification — CLOSED

The `3rdparty/Megatron-LM` submodule is not checked out in either repo, so
verification used the two standalone clones the user provided under `$HOME`.
**These are the authoritative versions** — use them, not the pyproject pins.

| Dep | Path | Commit | Notes |
| --- | --- | --- | --- |
| Megatron-Core | `/home/kerryliu/Megatron-LM-RadixArk` | `235952df` | branch `miles-main`, 2026-08-20, remote `github.com/radixark/Megatron-LM`. **1065 commits ahead** of the `5c7968af` pyproject pin (which is an ancestor). Spherelab's pin `731b7914` is not in this clone. |
| TransformerEngine | `/home/kerryliu/TransformerEngine` | `f031cf87` | **= tag `v2.14`**, 2026-04-06. Matches radixark's pin. |

### mcore result: clean

- All 5 high-risk symbols present:
  `megatron/core/quantization/quant_config.py` exists;
  `_initialize_affine_weight_cpu` at `tensor_parallel/layers.py:157` and
  imported at `extensions/transformer_engine.py:43`;
  `_get_fp8_autocast_for_quant_params` at `extensions/transformer_engine.py:355`;
  `_get_custom_recipe` at `fp8_utils.py:330`.
- All 32 `megatron.core` modules orbit imports resolve.
- All 56 module/symbol pairs resolve. The 3 initially flagged were
  submodule-import false positives (`from megatron.core import
  dist_checkpointing`, `...strategies import torch as torch_dist_strategy`,
  `from megatron.core.distributed import distributed_data_parallel`) — all
  confirmed present as a package dir or module file.

Risk is far lower than originally feared: this mcore is contemporary with
spherelab's, not the 4-month-old pin. Blocker #4 (the monkeypatch) is
therefore safe as written, though it remains the one extractability wart.

### TE result: clean, and C5 is a no-op

- All 39 TE internal imports in `orbit/oft/te_oft/te_oft_layernorm_linear.py`
  resolve against v2.14. (`cpp_extensions.general_gemm` lives in
  `cpp_extensions/gemm.py` and is star-re-exported via its `__all__`.)
- `_OFTLayerNormLinear.forward` signature matches TE v2.14's
  `_LayerNormLinear.forward` exactly, including the `non_tensor_args: Tuple`
  packing style, plus the one OFT addition (`adapter_fn`).
- The 41-field `non_tensor_args` tuple is **field-for-field identical** at both
  the unpack site (orbit:164 / TE:98) and the pack site (orbit:672 / TE:1546).
  Only difference is `self.` → `mod.`, because orbit's is a free function
  taking the module.

**Conclusion:** the vendored file's header ("updated for TE 2.14.0") is
accurate. Commit C5 needs no adaptation work — copy the file as-is.

Everything above is static analysis. No GPU verification has been done (that is
Phase 3, handed to the user).

**Import verification is now complete and clean:** all **604 imported names
across all 83 orbit Python files** resolve against radixark, mcore `235952df`,
TE v2.14 and modelopt 0.46.0, with **no remaining false positives**. Reaching
that required fixing three blind spots in the AST checker, each of which had
produced spurious "missing symbol" reports:

1. tuple-unpacking assignment (`X, HAVE_X = safe_import_from(...)`) — this had
   wrongly flagged 4 symbols in radixark's own `peft/utils.py`;
2. **relative** star imports (`from .gemm import *`) — this had wrongly flagged
   TE's `general_gemm` and modelopt's `SequentialQuantizer`;
3. star imports nested inside `with` blocks (modelopt's
   `with import_plugin(...): from .mcore_dist_checkpointing import *`).

## 9. Findings from executing Phase 1 (C3–C10)

### Orbit is QLoRA-style PEFT, not QAT

Confirmed in code, and it reframes the risk ranking:

- INT4 base lands as **buffers, not Parameters**
  (`model_bridges/deepseek_v3_int4_bridge.py:177-179` registers
  `_packed`/`_scale`/`_shape` with `persistent=True`), so the base cannot take
  gradients and never materializes a bf16 master copy.
- Dequant is transient per forward; `w_compute` is `del`'d in a `finally`.
- The memory win is the `saved_tensors_hooks(pack, unpack)` in
  `peft_ext/int4_lora_forward.py:79-100`: autograd saves the *packed* triplet
  and re-dequantizes in `unpack` during backward, so no bf16 base copy is held
  across the fwd/bwd boundary.
- No base-weight fake-quant anywhere. The one STE,
  `_fp8_activation_qdq_per_token_group_ste`, applies to *activations* in the OFT
  FP8 path.

**Consequence: adapter-only saving is the highest-consequence unknown.** If the
base is frozen buffers, the adapter *is* 100% of the training result, so any
save path that drops adapter weights yields a run that reports success and
writes nothing useful.

But the specific mechanism blocker #6 names does **not** apply to orbit, and an
earlier draft of this section overstated it. radixark's
`AdapterWrapper.sharded_state_dict` (`peft/adapter_wrapper.py:192-222`) calls
`self.to_wrap.sharded_state_dict(...)` and `self.adapter.sharded_state_dict(...)`
unconditionally, where spherelab had a `_plain_module_sharded_state_dict`
fallback — but every orbit `AdapterWrapper` path supplies the method:

- orbit never uses `LinearAdapter` or `patch_linear_module` (grep: no hits);
- `OFTRotationModule` defines its own `sharded_state_dict`
  (`oft/oft_layers.py:616`) despite subclassing plain `nn.Module`;
- `Int4LoRA` reuses upstream's `ParallelLinearAdapter`, which has it;
- `to_wrap` is always a Megatron parallel linear, which has it.

**The real exposure is narrower and elsewhere.** `CanonicalOFT.transform` can
return `OFTLinearSplitQKV`, `OFTLinearSplitFC1UpGate`, or
`OFTLinearGroupedSplitFC1UpGate` (`canonical_oft.py:1507-1538`). These are plain
`nn.Module` *replacements* for the target module, not `AdapterWrapper`
subclasses, so radixark's method never runs for them — and `canonical_oft.py`
originally defined no `sharded_state_dict` at all.

**Settled statically, and fixed.** Nothing was dropped: mcore's
`sharded_state_dict_default` takes its plain-`nn.Module` branch and calls
`module.state_dict(prefix='', keep_vars=True)` for the *whole subtree*, so the
`oft_r` parameters do reach the checkpoint. But it passes an empty
`tensor_parallel_layers_axis_map`, so every tensor is marked replicated — and
because the subtree is snapshotted in one shot, descendant `sharded_state_dict`
methods are never consulted. That silently discarded `to_wrap`'s own TP sharding
and `OFTRotationModule`'s `oft_r` axis map and expert `replica_id` fixups.

Fixed with a `_split_wrapper_sharded_state_dict` helper plus a thin
`sharded_state_dict` on each of the three wrappers that delegates per child,
passing `child.tp_group` straight through.

**The fix does not cover the grouped MoE path.** It fully lands only for
`OFTLinearSplitQKV` and `OFTLinearSplitFC1UpGate`, whose adapter children are
`OFTRotationModule`, which *does* define `sharded_state_dict`
(`oft/oft_layers.py:616`). `OFTLinearGroupedSplitFC1UpGate` builds
`GroupedOFTRotation` instead (`canonical_oft.py:687`), and that class has **no**
`sharded_state_dict` while owning an `nn.Parameter` (line 515) plus two buffers
— so delegating merely pushes the problem one level down and `oft_r` is still
emitted **replicated** for grouped MoE. `to_wrap`'s own TP sharding is now
honoured in all three, which was the larger loss.

Closing the MoE gap means deciding the correct axis map for a
`(num_local_experts, num_blocks, n_elements)` parameter under EP/ETP. That was
deliberately not invented — it is an open runtime task, not a static one.

Key names are unchanged, but sharding metadata is not: `oft_r` becomes TP-sharded
on axis 0 where it was previously replicated, so pre-fix and post-fix checkpoints
are **not interchangeable at TP>1**. At TP=1 nothing changes. This was
pre-existing spherelab behaviour, not a migration regression, and the fix is
**unverified** — TP>1 is not reproducible on a CPU-only box.

### Fallback audit — CLOSED

The migration authored exactly **two** fallbacks, both since removed:

- the four-level `tp_group` chain in `_split_wrapper_sharded_state_dict` (three
  levels were unreachable; now reads `child.tp_group` directly);
- `getattr(ckpt_cfg, "async_strategy", None)` in seam 2 (radixark declares the
  field as `str = "nvrx"`, so it always exists).

Every other defensive construct in orbit is **inherited verbatim** from
spherelab, verified against the `feature/generic-int4-adapter` blobs: 5
`except ImportError` guards for optional Triton/TE deps, and 16 broad
`except Exception` handlers.

The riskiest inherited one is `peft_ext/recompute_ext.py:71`, a blanket
`except Exception` that downgrades a *correctness* fix to a printed warning — if
it fires, adapter-only training with recompute silently produces zero gradients.
It mirrors upstream `peft/recompute.py:121` exactly, so it was left alone to keep
the port diffable against upstream. Worth a runtime assertion instead, if that
path ever misbehaves.

**Standing rule from the user:** do not add fallbacks. If one seems necessary,
surface it for review first.

### Qwen dense/MoE OFT+LoRA path review — findings

Traced `CanonicalOFT` (the default) end to end for Qwen3 dense and Qwen3-30B-A3B.

**Dense path looks correct.** Verified statically:

- The split-QKV GQA arithmetic (`canonical_oft.py:1182`) is right. Qwen3-30B-A3B
  (32 heads / 4 KV / head_dim 128) at TP=4 → `packed_units=10`,
  `units_per_group=10` → one local group of 8Q+K+V = 1280 rows. Qwen3-8B
  (32/8/128) checks out at TP=1/2/4/8.
- `_split_qkv_weight` and `_interleave_qkv` are exact inverses over Megatron's
  per-query-group interleaved layout.
- `to_wrap.bias` is already in that interleaved order, so `out + bias`
  (`canonical_oft.py:1319`) lines up; `skip_bias_add` defers correctly.
- `oft_r` is zero-init in both rotation classes → `R = I` at step 0, so the
  adapted model starts equal to the base. Confirmed on GPU: `Cayley(0) == I` exactly.

**MoE findings — all inherited. #1 is a design limitation; #2 and #3 are genuine
checkpoint concerns, left unfixed because they need MoE-capable hardware:**

1. **fc1/fc2 per-expert asymmetry — a known limitation, NOT a bug.** The grouped
   branch exists only for `linear_fc1`; `experts.linear_fc2` falls through to plain
   `OFTLinear`, whose forward applies one rotation to the whole concatenated token
   stream (`oft_layers.py:894`). So fc1 gets per-expert rotations and fc2 gets a
   single shared one.

   Git archaeology settles the intent: commit `85c84cbc` (the initial Sphere
   commit) introduced `OFTLinearGroupedSplitFC1UpGate` *and* the
   "expert linears ... use plain `OFTLinear` with a single rotation" docstring
   **together**. There was no partial migration — this was deliberate scope. The
   plausible reason: fc1's `oft_r` is replicated, whereas fc2 is RowParallel so its
   blocks are TP-sharded, making a per-expert 3D `oft_r` there materially harder.

   An earlier draft of this section called it unfinished work. That was wrong, and
   the supporting reasoning was weak: the loose docstring is consistent with either
   reading; `__getitem__` / `_PerExpertOFTRotationView` / `sgemm_oft_r_by_expert`
   are required by the fc1 grouped path itself, so their existence proves nothing
   about fc2; and the `CanonicalLoRA` analogy is imperfect because LoRA adds
   `ΔW = BA` to the *weight* (per-expert is the only sensible reading) while OFT
   rotates the *input* (`y = W(Rx)`).

   What remains true: it is undocumented as a limitation and conflicts with the
   "canonical = one OFT per matrix" principle. Treat it as a design question for
   the OFT owner, not a defect.

2. **Grouped fc1 adapters checkpoint as replicated.** `GroupedOFTRotation` has no
   `sharded_state_dict` yet owns the 3D `oft_r` `(E, num_blocks, n_elements)`
   (`canonical_oft.py:500`), so per-expert rotations are written with no EP/TP
   sharding metadata. Same root cause as blocker #6.

3. **The expert `replica_id` rewrite is suspect** (`oft_layers.py:650-660`,
   carrying the original authors' own TODO). It writes an EP-derived value into
   `replica_id`'s *TP-rank* slot, and `(ep+1)*(edp+1)-1` is not injective —
   `(ep=1,edp=0)` and `(ep=0,edp=1)` both give 1, so two ranks claim the same
   replica while other indices are never emitted. Fires for MoE fc2 adapters.

**Latent traps (not Qwen-breaking):**

- `_should_treat_linear_fc1_as_unfused` (`canonical_oft.py:176`) only tests the
  `vision_model.` prefix. There is no `gated_linear_unit` precondition, so a
  non-gated model would have fc1 split in half as if it were gate/up. Harmless
  for Qwen3 (SwiGLU). No guard added — the case is unreachable for orbit targets.

**Changed in this pass:** the hardcoded `full_name.endswith(".mlp.experts.linear_fc1")`
was replaced with radixark's `is_grouped_expert_linear` (dedup rule). Proven
behaviour-preserving: `wildcard_match` compiles an anchored `^...$` regex, so
`matched_pattern.endswith("linear_fc1")` implies `full_name` ends with
`linear_fc1`, making the predicates' `linear_fc2` divergence unreachable. The
stale `CanonicalOFT` docstring was corrected to describe the real fc1/fc2 split.

### Seam notes

- **Seam 3 is a gap-fill, not an invention.** radixark already applies the same
  restore-before-schema-generation pattern at
  `_load_model_weights_from_checkpoint` (`checkpointing.py:1717`), but
  `_load_checkpoint_from_path` — the training resume path — has no modelopt
  restore at all. Orbit calls the same `restore_modelopt_state` radixark
  imports, adding only a `has_modelopt_state` guard.
- **Seam 2 is justified — RESOLVED, no longer an open question.** modelopt 0.46.0
  is now installed at `/home/kerryliu/uv-cu12/.venv`, and static inspection
  settles it: ModelOpt's own
  `save_sharded_modelopt_state(model, checkpoint_name, sharded_strategy=None, prefix="")`
  (`torch/opt/plugins/mcore_dist_checkpointing.py:112`) has **no**
  `async_strategy` parameter and calls
  `dist_checkpointing.save(modelopt_state, name, sharded_strategy)` with three
  positional args. mcore's `save` **does** accept
  `async_strategy: Optional[str] = "nvrx"` (`serialization.py:342`). So orbit's
  fork is the only way to thread the strategy through. Keep seam 2.
- **The seam guards can silently disable orbit.** `modelopt.torch.opt.plugins`
  exposes `restore_modelopt_state` / `save_sharded_modelopt_state` only via
  `with import_plugin("megatron core dist checkpointing"): from
  .mcore_dist_checkpointing import *`, and `import_plugin` swallows
  `ModuleNotFoundError` (and every other exception) with a warning only. If that
  plugin fails to load, those names vanish, orbit's top-level import at
  `training/modelopt_checkpoint.py:16` raises ImportError, and the seams'
  `except ImportError: pass` quietly no-ops — ModelOpt state would then not be
  restored before the sharded load, with no clear diagnostic.
  **This is not hypothetical:** the venv above cannot import modelopt at all
  because `requests` is missing, so as it stands seams 2 and 3 would silently do
  nothing. Install `requests` before Phase 3, and consider logging at debug
  level in those `except ImportError` handlers rather than a bare `pass`.

### Known, deliberately not fixed

- **Intra-orbit duplicate:** `_module_bias_enabled` and
  `_get_active_bias_tensor` exist in both `oft/oft_layers.py:302,323` and
  `peft_ext/int4_lora_forward.py:17,31`, logic-identical (verified by AST
  compare; only docstrings differ). Left alone: consolidating would either force
  `peft_ext` to import the triton-heavy `oft_layers`, or require a new shared
  leaf module and layout drift from the source branch. The dedup rule targets
  orbit-vs-radixark, and neither copy duplicates radixark.
- `conversion/model_metadata_compare.py` reads a checkpoint's `.metadata` with
  `pickle.load`. Standard for torch-dist checkpoints and pre-existing, but it
  does mean pointing the comparison tooling at an untrusted checkpoint dir is
  code execution.
- `scripts/orbit/finetune_qoft.py` reaches its siblings via
  `sys.path.insert(0, dirname(__file__))` then `from models._qoft_common import
  ...`, where `models` is the `scripts/orbit/models/` namespace package (no
  `__init__.py`). Fragile but functional, and only this one script does it.

### Lint

radixark's own `training/setup.py` carries 4 pre-existing ruff findings (1
unsorted import block, 3 unused imports) — verified identical at
`radixark/bridge` with the repo config, and left untouched so the seam files
stay trivially re-appliable. Orbit's own paths are `ruff check` and
`ruff format --check` clean, using file-level `noqa` rather than invented
docstrings on the 17 operational scripts and on the vendored TE file (whose
import list must stay diffable against TE v2.14).

