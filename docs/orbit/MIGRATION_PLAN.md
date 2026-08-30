# Orbit → radixark migration plan

Branch: `feature/orbit-on-radixark` (off `radixark/bridge` @ `bb61fcd0`)
Repo: `/home/kerryliu/Megatron-Bridge-Spherelab`

Read `MIGRATION_MEMORY.md` first — it holds the verified facts this plan rests
on. This file is only the execution order.

Two guiding rules:

1. **Orbit lives in its own namespace.** Never edit a radixark file to suit
   orbit unless it is one of the four declared seam hunks (3 files). Because
   orbit code is namespaced, textual conflicts with radixark are rare by
   construction — this is not a conflict-resolution rule, it is a containment
   rule.
2. **Prefer radixark's implementation.** Where radixark already provides a
   capability, orbit imports it rather than carrying its own copy. Do not
   vendor, fork, or reimplement something radixark already has. When orbit and
   radixark genuinely differ on a shared extension point (e.g. an attribute
   orbit expects on `AdapterWrapper`), orbit adapts to radixark's shape.

## Phase 0 — verification gap — DONE

Verified against the user's authoritative standalone checkouts, **not** the
submodule pins: `/home/kerryliu/Megatron-LM-RadixArk` @ `235952df` and
`/home/kerryliu/TransformerEngine` @ `f031cf87` (= tag v2.14).

Both came back clean — full results in `MIGRATION_MEMORY.md` §8:

- mcore: all 32 modules and all 56 symbol pairs orbit imports resolve,
  including both `_initialize_affine_weight_cpu` monkeypatch targets.
- The `quant_config` contingency did **not** fire —
  `megatron/core/quantization/quant_config.py` is present, so no shim and no
  NVFP4 gating is needed.
- TE: all 39 internal imports resolve, and orbit's vendored
  `te_oft_layernorm_linear.py` is field-for-field identical to v2.14's
  `_LayerNormLinear` (41-field `non_tensor_args` tuple matches at both the
  pack and unpack sites). **C5 needs no adaptation.**

## Progress

Landed on `feature/orbit-on-radixark` (newest last):

| Commit | What |
|---|---|
| `d40a1c4e` | docs: plan, memory, seam manifest |
| `505641ca` | docs: Phase 0 verification results |
| `e170c021` | **C1** — orbit namespace root + NOTICE |
| `8e904156` | **C2** — orbit/quant + orbit/low_precision |
| `89362b78` | docs: correct the migration rule to containment plus dedup |
| `f30b856c` | **C3** — orbit/peft_ext (+ 2 oft leaf files) |
| `3e17ba98` | fix: make the C2 modules lint-clean |

Next: **C4**. Remaining: C4–C10, then Phase 2.

Deviations from the commit table below, all recorded rather than silent:

1. Subpackage `__init__.py` files ship with their own content commit rather than
   all landing in C1, because several of them eagerly import their submodules
   and would otherwise leave C1 un-importable.
2. `orbit/oft/__init__.py` and `orbit/oft/param_names.py` shipped in **C3**, not
   C4. `peft_ext/peft_mixin.py` and `peft_ext/recompute_ext.py` both import the
   param-name predicates at module level, so C3 would not import without them.
   Both are leaves — `param_names.py` has no imports at all and the `__init__`
   is docstring-only — so this is the same importability rule as deviation 1
   applied across a subpackage boundary. C4 adds the rest of `orbit/oft/`.
   Note `param_names.py`'s only consumers are the two `peft_ext` modules,
   despite it living under `oft/`; it was left in place rather than moved, to
   avoid gratuitous layout drift from the source branch.
3. A separate lint commit (`3e17ba98`) follows C3 because the C2 files carried
   pre-existing ruff violations inherited from spherelab. `ruff.toml` is
   byte-identical between the two bases, so this was never a migration
   artifact — spherelab just never linted those paths. Kept out of C3 so the
   content commit stays reviewable.

**Blocker #2 is retracted.** radixark already defines `dequantize_int4` and
`quantize_to_int4` in `models/kimi_vl/utils.py`, functionally identical to the
upstream versions (only comments differ, plus a lint-only `del` of two unused
params). Upstream later moved these into `models/conversion/quantization_utils.py`
and left `kimi_vl/utils.py` as a re-export shim, so
`megatron.bridge.models.kimi_vl.utils` is a **stable import path on both
bases**. Orbit imports from there. Nothing is vendored — an earlier attempt to
vendor a copy was reverted under guiding rule 2.

## Phase 1 — the commit series

Ten commits, dependency-ordered so each one is reviewable in isolation. Commit
titles loosely follow radixark's `[{area}] {type}: {description}` shape for
readability only — the repo's process rules (sign-off, labels, headers) are
waived for this work.

| # | Title | Contents |
|---|---|---|
| C1 | `[misc] chore: add orbit namespace, NOTICE, and migration docs` | `orbit/__init__.py` + the 9 subpackage `__init__.py`; `NOTICE`; `docs/orbit/{MIGRATION_MEMORY,MIGRATION_PLAN,UPSTREAM_SEAMS}.md` |
| C2 | `[quant] feat: add orbit low-precision core and quant utilities` | `orbit/quant/**`, `orbit/low_precision/**`. Nothing vendored — blocker #2 retracted, see Progress |
| C3 | `[peft] feat: add orbit PEFT extensions for quantized bases` | `orbit/peft_ext/**`, with the `_base_returns_tuple` branches removed (blocker #1) |
| C4 | `[peft] feat: add orbit OFT method, layers, and Triton kernels` | `orbit/oft/**` except `te_oft/`; **omit** the 3 dead `ref_*.py` |
| C5 | `[peft] feat: add orbit TE LayerNormLinear OFT path` | `orbit/oft/te_oft/`; already matches TE v2.14 per Phase 0, so a straight copy — only change is replacing the `exit(1)` ImportError handler with a raise |
| C6 | `[ckpt] feat: add orbit conversion mixins and quantized model bridges` | `orbit/conversion/**` (**omit** `nccl_byte_view.py`), `orbit/model_bridges/**`. Repoint 2 int4-helper imports at `megatron.bridge.models.kimi_vl.utils`: `conversion/compressed_tensors_int4.py:152` (+ docstring ref :36) and `model_bridges/deepseek_v3_int4_bridge.py:41` |
| C7 | `[training] feat: add orbit ModelOpt checkpoint helpers and PEFT reports` | `orbit/training/**` |
| C8 | `[training, ckpt] feat: add four optional orbit seams` | The only radixark files touched — 4 hunks across 3 files. See `UPSTREAM_SEAMS.md` |
| C9 | `[recipe] feat: add orbit finetune entrypoints and conversion scripts` | `scripts/orbit/**`, with the `default_peft_config` import fixed (blocker #3) |
| C10 | `[test] test: add orbit unit tests` | `tests/unit_tests/orbit/**`. Repoint 3 int4-helper imports at `megatron.bridge.models.kimi_vl.utils`: `test_compressed_tensors_int4.py:6`, `test_int4_requantize.py:6`, `test_quant_adapters.py:108` |

Dropped deliberately (user decision): 3 dead `ref_*.py` (~4,474 lines),
`orbit/conversion/nccl_byte_view.py`, and the 4 `docs/reports/` status reports.

### Mechanics

Do **not** cherry-pick the 41 spherelab commits — they are fix-on-fix and touch
files that no longer match. Instead copy file trees from the source branch:

```
git checkout feature/generic-int4-adapter -- <paths for this commit>
```

then apply the blocker fixes, stage, and commit. `git diff --stat
radixark/bridge..HEAD` at the end must show only orbit paths plus the 3 seam
files (4 hunks).

**Dedup check — do this for every commit, before committing.** Guiding rule 2
means a commit is not just a copy. For each file being added, check whether
radixark already provides the same capability, and if so import it instead:

```
# does radixark define these symbols already?
git grep -n "def <symbol>" radixark/bridge -- src
# for a same-named file, compare the two sides:
git show radixark/bridge:<path>  vs  git show feature/generic-int4-adapter:<path>
```

Blocker #2 is the precedent and shows why this is not optional: orbit appeared
to need an upstream module radixark lacks, and the first instinct was to vendor
it. In fact radixark already had both functions under a different path, and
spherelab's own tree had quietly turned that path into a re-export shim. A
vendored duplicate would have been dead weight that drifts on every future
re-fetch. Prefer the import path that resolves on *both* bases.

## Phase 2 — extractability gates

The migration is not done until all four hold:

1. `git diff --name-status radixark/bridge..feature/orbit-on-radixark` lists
   **only** orbit-namespaced paths plus exactly 3 radixark files.
2. Every hunk in those 3 files carries an `orbit-seam(<tag>)` comment and is
   listed in `UPSTREAM_SEAMS.md`.
3. Deleting `src/megatron/bridge/orbit/`, `scripts/orbit/`,
   `tests/unit_tests/orbit/` and reverting the 4 seams returns the tree to
   byte-identical `radixark/bridge`. Verify with
   `git diff radixark/bridge -- src/megatron/bridge/{training,peft,models}`.
4. `pyproject.toml` is untouched. Not a process rule — an untouched
   `pyproject.toml` is what lets orbit copy cleanly onto a future radixark
   without dependency conflicts. Orbit currently adds no dependencies at all
   (spherelab's diff against its own base is empty here), so if one turns out
   to be required, flag it rather than sneaking it in.

## Phase 3 — verification (user, on GPU)

Run against `/home/kerryliu/Megatron-LM-RadixArk` @ `235952df` and
`/home/kerryliu/TransformerEngine` @ `f031cf87`, not the submodule pins.

Nothing in Phases 0–2 executes Python. Hand off with this order, cheapest first:

1. `uv run ruff check .` and `uv run ruff format --check .`
2. Import smoke test: `import megatron.bridge.orbit` and each subpackage.
3. `uv run python -m pytest tests/unit_tests/orbit/`
4. Delete-orbit test: confirm training/resume still imports with orbit absent
   (this is what blocker #5 protects).
5. One INT4 conversion via `scripts/orbit/conversion/convert_int4_checkpoint_direct.py`.
6. One short QOFT finetune via `scripts/orbit/finetune_peft.py`.

Blockers #4 (mcore monkeypatch), #6 (`sharded_state_dict` fallback) and #7
(`share_expert_adapters` default flip) can only be settled at step 4–6. They
are runtime concerns, flagged not fixed.

## Future upstream re-fetch (the actual point of all this)

```
git fetch radixark
git switch -c orbit-rebase-<date> radixark/bridge
git checkout feature/orbit-on-radixark -- src/megatron/bridge/orbit scripts/orbit tests/unit_tests/orbit docs/orbit NOTICE
# then re-apply the 4 seams by hand using UPSTREAM_SEAMS.md
```

Orbit files copy across wholesale because they are namespaced. Only the four
seams (3 files) need human judgement. That is the whole design.
