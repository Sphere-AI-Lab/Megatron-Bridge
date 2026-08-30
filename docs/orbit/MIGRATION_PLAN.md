# Orbit → radixark migration plan

Branch: `feature/orbit-on-radixark` (off `radixark/bridge` @ `bb61fcd0`)
Repo: `/home/kerryliu/Megatron-Bridge-Spherelab`

Read `MIGRATION_MEMORY.md` first — it holds the verified facts this plan rests
on. This file is only the execution order.

Guiding rule: **radixark is upstream.** When orbit and radixark disagree, orbit
changes. Never edit a radixark file to suit orbit unless it is one of the three
declared seams.

## Phase 0 — close the verification gap (no commits)

Blocking. Do this before writing any code.

1. Check out mcore at radixark's pin:
   `git submodule update --init 3rdparty/Megatron-LM` (yields `5c7968af`).
2. Verify the high-risk `megatron.core` surface listed in
   `MIGRATION_MEMORY.md` §8. Static grep only, no execution.
3. Check out TransformerEngine at `f031cf87` and diff its
   `pytorch/module/layernorm_linear.py` against
   `orbit/oft/te_oft/te_oft_layernorm_linear.py` (which is a copy of TE
   2.13/2.14). This scopes commit C5.
4. Record results back into `MIGRATION_MEMORY.md` §8.

If `megatron.core.quantization.quant_config` is absent at `5c7968af`, stop and
decide: vendor a shim into `orbit/quant/`, or gate the NVFP4 path. Do not bump
the submodule pin — that contradicts the mcore decision.

## Phase 1 — the commit series

Ten commits, dependency-ordered so each one is reviewable in isolation. Commit
titles loosely follow radixark's `[{area}] {type}: {description}` shape for
readability only — the repo's process rules (sign-off, labels, headers) are
waived for this work.

| # | Title | Contents |
|---|---|---|
| C1 | `[misc] chore: add orbit namespace, NOTICE, and migration docs` | `orbit/__init__.py` + the 9 subpackage `__init__.py`; `NOTICE`; `docs/orbit/{MIGRATION_MEMORY,MIGRATION_PLAN,UPSTREAM_SEAMS}.md` |
| C2 | `[quant] feat: add orbit low-precision core and quant utilities` | `orbit/quant/**`, `orbit/low_precision/**`. Includes the **vendored** `dequantize_int4` / `quantize_to_int4` (blocker #2) |
| C3 | `[peft] feat: add orbit PEFT extensions for quantized bases` | `orbit/peft_ext/**`, with the `_base_returns_tuple` branches removed (blocker #1) |
| C4 | `[peft] feat: add orbit OFT method, layers, and Triton kernels` | `orbit/oft/**` except `te_oft/`; **omit** the 3 dead `ref_*.py` |
| C5 | `[peft] feat: adapt orbit TE LayerNormLinear OFT path to TE f031cf87` | `orbit/oft/te_oft/`; scoped by Phase 0 step 3; replace the `exit(1)` ImportError handler with a raise |
| C6 | `[ckpt] feat: add orbit conversion mixins and quantized model bridges` | `orbit/conversion/**` (**omit** `nccl_byte_view.py`), `orbit/model_bridges/**` |
| C7 | `[training] feat: add orbit ModelOpt checkpoint helpers and PEFT reports` | `orbit/training/**` |
| C8 | `[training, ckpt] feat: add four optional orbit seams` | The only radixark files touched — 4 hunks across 3 files. See `UPSTREAM_SEAMS.md` |
| C9 | `[recipe] feat: add orbit finetune entrypoints and conversion scripts` | `scripts/orbit/**`, with the `default_peft_config` import fixed (blocker #3) |
| C10 | `[test] test: add orbit unit tests` | `tests/unit_tests/orbit/**` |

Dropped deliberately (user decision): 3 dead `ref_*.py` (~4,474 lines),
`orbit/conversion/nccl_byte_view.py`, and the 4 `docs/reports/` status reports.

### Mechanics

Do **not** cherry-pick the 41 spherelab commits — they are fix-on-fix and touch
files that no longer match. Instead copy file trees from the source branch:

```
git checkout feature/generic-int4-adapter -- <paths for this commit>
```

then apply the blocker fixes, stage, and commit. `git diff --stat
radixark/bridge..HEAD` at the end must show only orbit paths plus the three
seam files.

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
