# Orbit upstream seams

Orbit is designed to be extractable: it lives entirely in its own namespaces
(`src/megatron/bridge/orbit/`, `scripts/orbit/`, `tests/unit_tests/orbit/`,
`docs/orbit/`) and touches radixark code in **four places across three files**.

Every seam is marked in-source with a comment containing `orbit-seam(<tag>)`.
Find them all with:

```
grep -rn "orbit-seam" src/megatron/bridge/
```

If that command returns hunks not listed here, the manifest is stale — fix it.

Design rules for a seam:

1. It must be small enough to re-apply by hand after an upstream re-fetch.
2. It must import orbit **lazily, inside the function** — never at module top
   level, so import order stays unaffected.
3. It must **not** be wrapped in `try/except ImportError`. Extraction reverts the
   whole hunk, so a seam never runs with orbit deleted; a guard would only mask
   real failures. Import orbit directly and let errors surface.
4. It must carry an `orbit-seam(<tag>)` comment naming the tag below.

---

## Seam 1 — `orbit-seam(hook-order)`

**File:** `src/megatron/bridge/training/setup.py`
**Type:** pure reorder. No orbit import. Zero added dependency.

**Why:** the ModelOpt pre-wrap hook must *register* — and therefore run —
before the PEFT pre-wrap hook, so quantizer submodules exist on the model
before PEFT loads a quantized pretrained checkpoint. Otherwise the sharded
state dict lacks the per-layer quantizer keys the checkpoint provides and load
validation raises `KeyError`.

**Anchor:** in `setup()`, just after
`timers("model-and-optimizer-setup", log_level=0).start(barrier=True)`.

**Action:** move this block —

```python
    # Register PEFT pre-wrap hook if PEFT is configured
    if cfg.peft is not None:
        peft_hook = _create_peft_pre_wrap_hook(cfg, state)
        _register_pre_wrap_hook(cfg.model, peft_hook)
        print_rank_0("Registered PEFT pre-wrap hook")
```

— to *after* the `if getattr(cfg.model, "restore_modelopt_state", False):`
block ends (i.e. after its `_register_pre_wrap_hook(cfg.model,
modelopt_pre_wrap_hook)`), and before `start_memory_history_recording(...)`.
Leave an `orbit-seam(hook-order)` comment where the block used to be.

**radixark adaptation:** keep radixark's `_register_pre_wrap_hook(cfg.model,
hook)` signature. Do **not** port spherelab's newer
`_register_setup_pre_wrap_hook(..., setup_hook_name="peft")` — that API does not
exist on radixark.

**If orbit is removed:** revert the ordering. Nothing else to do.

---

## Seam 2 — `orbit-seam(modelopt)` — ModelOpt save with async strategy

**File:** `src/megatron/bridge/training/checkpointing.py`, in `save_checkpoint()`

**Why:** honour an explicitly configured non-`nvrx` async strategy when saving
sharded ModelOpt state. The default path must stay on the module-level
`save_sharded_modelopt_state` symbol so existing tests can patch it.

**Anchor:**

```python
            # [ModelOpt]: save sharded modelopt_state (skip if model is empty, ...)
            if model:
                # cfg.dist can be None during checkpoint conversion (save_megatron_model)
                if not (cfg.dist and cfg.dist.use_decentralized_pg):
                    save_sharded_modelopt_state(model, checkpoint_name, (ckpt_cfg.ckpt_format, 1))
```

**Action:** branch the innermost call — non-default strategies go through
`orbit.training.modelopt_checkpoint._save_sharded_modelopt_state_with_async_strategy`,
the default stays on `save_sharded_modelopt_state`.

**radixark adaptation:** read the field directly as `ckpt_cfg.async_strategy`.
radixark declares `async_strategy: str = "nvrx"` on `CheckpointConfig`
(`training/config.py:678`), so it always exists and can never be `None` — no
`getattr` default, no `is not None` check. (An earlier version of this note claimed
radixark treats the field as possibly-absent. That was wrong: the `_save_params`
introspection a few dozen lines above concerns mcore's `save()` signature, not the
config field.)

**If orbit is removed:** revert this hunk, which restores the single
`save_sharded_modelopt_state` call. Do not guard the import.

---

## Seam 3 — `orbit-seam(modelopt)` — restore ModelOpt state before sharded load

**File:** `src/megatron/bridge/training/checkpointing.py`, in
`_load_checkpoint_from_path()`

**Why:** ModelOpt state must be restored *before* the sharded-load schema is
built, so quantizer keys exist for direct-load checkpoints.

**Anchor:** immediately after the rerun-state block —

```python
            if not tp_pp_match:
                print_rank_0("{}: Rerun state will be ignored".format(mismatch_msg))
```

— and immediately **before**:

```python
        sharded_sd_metadata["dp_cp_group"] = pg_collection.dp_cp
```

**Action:** call
`orbit.training.modelopt_checkpoint._maybe_restore_modelopt_state_for_sharded_load(model, checkpoint_name, state_dict)`.

**radixark adaptation:** spherelab's tree has an `if sharded_sd_metadata is
None:` guard between those two statements; radixark does that normalisation
earlier and has no guard here. Insert on the `dp_cp_group` anchor, not on the
`None` guard.

**If orbit is removed:** this is on the **main checkpoint-resume path** and the
import is unconditional, so revert this hunk — do not guard it.

---

## Seam 4 — `orbit-seam(modelopt)` — compress packed checkpoints on restore

**File:** `src/megatron/bridge/training/post_training/checkpointing.py`, at the
end of `load_modelopt_state()`

**Why:** compress packed low-precision checkpoints after ModelOpt state is
restored.

**Anchor:** the last line of the function —

```python
    modelopt_checkpoint_path = _get_modelopt_checkpoint_path(checkpoint_path)
    unwrapped_model = unwrap_model(model)
    restore_sharded_modelopt_state(unwrapped_model, modelopt_checkpoint_path)
```

**Action:** append a call to
`orbit.training.modelopt_packed_restore._maybe_compress_restored_modelopt_model(unwrapped_model, modelopt_checkpoint_path)`.

**radixark adaptation:** radixark's `load_modelopt_state(model, checkpoint_path)`
and `_get_modelopt_checkpoint_path(checkpoint_path)` take no `ckpt_step`
argument. Orbit already calls the latter with a single positional arg, so no
change is needed — just do not introduce `ckpt_step`.

**If orbit is removed:** the import is unconditional, so revert this hunk — do
not guard it.

---

## Non-seam couplings to remember

These are not upstream edits, but they are places orbit reaches into radixark
internals and will break silently on a version bump:

- `orbit/low_precision/common.py` — `patch_meta_init_for_te_modules()`
  monkeypatches `_initialize_affine_weight_cpu` in both
  `megatron.core.tensor_parallel.layers` and
  `megatron.core.extensions.transformer_engine`.
- `orbit/peft_ext/peft_mixin.py` — uses `PEFT._walk_model`, `params_to_save`.
- `orbit/peft_ext/int4_lora.py` — `type(out) is LoRALinear` exact-type check.
- `orbit/conversion/{fp8_preserve,oft_export}.py` — use
  `MegatronModelBridge._with_progress_tracking`, `_unwrap_name`.
- `orbit/conversion/oft_export.py` — uses
  `model_bridge._megatron_local_name_to_global`.
- `orbit/oft/te_oft/te_oft_layernorm_linear.py` — vendored copy of TE's
  `LayerNormLinear`; tied to a specific TransformerEngine build.
