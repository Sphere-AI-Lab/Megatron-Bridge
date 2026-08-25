# Sphere fork layout: package + shims + seams

Contract of this fork (`Sphere-AI-Lab/Megatron-Bridge`, branch `orbit-main`):

- Everything wholly sphere-owned lives in `src/megatron/bridge/sphere/`:
  - `oft/` — OFT/CanonicalOFT PEFT methods, layers, param names, TE and triton kernels
  - `quant/` — fp8/int4/nvfp4 state-dict transforms, Qwen3 FP8 GEMM helpers
  - `low_precision/` — direct-load quantized checkpoint machinery (fp4/fp8/int4/nvfp4)
  - `model_bridges/` — DSV4 bridge and the quantized-checkpoint bridge variants
  - `conversion/` — OFT export mixin (`oft_export.py`), `QuantScaleMapping`,
    NCCL byte-view workaround, model-metadata compare
  - `training/` — ModelOpt packed-restore patch stack, async modelopt save /
    sharded-load restore, PEFT parameter reports
  - `peft_ext/` — bias-placeholder normalization, INT4 base forward for `LoRALinear`
- Launcher scripts live in `scripts/sphere/` (run from the repo root; the
  self-locating ones resolve the root via `BASH_SOURCE/../..`).
- **Compatibility shims** (22 module files + the `models/conversion/low_precision/`
  package `__init__`) remain at every old externally-imported path. Each shim
  aliases `sys.modules[old] = <sphere module>`, so old and new dotted paths
  yield the *same module object* — monkey-patches and `isinstance` checks keep
  working, and nothing double-imports. Orbit imports 10+ of these paths (some
  at module top-level); keep the shims until orbit migrates to
  `megatron.bridge.sphere.*` paths.
- Everything else that differs from upstream `NVIDIA-NeMo/Megatron-Bridge` is a
  deliberate **seam**, inventoried below. Seams inside otherwise-upstream
  functions carry a `# sphere-seam(<tag>):` marker.
- Review rule for upstream pulls and cross-fork ports:

  ```bash
  git diff <upstream-base> --diff-filter=M -- src pyproject.toml .gitmodules .gitignore README.md 3rdparty
  ```

  must show only the seams listed here (added files never conflict on merges).

Upstream base at the time of writing: `fad15ab2`. Totals: 32 modified files,
+671/−227 (before the restructure: 37 files, +2,125/−228).

## Seam inventory (modified upstream files)

| File | Delta | Kind | Purpose |
|---|---|---|---|
| `models/conversion/peft_bridge.py` | +123/−29 | mixin + rewiring | `MegatronPeftBridge(SphereOFTExportMixin)` base (methods live in `sphere/conversion/oft_export.py`); widened `_get_lora_unwrapped_name` / `_is_adapter_param_name` predicates; OFT-aware rewrites of `infer_target_modules_from_adapter_weights` / `build_adapter_config_dict` (+`_HF_OFT_SUFFIXES`) — upstream-unit-tested, LoRA-behavior-entangled, kept in place |
| `models/conversion/auto_bridge.py` | +73/−13 | behavior | `export_oft_adapter_weights` public method (orbit API); `save_hf_adapter` OFT-aware generator flow (all ranks drain; note: LoRA tensors now `.detach().cpu()` — dtype behavior change vs upstream) |
| `models/conversion/model_bridge.py` | +36/−8 | dispatch + guards | `@dispatch stream_oft_adapter_weights_megatron_to_hf` stub + per-bridge registration; `task is None` guards; expert-number regex robust to quant param names |
| `models/conversion/param_mapping.py` | +68/−10 | behavior | `uses_expert_tp_group` property + `is_expert` DSV4 widening (shared-expert ETP routing); NCCL byte-view wraps in broadcast/scatter/gather (impl in `sphere/conversion/nccl_byte_view.py`). Orbit monkey-patches `MegatronParamMapping.broadcast_obj_from_pp_rank` — do not move/rename the class |
| `models/hf_pretrained/state.py` | +23/−13 | behavior | `load_tensors` whole-shard cached reads (`vm.max_map_count` exhaustion). Orbit monkey-patches `SafeTensorsStateSource.load_tensors` — keep in place |
| `models/kimi_vl/utils.py` | +9/−123 | shim | INT4 helpers consolidated into `sphere/low_precision/int4.py`; file-path loaders depend on this shim |
| `models/qwen_vl/qwen35_vl_provider.py` | +90/−2 | compat | transformers==5.12.1 fallback config classes + `AutoConfig.register`; delete when the pin catches up |
| `models/mamba/mamba_provider.py` | +30/−2 | compat | MCore-pin fallback `parse_hybrid_pattern` / `MTP_SEPARATOR` |
| `models/model_provider.py` +11/−2, `models/common/unimodal.py` +15/−2 | behavior | meta-device init materialization incl. ModelOpt `QTensorWrapper` unwrap (orbit imports `to_empty_if_meta_device`) |
| `models/gpt_provider.py` | +3/−1 | behavior | modelopt layer spec priority when `restore_modelopt_state` |
| `models/qwen/qwen3_moe_bridge.py` | +12 | behavior | fp32 router dtype + `moe_layer_freq` derivation (inherited by sphere fp8/int4 bridges) |
| `models/kimi_vl/{kimi_k25_vl_bridge,modeling_kimi_k25_vl}.py` | +3/−1 | thin | int4 import retarget; `forward(**kwargs)` passthrough |
| `models/hf_pretrained/causal_lm.py` | +1 | thin | `_model_name_or_path` alias |
| `models/{,conversion/,deepseek/,kimi_vl/,qwen/}__init__.py` | +27/−4 | registration | sphere bridge imports/exports (registration timing), Bailing try/except, `low_precision` re-export |
| `peft/__init__.py` +7, `peft/base.py` +5/−1, `peft/recompute.py` +10/−2, `peft/walk_utils.py` +8, `peft/utils.py` +11/−3, `peft/lora_layers.py` +15/−2 | behavior/thin | OFT-aware param-name predicates; bias normalization + INT4 forward hooks into sphere/peft_ext; `AdapterAttributes.adapter_type` plumbing; DSV4 `is_expert_linear` widening; `map()` None-passthrough |
| `training/setup.py` | +24/−6 | behavior | **hook-registration reorder** (ModelOpt pre-wrap before PEFT pre-wrap — order is load-bearing, invisible to grep); PEFT stats prints + rank-0 report hook |
| `training/checkpointing.py` | +18/−1 | hooks | async modelopt save + sharded-load restore call sites (impl in `sphere/training/modelopt_checkpoint.py`) |
| `training/post_training/checkpointing.py` | +6 | hook | packed-layout compress on restore (impl in `sphere/training/modelopt_packed_restore.py`); orbit imports `has_modelopt_state`/`load_modelopt_state` from here |
| `training/model_load_save.py` +28, `training/utils/omegaconf_utils.py` +9/−1, `utils/yaml_utils.py` +6/−1 | behavior | GenerationConfig None-scrubbing; conversion-save CheckpointConfig knobs |
| `pyproject.toml` | +1/−1 | pin | `transformers==5.12.1` (upstream range `>=5.0.0,<=5.3.0`) — conflicts on every upstream merge by design |
| `.gitmodules` + `3rdparty/Megatron-LM` | +3/−2 | pin | submodule → `Sphere-AI-Lab/Megatron-LM@orbit-main` |
| `README.md` +8, `.gitignore` +5 | docs | fork banner; ignore entries |

## Known deferred items (not part of the restructure)

- `peft/utils.py:get_adapter_attributes_from_linear` uses `adapter_type is not "oft"`
  (string-identity; SyntaxWarning on CPython 3.12). The impossible-inc port fixed
  it to `!=`; fix here as its own reviewed change.
- `_maybe_compress_restored_modelopt_model` uses bare `print` (style rule says
  `print_rank_0`/logging).
- Duplicated helpers: `_module_bias_enabled`/`_get_active_bias_tensor` exist in
  `sphere/peft_ext/int4_lora_forward.py` and `sphere/oft/oft_layers.py`;
  `_compose_module_name` in `sphere/peft_ext/bias_normalization.py` and the DSV4
  bridge — dedupe inside the sphere package at leisure.
- `SphereOFTExportMixin._merge_oft_adapter_weights` / `_compute_oft_rotation_matrix`
  have zero callers in bridge and orbit (reserved code).
- Pre-existing `ruff format` nonconformance in many sphere files (predates the
  restructure; the restructure deliberately did not reformat).

## Pulling upstream / porting to another fork

- **Upstream pull:** merge; `sphere/`, `scripts/sphere/`, examples, and the shims
  are added files and merge clean. Conflicts can only appear in the files above —
  re-apply the seam intent per this table (mind the two grep-invisible items:
  the `setup()` hook order and the dispatch registration block), then re-run the
  review rule.
- **Cross-fork port:** copy `sphere/` verbatim, re-apply the seams, and decide
  per-fork whether to carry the shims (only needed for consumers that import the
  old dotted paths).
