# `scripts/orbit/` — Orbit launcher scripts

Launchers and utilities for Orbit's PEFT (OFT / LoRA) finetuning and quantized-checkpoint
workflows. Everything here is a thin wrapper: the real work lives in `examples/` and in
`src/megatron/bridge/orbit/`.

**Run every script from the repository root**, not from this directory:

```bash
bash scripts/orbit/run_peft_finetune.sh
```

All scripts are configured through **environment variables**, not flags (the two conversion
utilities and `stage_nvfp4_checkpoint_pair.sh` take positional/flag arguments instead). Each
script's header comment is the authoritative list of variables it honours; this README is a map,
not a replacement.

---

## Start here

| Want to… | Use |
|---|---|
| Finetune with OFT or LoRA on a normal (BF16 / FP8 / NVFP4) checkpoint | **`run_peft_finetune.sh`** |
| Finetune on an **INT4** checkpoint | a model-specific `run_qoft_finetune_*_int4.sh` |
| Finetune a very large MoE (Kimi-K2.5, Moonlight) | a model-specific `run_qoft_*` script |
| Convert / quantize a checkpoint first | see [Preparing a checkpoint](#preparing-a-checkpoint) |

### `run_peft_finetune.sh` — the generic launcher

Covers any model whose recipe is "`_peft_common()` plus an HF path", which is most of them.
Dispatches to `examples/finetune_peft.py`.

```bash
# OFT on Llama-3.2-1B, single GPU (all defaults)
bash scripts/orbit/run_peft_finetune.sh

# LoRA instead
PEFT=lora bash scripts/orbit/run_peft_finetune.sh

# OFT + NVFP4 on Qwen3-14B across 8 GPUs
PEFT=oft QUANT=nvfp4 HF_MODEL=Qwen/Qwen3-14B NUM_GPUS=8 TP=1 \
  MEGATRON_CKPT=./checkpoints/Qwen3-14B-NVFP4 \
  bash scripts/orbit/run_peft_finetune.sh
```

| Variable | Values | Default |
|---|---|---|
| `PEFT` | `oft`, `lora`, `dora`, `none` | `oft` |
| `QUANT` | `none`, `fp8`, `mxfp8`, `nvfp4` | `none` |
| `HF_MODEL` | HF model id or local path (supplies architecture **and** tokenizer) | `meta-llama/Llama-3.2-1B` |
| `MEGATRON_CKPT` | Megatron checkpoint to finetune from | `./checkpoints/<basename of HF_MODEL>` |
| `NUM_GPUS` | `torchrun --nproc_per_node` | `1` |
| `TP` / `PP` / `EP` / `CP` | parallelism sizes | `1` |
| `OUTPUT_DIR` | run directory | derived from model/peft/quant |
| `EXTRA_ARGS` | appended verbatim to the `python` command | — |

`QUANT=int4` is **not** available here. The INT4 path installs a checkpoint monkey-patch stack
(key rewriting, INT4 buffer registration) rather than setting config flags, so it still needs the
model-specific scripts below.

> **Constraint:** ModelOpt's `QuantSequentialMLP` forbids `TP>1` **and** `EP>1` at the same time
> for quantized MoE. `run_peft_finetune.sh` rejects that combination for `nvfp4` up front with an
> explanatory error. INT4 is exempt — it does not route through `QuantSequentialMLP`, which is why
> the INT4 scripts can use `TP=2 EP=4`.

---

## Model-specific finetuning launchers

Reach for these when `run_peft_finetune.sh` cannot express the run: INT4, or a model that needs a
custom pipeline layout / callbacks. Each pairs with an entrypoint under `examples/models/`.

| Script | Model | Base weights | Default GPUs / parallelism |
|---|---|---|---|
| `run_qoft_finetune_kimi_k25_int4.sh` | Kimi-K2.5 | INT4 | 8 · TP=2 EP=4 PP=1 |
| `run_qoft_finetune_kimi_k25_nvfp4.sh` | Kimi-K2.5 | NVFP4 | 8 · TP=1 EP=8 |
| `run_qoft_finetune_moonlight_16b_int4.sh` | Moonlight-16B-A3B | INT4 | 1 · TP=1 EP=1 PP=1 |
| `run_qoft_finetune_qwen3_14b_nvfp4.sh` | Qwen3-14B (dense) | NVFP4 | 8 · TP=4 PP=2 |
| `run_qoft_finetune_qwen3_30b_a3b_nvfp4.sh` | Qwen3-30B-A3B | NVFP4 | 8 · TP=1 EP=8 PP=1 |
| `run_qoft_finetune_qwen3_moe_fp8.sh` | Qwen3-30B-A3B | FP8 | 4 · TP=2 EP=2 |
| `run_qoft_finetune_qwen3_moe_int4.sh` | Qwen3-30B-A3B | INT4 W4A16 | 4 · TP=2 EP=2 PP=1 |
| `run_oft_finetune_moe.sh` | Qwen3-30B-A3B | **BF16** | 4×B200 · TP=2 EP=2 |
| `run_oft_finetune_qwen3_moe.sh` | Qwen3-30B-A3B | **FP8** | 4×B200 · TP=2 EP=2 |

The last two differ only in precision (BF16 vs FP8) despite their names suggesting different
models — `run_peft_finetune.sh` with `QUANT=none` / `QUANT=fp8` now covers the same ground.

Common variables across these: `MEGATRON_CKPT`, `HF_MODEL_PATH`, `NUM_GPUS`, `TP`/`EP`/`PP`,
`TRAIN_ITERS`, `OUTPUT_DIR`, `BLOCK_SIZE` (OFT block size, default 32). Several also accept
`EPS`, `COFT`, `BLOCK_SHARE`, `SEQ_LENGTH`, `GLOBAL_BATCH_SIZE`, `MICRO_BATCH_SIZE`.
Check the script header for the exact set.

### Smoke test

`run_qoft_load_kimi_k25_int4.sh` runs the expensive setup path **without training**: meta-device
build, INT4 checkpoint load, expert-key rewrite into INT4 triplets, buffer registration, OFT
wrapper init. Use it to validate a checkpoint before committing to a full run.

---

## Preparing a checkpoint

Finetuning expects a **Megatron** checkpoint. Typical flow from a BF16 HF model:

```bash
# 1. Quantize HF expert weights to INT4 (Kimi-K2 native format).
#    Non-expert weights (attention, norms, embeddings) stay BF16.
bash scripts/orbit/quantize_to_int4.sh <input_model> [output_model]

# 2. Convert the HF checkpoint to Megatron.
python scripts/orbit/conversion/convert_int4_checkpoint_direct.py \
    --hf-model-path <hf_int4_model> --megatron-path ./checkpoints/<name>

# 3. Verify the conversion round-tripped.
bash scripts/orbit/compare_converted_model_metadata.sh <hf_model> <megatron_path>
```

NVFP4 and FP8 use the corresponding `scripts/orbit/conversion/convert_{nvfp4,fp8}_checkpoint*.py`.

> Several script headers still tell you to run `convert_int4_checkpoint_direct.sh` or
> `convert_nvfp4_checkpoint_direct.sh`. **Those shell wrappers do not exist** — use the `.py`
> files in `scripts/orbit/conversion/` directly, as shown above.

| Utility | Arguments | Purpose |
|---|---|---|
| `quantize_to_int4.sh` | `<input_model> [output_model]` | BF16 HF expert weights → INT4 triplets |
| `compare_converted_model_metadata.sh` | `<hf_model> <megatron_path> [extra…]` | Tensor-metadata diff, HF source vs converted |
| `convert_qwen3_deepseek_style.py` | `--model-dir --save-dir --mode {fp4_only,mixed}` | Qwen3 BF16 → DeepSeek-style FP4/FP8 layout |
| `check_ckpt_nan.py` | `[checkpoint_path]` (positional, has a default) | Scan a `torch_dist` checkpoint for NaN/Inf/outliers |
| `stage_nvfp4_checkpoint_pair.sh` | `--hf-model --megatron-dist --env-file`, `--hf-only` | Copy an HF NVFP4 model + its Megatron dist checkpoint to node-local storage |

Staging matters on clusters where the shared filesystem is slow: load from node-local disk
instead. The script writes resolved paths to an env file you then source.

---

## Conventions

- **CUDA setup.** Each script tries `module load cuda/13.2` and `module load nccl` (both
  best-effort), then sets `CUDA_HOME`, `PATH`, and `LD_LIBRARY_PATH`. Override with
  `CUDA_MODULE`, `NCCL_MODULE`, `CUDA_HOME`, `CUDNN_HOME`.
- **Weights & Biases.** Set `WANDB_API_KEY` in your environment or run `wandb login` first. Some
  scripts also read `${HOME}/.wandb_key` when the variable is unset.
  **Never commit an API key into a script**, even commented out — one was published to GitHub
  this way and had to be rotated.
- **Defaults are cluster-shaped.** `HF_MODEL_ROOT` defaults to `${HOME}/hf_models` and
  `MEGATRON_CKPT_ROOT` to `${PWD}/checkpoints`. Set them once for your machine.
- **No SLURM.** These are interactive launchers using `torchrun`. Wrap them in your own
  `sbatch`/`srun` for batch submission.

## Adding a script

Prefer extending `run_peft_finetune.sh` (a new `QUANT` preset, a new flag) over copying a script.
The launchers here are heavily duplicated — the two Kimi scripts are 89% identical across 550
lines — and every copy is another place a fix has to land.
