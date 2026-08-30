# `scripts/orbit/` — Orbit launcher scripts

Launchers and utilities for Orbit's PEFT (OFT / LoRA) finetuning and quantized-checkpoint
workflows. The `run_*.sh` scripts are thin wrappers around the Python entrypoints in this
tree (`finetune_peft.py`, `models/`, `conversion/`); the core logic lives in
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
Dispatches to `scripts/orbit/finetune_peft.py`.

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
custom pipeline layout / callbacks. Each pairs with an entrypoint under `scripts/orbit/models/`.

| Script | Model | Base weights | Default GPUs / parallelism |
|---|---|---|---|
| `run_qoft_finetune_kimi_k25_int4.sh` | Kimi-K2.5 | INT4 | 8 · TP=2 EP=4 PP=1 |
| `run_qoft_finetune_kimi_k25_nvfp4.sh` | Kimi-K2.5 | NVFP4 | 8 · TP=1 EP=8 |
| `run_qoft_finetune_moonlight_16b_int4.sh` | Moonlight-16B-A3B | INT4 | 1 · TP=1 EP=1 PP=1 |
| `run_qoft_finetune_qwen3_moe_fp8.sh` | Qwen3-30B-A3B | FP8 | 4 · TP=2 EP=2 |
| `run_qoft_finetune_qwen3_moe_int4.sh` | Qwen3-30B-A3B | INT4 W4A16 | 4 · TP=2 EP=2 PP=1 |

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

### Verified conversion examples

All commands run from the repository root; each was exercised end-to-end on a real
checkpoint (single GPU). `MODELS` stands for your HF model root, e.g.
`${HF_MODEL_ROOT:-$HOME/hf_models}`.

```bash
# FP8 (e.g. Qwen/Qwen3-30B-A3B-FP8). Streaming direct writer:
python scripts/orbit/conversion/convert_fp8_checkpoint_direct.py     --hf-model-path $MODELS/Qwen3-30B-A3B-FP8     --megatron-path ./checkpoints/Qwen3-30B-A3B-FP8-mcore
# Full variant (instantiates the BF16 model; needs model-sized RAM):
python scripts/orbit/conversion/convert_fp8_checkpoint.py     --hf-model-path $MODELS/Qwen3-30B-A3B-FP8     --megatron-path ./checkpoints/Qwen3-30B-A3B-FP8-mcore-full

# NVFP4 (a ModelOpt NVFP4 export, e.g. nvidia/Qwen3-30B-A3B-NVFP4):
python scripts/orbit/conversion/convert_nvfp4_checkpoint_direct.py     --hf-model-path $MODELS/Qwen3-30B-A3B-NVFP4     --megatron-path ./checkpoints/Qwen3-30B-A3B-NVFP4-mcore
# Inspect the ModelOpt quantizer/meta keys of such a bundle:
python scripts/orbit/conversion/dump_nvfp4_meta_keys.py     --hf-model-path $MODELS/Qwen3-30B-A3B-NVFP4

# INT4 from a BF16 model (e.g. Moonlight-16B-A3B): quantize, then convert.
bash scripts/orbit/quantize_to_int4.sh $MODELS/Moonlight-16B-A3B $MODELS/Moonlight-16B-A3B-INT4
python scripts/orbit/conversion/convert_int4_checkpoint_direct.py     --hf-model-path $MODELS/Moonlight-16B-A3B-INT4     --megatron-path ./checkpoints/Moonlight-16B-A3B-INT4-mcore

# Native-INT4 releases (Kimi-K2.5 / Kimi-K2.7 ship INT4 triplets already):
python scripts/orbit/conversion/convert_int4_checkpoint_direct.py     --hf-model-path $MODELS/Kimi-K2.7-Code     --megatron-path ./checkpoints/Kimi-K2.7-Code-INT4-mcore

# Verify any of the above (strict tensor-metadata diff, HF vs Megatron):
python scripts/orbit/conversion/compare_model_metadata.py     --hf-model-path $MODELS/<hf_model> --megatron-path ./checkpoints/<name>
```

### Qwen3-30B-A3B NVFP4 QOFT bring-up

The following converts a ModelOpt NVFP4 export and runs one QOFT training iteration. It was
validated on a single NVIDIA B200 with TP=EP=PP=1 and sequence length 256. Use a
Blackwell-class GPU such as B200 for native NVFP4 execution.

On Radixark/Miles integration branches, make the `miles` package importable before conversion
or training. Skip this export when `miles` is already installed in the active environment:

```bash
export MILES_ROOT=/path/to/miles
export PYTHONPATH="$PWD/src:$PWD/3rdparty/Megatron-LM:$MILES_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

Convert the Hugging Face NVFP4 checkpoint to Megatron's distributed-checkpoint format:

```bash
MODELS=${HF_MODEL_ROOT:-$HOME/hf_models}
CHECKPOINTS=${MEGATRON_CKPT_ROOT:-$PWD/checkpoints}
HF_MODEL=$MODELS/Qwen3-30B-A3B-NVFP4
MEGATRON_CKPT=$CHECKPOINTS/Qwen3-30B-A3B-NVFP4-mcore

uv run python scripts/orbit/conversion/convert_nvfp4_checkpoint_direct.py \
    --hf-model-path "$HF_MODEL" \
    --megatron-path "$MEGATRON_CKPT"
```

Then run a one-iteration QOFT check before increasing the sequence length, batch size, GPU
count, or training duration:

```bash
WANDB_MODE=disabled uv run python -m torch.distributed.run \
    --nproc_per_node=1 \
    scripts/orbit/finetune_qoft.py \
    --quant nvfp4 \
    --hf-model-path "$HF_MODEL" \
    --pretrained-checkpoint "$MEGATRON_CKPT" \
    --peft oft \
    --tp 1 \
    --ep 1 \
    --pp 1 \
    --no-sp \
    --train-iters 1 \
    --global-batch-size 1 \
    --micro-batch-size 1 \
    --seq-length 256 \
    --log-interval 1 \
    --skip-eval \
    --debug-nan \
    --debug-nan-steps 1 \
    --output-dir ./outputs/qwen3-30b-a3b-nvfp4-qoft-bringup
```

A successful check reaches iteration `1/1` with a finite, nonzero loss and gradient norm,
zero skipped iterations, and zero NaN iterations.

Container runtimes must expose the NVIDIA driver library to Triton. If startup reports that
`libcuda.so` cannot be found, point Triton and the compiler/linker at a directory containing
both `libcuda.so` and `libcuda.so.1` (a node-local shim is sufficient):

```bash
export TRITON_LIBCUDA_PATH=/path/to/libcuda-directory
export LIBRARY_PATH="$TRITON_LIBCUDA_PATH${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$TRITON_LIBCUDA_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Leave `NVTE_FUSED_ATTN` unset when using the default `auto` attention backend. A contradictory
override such as `NVTE_FUSED_ATTN=0` is rejected during model construction; select an explicit
`--attention-backend` instead when a non-default backend is required.

### Qwen3-30B-A3B FP8 QOFT bring-up

The FP8 path uses the same Qwen3 MoE architecture and QOFT entrypoint. This one-iteration
check was validated on a single NVIDIA B200 with TP=EP=PP=1 and sequence length 256, but FP8
does not require a Blackwell-class GPU.

Use the `MILES_ROOT` and `PYTHONPATH` setup from the NVFP4 example above when `miles` is not
already installed, then convert the Hugging Face FP8 checkpoint:

```bash
MODELS=${HF_MODEL_ROOT:-$HOME/hf_models}
CHECKPOINTS=${MEGATRON_CKPT_ROOT:-$PWD/checkpoints}
HF_MODEL=$MODELS/Qwen3-30B-A3B-FP8
MEGATRON_CKPT=$CHECKPOINTS/Qwen3-30B-A3B-FP8-mcore

uv run python scripts/orbit/conversion/convert_fp8_checkpoint_direct.py \
    --hf-model-path "$HF_MODEL" \
    --megatron-path "$MEGATRON_CKPT"
```

The direct converter currently writes ModelOpt metadata using the MCore distributed format,
while the generic checkpoint probe looks for `modelopt_state/common.pt`. If the generated
checkpoint has distributed ModelOpt metadata instead, move that directory aside before
training. This is reversible; the dedicated FP8 loader still restores the quantized weights
and scales directly from the main checkpoint:

```bash
MODELOPT_STATE=$MEGATRON_CKPT/iter_0000000/modelopt_state
if [[ -d "$MODELOPT_STATE" && ! -f "$MODELOPT_STATE/common.pt" ]]; then
    test ! -e "${MODELOPT_STATE}.mcore-unused"
    mv "$MODELOPT_STATE" "${MODELOPT_STATE}.mcore-unused"
fi
```

Run the same bounded QOFT check:

```bash
WANDB_MODE=disabled uv run python -m torch.distributed.run \
    --nproc_per_node=1 \
    scripts/orbit/finetune_qoft.py \
    --quant fp8 \
    --hf-model-path "$HF_MODEL" \
    --pretrained-checkpoint "$MEGATRON_CKPT" \
    --peft oft \
    --tp 1 \
    --ep 1 \
    --pp 1 \
    --no-sp \
    --train-iters 1 \
    --global-batch-size 1 \
    --micro-batch-size 1 \
    --seq-length 256 \
    --log-interval 1 \
    --skip-eval \
    --debug-nan \
    --debug-nan-steps 1 \
    --output-dir ./outputs/qwen3-30b-a3b-fp8-qoft-bringup
```

A successful check reaches iteration `1/1` with a finite, nonzero loss and gradient norm,
zero skipped iterations, and zero NaN iterations.

**Tiny-model validation** — before committing to a multi-hundred-GB conversion, cut a
byte-identical few-layer slice with upstream's toy tool and run the same chain on it
(minutes instead of hours; this exact recipe validated Kimi-K2.7 INT4 end-to-end):

```bash
python examples/conversion/create_hf_toy_model.py     $MODELS/Kimi-K2.7-Code $MODELS/Kimi-K2.7-Code-4layers --num-hidden-layers 4
python scripts/orbit/conversion/convert_int4_checkpoint_direct.py     --hf-model-path $MODELS/Kimi-K2.7-Code-4layers     --megatron-path ./checkpoints/Kimi-K2.7-Code-4layers-INT4-mcore
python scripts/orbit/conversion/compare_model_metadata.py     --hf-model-path $MODELS/Kimi-K2.7-Code-4layers     --megatron-path ./checkpoints/Kimi-K2.7-Code-4layers-INT4-mcore
```

**Memory-constrained hosts** — a full 555 GB Kimi conversion peaks around 1.1 TB by
default (whole output state in RAM plus source mmaps). The INT4 direct converter has
opt-in relief valves; stage the source model on node-local disk and set:

```bash
export MEGATRON_BRIDGE_DIRECT_USE_SPILL=1        # spill converted tensors to disk
export MEGATRON_BRIDGE_DIRECT_SPILL_DIR=/path/to/node-local  # spill location
export MEGATRON_BRIDGE_PYMMAP_READER=1           # lazy no-populate source reads
```

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
