#!/bin/bash
# Generic quantized-base PEFT (QOFT / QLoRA) finetuning launcher.
#
# Finetunes OFT or LoRA adapters over a base model whose weights stay in their
# quantized format (FP8 / INT4 / NVFP4). Requires a converted Megatron
# checkpoint from scripts/orbit/conversion/ — see the "Preparing a checkpoint"
# section of scripts/orbit/README.md. The architecture (Qwen3-MoE, Kimi-K2.5,
# Moonlight) is detected from HF_MODEL_PATH and brings its own defaults.
#
# Replaces the per-model run_qoft_finetune_*.sh launchers.
#
# Usage:
#   QUANT=fp8 \
#   HF_MODEL_PATH=${HF_MODEL_ROOT:-$HOME/hf_models}/Qwen3-30B-A3B-FP8 \
#   MEGATRON_CKPT=./checkpoints/Qwen3-30B-A3B-FP8-mcore \
#   bash scripts/orbit/run_qoft_finetune.sh
#
# Environment variables:
#   QUANT            - fp8 | int4 | nvfp4 (required)
#   HF_MODEL_PATH    - HF model directory or id (required)
#   MEGATRON_CKPT    - converted Megatron checkpoint directory (required)
#   PEFT             - oft | lora (default: oft)
#   NUM_GPUS         - torchrun --nproc_per_node (default: 4)
#   TP, EP, PP       - parallelism overrides (defaults come from the architecture)
#   TRAIN_ITERS, GLOBAL_BATCH_SIZE, MICRO_BATCH_SIZE, SEQ_LENGTH, BLOCK_SIZE
#   OUTPUT_DIR       - experiment directory (default: ./nemo_experiments/<slug>)
#   SKIP_TRAIN       - 1 = load-only smoke: load checkpoint, init PEFT, exit
#   SAVE_CHECKPOINTS - 1 = save run checkpoints under OUTPUT_DIR/checkpoints
#   EXTRA_ARGS       - appended verbatim to the python command
#
# On clusters with slow shared filesystems, stage the pair to node-local
# storage first and point the variables at the staged copies:
#   bash scripts/orbit/stage_nvfp4_checkpoint_pair.sh \
#       --hf-model "$HF_MODEL_PATH" --megatron-dist "$MEGATRON_CKPT" \
#       --env-file /tmp/qoft_staged.env
#   source /tmp/qoft_staged.env

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

: "${QUANT:?set QUANT=fp8|int4|nvfp4}"
: "${HF_MODEL_PATH:?set HF_MODEL_PATH to the HF model directory or id}"
: "${MEGATRON_CKPT:?set MEGATRON_CKPT to the converted Megatron checkpoint}"

NUM_GPUS="${NUM_GPUS:-4}"

args=(
    --quant "${QUANT}"
    --hf-model-path "${HF_MODEL_PATH}"
    --pretrained-checkpoint "${MEGATRON_CKPT}"
    --peft "${PEFT:-oft}"
)
[ -n "${TP:-}" ] && args+=(--tp "${TP}")
[ -n "${EP:-}" ] && args+=(--ep "${EP}")
[ -n "${PP:-}" ] && args+=(--pp "${PP}")
[ -n "${TRAIN_ITERS:-}" ] && args+=(--train-iters "${TRAIN_ITERS}")
[ -n "${GLOBAL_BATCH_SIZE:-}" ] && args+=(--global-batch-size "${GLOBAL_BATCH_SIZE}")
[ -n "${MICRO_BATCH_SIZE:-}" ] && args+=(--micro-batch-size "${MICRO_BATCH_SIZE}")
[ -n "${SEQ_LENGTH:-}" ] && args+=(--seq-length "${SEQ_LENGTH}")
[ -n "${BLOCK_SIZE:-}" ] && args+=(--block-size "${BLOCK_SIZE}")
[ -n "${OUTPUT_DIR:-}" ] && args+=(--output-dir "${OUTPUT_DIR}")
[ "${SKIP_TRAIN:-0}" = "1" ] && args+=(--skip-train)
[ "${SAVE_CHECKPOINTS:-0}" = "1" ] && args+=(--save-checkpoints)

echo "======================================"
echo "QOFT finetuning (${QUANT}, ${PEFT:-oft})"
echo "  HF model:   ${HF_MODEL_PATH}"
echo "  checkpoint: ${MEGATRON_CKPT}"
echo "  GPUs:       ${NUM_GPUS}"
echo "======================================"

torchrun --nproc_per_node="${NUM_GPUS}" \
    scripts/orbit/finetune_qoft.py \
    "${args[@]}" ${EXTRA_ARGS:-}
