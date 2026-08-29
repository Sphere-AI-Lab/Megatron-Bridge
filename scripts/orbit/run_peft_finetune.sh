#!/bin/bash
# Generic PEFT finetuning launcher.
#
# Replaces run_lora_finetune.sh / run_oft_finetune.sh (which differed only in
# their target .py) and covers any model whose recipe is "_peft_common() plus an
# HF path" -- see scripts/orbit/finetune_peft.py.
#
# Usage:
#   bash scripts/orbit/run_peft_finetune.sh                       # oft, Llama-3.2-1B
#   PEFT=lora bash scripts/orbit/run_peft_finetune.sh
#   PEFT=oft QUANT=nvfp4 HF_MODEL=Qwen/Qwen3-14B NUM_GPUS=8 TP=1 \
#       MEGATRON_CKPT=./checkpoints/Qwen3-14B-NVFP4 \
#       bash scripts/orbit/run_peft_finetune.sh
#
# Environment variables:
#   PEFT           - oft | lora | dora | none            (default: oft)
#   QUANT          - none | fp8 | mxfp8 | nvfp4          (default: none)
#   HF_MODEL       - HF model id or local path           (default: meta-llama/Llama-3.2-1B)
#   MEGATRON_CKPT  - Megatron checkpoint to finetune from
#                    (default: ./checkpoints/<basename of HF_MODEL>)
#   NUM_GPUS       - torchrun --nproc_per_node           (default: 1)
#   TP / PP / EP / CP - parallelism sizes                (default: 1)
#   OUTPUT_DIR     - run directory                       (default: chosen by the entrypoint)
#   EXTRA_ARGS     - appended verbatim to the python command
#
# INT4 is not covered here: that path installs a checkpoint monkey-patch stack.
# Use scripts/orbit/models/*/finetune_qoft_int4.py for INT4.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# CUDA / cuDNN / NCCL paths
if command -v module >/dev/null 2>&1; then
    module load "${CUDA_MODULE:-cuda/13.2}" || true
    module load "${NCCL_MODULE:-nccl}" || true
fi
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -n "${CUDNN_HOME:-}" ]]; then
    export LD_LIBRARY_PATH="${CUDNN_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# W&B API key: set WANDB_API_KEY in the environment or run wandb login beforehand.
# Never commit a key into this file.

PEFT="${PEFT:-oft}"
QUANT="${QUANT:-none}"
HF_MODEL="${HF_MODEL:-meta-llama/Llama-3.2-1B}"
MEGATRON_CKPT="${MEGATRON_CKPT:-./checkpoints/$(basename "${HF_MODEL}")}"
NUM_GPUS="${NUM_GPUS:-1}"
TP="${TP:-1}"; PP="${PP:-1}"; EP="${EP:-1}"; CP="${CP:-1}"

args=(
    --model-path "${HF_MODEL}"
    --pretrained-checkpoint "${MEGATRON_CKPT}"
    --peft "${PEFT}"
    --quant "${QUANT}"
    --tp "${TP}" --pp "${PP}" --ep "${EP}" --cp "${CP}"
)
if [[ -n "${OUTPUT_DIR:-}" ]]; then
    args+=(--output-dir "${OUTPUT_DIR}")
fi

echo "Starting ${PEFT^^} finetuning of ${HF_MODEL} (quant=${QUANT}) on ${NUM_GPUS} GPU(s)..."
echo "  checkpoint: ${MEGATRON_CKPT}"
echo "  TP=${TP} PP=${PP} EP=${EP} CP=${CP}"

torchrun --nproc_per_node="${NUM_GPUS}" \
    scripts/orbit/finetune_peft.py \
    "${args[@]}" ${EXTRA_ARGS:-}
