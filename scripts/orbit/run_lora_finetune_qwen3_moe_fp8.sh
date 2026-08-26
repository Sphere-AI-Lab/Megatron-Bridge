#!/bin/bash
# Qwen3-30B-A3B MoE LoRA finetuning with FP8 — interactive, 4x B200 GPUs (no SLURM)
#
# Usage:
#   bash run_lora_finetune_qwen3_moe_fp8.sh
#
# Environment variables:
#   MEGATRON_CKPT  - Path to Megatron checkpoint (default: ./checkpoints/Qwen3-30B-A3B)
#   NUM_GPUS       - Number of GPUs (default: 4)
#   TP             - Tensor parallel size (default: 2)
#   EP             - Expert parallel size (default: 2)
#   TRAIN_ITERS    - Training iterations (default: 10)
#   OUTPUT_DIR     - Output directory for checkpoints and logs
#   FP8_RECIPE     - FP8 recipe: "hopper" or "blackwell" (default: hopper)

# 1. CUDA setup
if command -v module >/dev/null 2>&1; then
    module load "${CUDA_MODULE:-cuda/13.2}" || true
fi
if command -v module >/dev/null 2>&1; then
    module load "${NCCL_MODULE:-nccl}" || true
fi
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH

# 2. Library paths
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -n "${CUDNN_HOME:-}" ]]; then
    export LD_LIBRARY_PATH="${CUDNN_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

set -euo pipefail

# W&B API key
# export WANDB_API_KEY="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"

# NCCL optimizations
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

MEGATRON_CKPT="${MEGATRON_CKPT:-./checkpoints/Qwen3-30B-A3B-Base}"
NUM_GPUS="${NUM_GPUS:-4}"
TP="${TP:-2}"
EP="${EP:-2}"
TRAIN_ITERS="${TRAIN_ITERS:-2000}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/qwen3_30b_a3b_lora_fp8}"
FP8_RECIPE="${FP8_RECIPE:-hopper}"

echo "======================================"
echo "Qwen3-30B-A3B MoE LoRA + FP8 Fine-Tuning"
echo "GPUs: ${NUM_GPUS}, TP: ${TP}, EP: ${EP}"å
echo "FP8 Recipe: ${FP8_RECIPE}"
echo "Output: ${OUTPUT_DIR}"
echo "======================================"

torchrun --nproc_per_node="${NUM_GPUS}" \
    examples/models/qwen3_moe/finetune_lora_fp8.py \
    --pretrained-checkpoint "${MEGATRON_CKPT}" \
    --tp "${TP}" --ep "${EP}" \
    --train-iters "${TRAIN_ITERS}" \
    --output-dir "${OUTPUT_DIR}" \
    --fp8-recipe "${FP8_RECIPE}"
