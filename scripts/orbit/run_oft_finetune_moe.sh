#!/bin/bash
# Qwen3-30B-A3B MoE OFT finetuning (BF16, no FP8) — interactive, 4x B200 GPUs (no SLURM)
#
# Usage:
#   bash run_oft_finetune_moe.sh
#
# Environment variables:
#   MEGATRON_CKPT  - Path to Megatron checkpoint (default: ./checkpoints/Qwen3-30B-A3B-Base)
#   NUM_GPUS       - Number of GPUs (default: 4)
#   TP             - Tensor parallel size (default: 2)
#   EP             - Expert parallel size (default: 2)
#   TRAIN_ITERS    - Training iterations (default: 2000)
#   BLOCK_SIZE     - OFT block size (default: 32)
#   OUTPUT_DIR     - Output directory for checkpoints and logs

set -euo pipefail

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

# W&B API key
# W&B API key: set WANDB_API_KEY in the environment or run wandb login beforehand.

# NCCL optimizations
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

MEGATRON_CKPT="${MEGATRON_CKPT:-./checkpoints/Qwen3-30B-A3B-Base}"
NUM_GPUS="${NUM_GPUS:-4}"
TP="${TP:-2}"
EP="${EP:-2}"
TRAIN_ITERS="${TRAIN_ITERS:-2000}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/qwen3_30b_a3b_oft}"

echo "======================================"
echo "Qwen3-30B-A3B MoE OFT Fine-Tuning (BF16)"
echo "GPUs: ${NUM_GPUS}, TP: ${TP}, EP: ${EP}"
echo "Block size: ${BLOCK_SIZE}"
echo "Output: ${OUTPUT_DIR}"
echo "======================================"
torchrun --nproc_per_node="${NUM_GPUS}" \
    scripts/orbit/models/qwen3_moe/finetune_oft_fp8.py \
    --pretrained-checkpoint "${MEGATRON_CKPT}" \
    --tp "${TP}" --ep "${EP}" \
    --block-size "${BLOCK_SIZE}" \
    --train-iters "${TRAIN_ITERS}" \
    --output-dir "${OUTPUT_DIR}" \
    --fp8-recipe none
