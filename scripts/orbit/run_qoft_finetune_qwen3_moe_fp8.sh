#!/bin/bash
# Qwen3-30B-A3B MoE QOFT finetuning — FP8 base weights + BF16 OFT adapters
#
# Two steps:
#   1. Convert HF FP8 checkpoint to Megatron format (preserving FP8):
#        python scripts/orbit/conversion/convert_fp8_checkpoint.py \
#            --hf-model-path $HF_MODEL_PATH \
#            --megatron-path $MEGATRON_CKPT
#
#   2. Finetune (this script):
#        bash run_qoft_finetune_qwen3_moe.sh
#
# Environment variables:
#   MEGATRON_CKPT  - Path to FP8 Megatron checkpoint (from step 1)
#   NUM_GPUS       - Number of GPUs (default: 4)
#   TP / EP        - Tensor / Expert parallel size (default: 2)
#   TRAIN_ITERS    - Training iterations (default: 2000)
#   OUTPUT_DIR     - Output directory
#   BLOCK_SIZE     - OFT block size (default: 32)

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

# Setup wandb
if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.wandb_key" ]]; then
    export WANDB_API_KEY
    WANDB_API_KEY="$(<"${HOME}/.wandb_key")"
fi

export CUDA_DEVICE_MAX_CONNECTIONS=1

set -euo pipefail

# NCCL optimizations
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

MEGATRON_CKPT="${MEGATRON_CKPT:-./checkpoints/Qwen3-30B-A3B-Instruct-2507-FP8}"
NUM_GPUS="${NUM_GPUS:-4}"
TP="${TP:-2}"
EP="${EP:-2}"
TRAIN_ITERS="${TRAIN_ITERS:-2000}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/qwen3_30b_a3b_qoft}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"

echo "======================================"
echo "Qwen3-30B-A3B MoE QOFT Fine-Tuning"
echo "  FP8 base weights (no BF16 alloc)"
echo "  GPUs: ${NUM_GPUS}, TP: ${TP}, EP: ${EP}"
echo "  OFT Block Size: ${BLOCK_SIZE}"
echo "  Megatron Checkpoint: ${MEGATRON_CKPT}"
echo "  Output: ${OUTPUT_DIR}"
echo "======================================"

torchrun --nproc_per_node="${NUM_GPUS}" \
    scripts/orbit/models/qwen3_moe/finetune_qoft.py \
    --pretrained-checkpoint "${MEGATRON_CKPT}" \
    --tp "${TP}" --ep "${EP}" \
    --train-iters "${TRAIN_ITERS}" \
    --output-dir "${OUTPUT_DIR}" \
    --block-size "${BLOCK_SIZE}"
