#!/bin/bash
# Qwen3-30B-A3B MoE QOFT finetuning — INT4 W4A16 base weights + BF16 OFT adapters
#
# Environment variables:
#   MEGATRON_CKPT  - Path to INT4 Megatron checkpoint
#   NUM_GPUS       - Number of GPUs (default: 4)
#   TP / EP        - Tensor / Expert parallel size (default: 2 / 2)
#   PP             - Pipeline parallel size (default: 1)
#   TRAIN_ITERS    - Training iterations (default: 2000)
#   OUTPUT_DIR     - Output directory
#   BLOCK_SIZE     - OFT block size (default: 32)
#   GROUP_SIZE     - INT4 quantization group size (default: 128)

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

MEGATRON_CKPT="${MEGATRON_CKPT:-${MEGATRON_CKPT_ROOT:-${PWD}/checkpoints}/Qwen3-30B-A3B-Instruct-2507-w4a16}"
NUM_GPUS="${NUM_GPUS:-4}"
TP="${TP:-2}"
EP="${EP:-2}"
PP="${PP:-1}"
TRAIN_ITERS="${TRAIN_ITERS:-2000}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/qwen3_30b_a3b_qoft_int4}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
GROUP_SIZE="${GROUP_SIZE:-128}"

echo "======================================"
echo "Qwen3-30B-A3B MoE QOFT Fine-Tuning"
echo "  INT4 W4A16 base weights (no BF16 alloc)"
echo "  GPUs: ${NUM_GPUS}, TP: ${TP}, EP: ${EP}, PP: ${PP}"
echo "  OFT Block Size: ${BLOCK_SIZE}"
echo "  INT4 Group Size: ${GROUP_SIZE}"
echo "  Megatron Checkpoint: ${MEGATRON_CKPT}"
echo "  Output: ${OUTPUT_DIR}"
echo "======================================"

torchrun --nproc_per_node="${NUM_GPUS}" \
    examples/models/qwen3_moe/finetune_qoft_int4.py \
    --pretrained-checkpoint "${MEGATRON_CKPT}" \
    --tp "${TP}" --ep "${EP}" --pp "${PP}" \
    --train-iters "${TRAIN_ITERS}" \
    --output-dir "${OUTPUT_DIR}" \
    --block-size "${BLOCK_SIZE}" \
    --group-size "${GROUP_SIZE}"
