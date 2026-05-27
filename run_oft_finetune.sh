#!/bin/bash
# Quickstart script for OFT finetuning of Llama 3.2 1B
#
# Usage:
#   bash run_oft_finetune.sh
#
# Environment variables:
#   HF_MODEL       - HuggingFace model name (default: meta-llama/Llama-3.2-1B)
#   MEGATRON_CKPT  - Path to Megatron checkpoint (default: ./checkpoints/llama32_1b)
#   NUM_GPUS       - Number of GPUs (default: 1)

set -euo pipefail

# 1. 设置 CUDA 主路径
if command -v module >/dev/null 2>&1; then
    module load "${CUDA_MODULE:-cuda/13.2}" || true
fi
if command -v module >/dev/null 2>&1; then
    module load "${NCCL_MODULE:-nccl}" || true
fi
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH

# 2. 将 CUDA 的库路径、cuDNN 路径和 NCCL 路径全部加入 LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -n "${CUDNN_HOME:-}" ]]; then
    export LD_LIBRARY_PATH="${CUDNN_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# W&B API key (set your key here or run `wandb login` beforehand)
export WANDB_API_KEY="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"

HF_MODEL="${HF_MODEL:-meta-llama/Llama-3.2-1B}"
MEGATRON_CKPT="${MEGATRON_CKPT:-./checkpoints/Llama-3.2-1B}"
NUM_GPUS="${NUM_GPUS:-1}"

echo "Starting OFT finetuning with ${NUM_GPUS} GPU(s)..."
torchrun --nproc_per_node="${NUM_GPUS}" \
    tutorials/recipes/llama/01_quickstart_finetune_oft.py \
    --pretrained-checkpoint "${MEGATRON_CKPT}"
