#!/bin/bash
# GPT-OSS 20B MoE LoRA finetuning — interactive, 4x B200 GPUs (no SLURM)
#
# Usage:
#   bash run_lora_finetune_moe.sh
#
# Environment variables:
#   HF_MODEL       - HuggingFace model name (default: openai/gpt-oss-20b)
#   MEGATRON_CKPT  - Path to Megatron checkpoint (default: ./checkpoints/gpt-oss-20b)
#   NUM_GPUS       - Number of GPUs (default: 4)
#   TP             - Tensor parallel size (default: 2)
#   EP             - Expert parallel size (default: 2)
#   TRAIN_ITERS    - Training iterations (default: 10)
#   OUTPUT_DIR     - Output directory for checkpoints and logs

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
# 注意：你需要根据实际版本匹配，这里以 12.x 系列为例
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -n "${CUDNN_HOME:-}" ]]; then
    export LD_LIBRARY_PATH="${CUDNN_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

set -euo pipefail

# W&B API key (set your key here or run `wandb login` beforehand)
# export WANDB_API_KEY="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"

HF_MODEL="${HF_MODEL:-openai/gpt-oss-20b}"
MEGATRON_CKPT="${MEGATRON_CKPT:-./checkpoints/gpt-oss-20b}"
NUM_GPUS="${NUM_GPUS:-4}"
TP="${TP:-2}"
EP="${EP:-2}"
TRAIN_ITERS="${TRAIN_ITERS:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/gpt_oss_20b_lora}"

# NCCL optimizations
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

echo "======================================"
echo "GPT-OSS 20B MoE LoRA Fine-Tuning"
echo "GPUs: ${NUM_GPUS}, TP: ${TP}, EP: ${EP}"
echo "Output: ${OUTPUT_DIR}"
echo "======================================"
torchrun --nproc_per_node="${NUM_GPUS}" \
    examples/models/gpt_oss/finetune_lora.py \
    --pretrained-checkpoint "${MEGATRON_CKPT}" \
    --tp "${TP}" --ep "${EP}" \
    --train-iters "${TRAIN_ITERS}" \
    --output-dir "${OUTPUT_DIR}"
