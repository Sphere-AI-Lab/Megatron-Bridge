#!/bin/bash
# Kimi-K2.5 INT4 QOFT load-only smoke test.
#
# This validates the expensive part first:
#   - build model on meta device
#   - load the INT4 Megatron checkpoint
#   - rewrite expert checkpoint keys into INT4 triplets
#   - register INT4 buffers on expert modules
#   - initialize OFT wrappers
#
# No training step is run.
#
# Environment variables:
#   MEGATRON_CKPT  - Path to INT4 Megatron checkpoint
#   NUM_GPUS       - Number of GPUs (default: 8)
#   TP / EP        - Tensor / Expert parallel size (default: 2 / 4)
#   PP             - Pipeline parallel size (default: 1)
#   TRAIN_ITERS    - Dummy train iters value for scheduler setup (default: 1)
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

export CUDA_DEVICE_MAX_CONNECTIONS=1

set -euo pipefail

# NCCL optimizations
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

# Ensure megatron.legacy is importable from the submodule source tree
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/3rdparty/Megatron-LM:${PYTHONPATH:-}"

MEGATRON_CKPT="${MEGATRON_CKPT:-./checkpoints/Kimi-K2.5}"
NUM_GPUS="${NUM_GPUS:-8}"
TP="${TP:-2}"
EP="${EP:-4}"
PP="${PP:-1}"
TRAIN_ITERS="${TRAIN_ITERS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/kimi_k25_qoft_int4_load_only}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"

echo "======================================"
echo "Kimi-K2.5 QOFT Load-Only Smoke Test"
echo "  INT4 base weights (no training step)"
echo "  GPUs: ${NUM_GPUS}, TP: ${TP}, EP: ${EP}, PP: ${PP}"
echo "  OFT Block Size: ${BLOCK_SIZE}"
echo "  Megatron Checkpoint: ${MEGATRON_CKPT}"
echo "  Output: ${OUTPUT_DIR}"
echo "======================================"

torchrun --nproc_per_node="${NUM_GPUS}" \
    examples/models/kimi_k25/finetune_qoft_int4.py \
    --pretrained-checkpoint "${MEGATRON_CKPT}" \
    --tp "${TP}" --ep "${EP}" --pp "${PP}" \
    --train-iters "${TRAIN_ITERS}" \
    --output-dir "${OUTPUT_DIR}" \
    --block-size "${BLOCK_SIZE}" \
    --skip-train
