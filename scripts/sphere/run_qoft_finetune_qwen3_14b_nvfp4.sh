#!/bin/bash
# Qwen3-14B QOFT finetuning — NVFP4 base weights + BF16 OFT adapters
#
# Two steps:
#   1. Convert HF NVFP4 checkpoint to Megatron format:
#        bash convert_nvfp4_checkpoint_direct.sh \
#            ${HF_MODEL_ROOT:-${HOME}/hf_models}/Qwen3-14B-NVFP4 \
#            ./checkpoints/Qwen3-14B-NVFP4
#
#   2. Finetune (this script):
#        bash run_qoft_finetune_qwen3_14b_nvfp4.sh
#
# Environment variables:
#   MEGATRON_CKPT  - Path to NVFP4 Megatron checkpoint (from step 1)
#   NUM_GPUS       - Number of GPUs (default: 8)
#   TP             - Tensor parallel size (default: 4)
#   PP             - Pipeline parallel size (default: 2)
#   TRAIN_ITERS    - Training iterations (default: 2000)
#   OUTPUT_DIR     - Output directory
#   BLOCK_SIZE     - OFT block size (default: 32)
#   EPS            - OFT epsilon (default: 6e-5)
#   COFT           - Set to 1 to enable COFT
#   BLOCK_SHARE    - Set to 1 to enable block sharing

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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/3rdparty/Megatron-LM:${PYTHONPATH:-}"

MEGATRON_CKPT="${MEGATRON_CKPT:-${MEGATRON_CKPT_ROOT:-${PWD}/checkpoints}/Qwen3-14B-NVFP4/iter_0000000}"
NUM_GPUS="${NUM_GPUS:-8}"
TP="${TP:-4}"
PP="${PP:-2}"
TRAIN_ITERS="${TRAIN_ITERS:-2000}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/qwen3_14b_qoft_nvfp4_tp4_pp2}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
EPS="${EPS:-6e-5}"
COFT="${COFT:-0}"
BLOCK_SHARE="${BLOCK_SHARE:-0}"

EXTRA_ARGS=()
if [ "${COFT}" = "1" ]; then
    EXTRA_ARGS+=(--coft)
fi
if [ "${BLOCK_SHARE}" = "1" ]; then
    EXTRA_ARGS+=(--block-share)
fi

echo "======================================"
echo "Qwen3-14B QOFT Fine-Tuning"
echo "  NVFP4 base weights (no BF16 alloc)"
echo "  GPUs: ${NUM_GPUS}, TP: ${TP}, PP: ${PP}"
echo "  OFT Block Size: ${BLOCK_SIZE}, eps: ${EPS}"
echo "  COFT: ${COFT}, Block Share: ${BLOCK_SHARE}"
echo "  Megatron Checkpoint: ${MEGATRON_CKPT}"
echo "  Output: ${OUTPUT_DIR}"
echo "======================================"

echo "[INFO] This launcher exercises resharded load from a TP=1 direct NVFP4 checkpoint." >&2
echo "[INFO] Default test layout is TP=${TP}, PP=${PP}, GPUs=${NUM_GPUS}." >&2

torchrun --nproc_per_node="${NUM_GPUS}" \
    examples/models/qwen3_14b/finetune_qoft_nvfp4.py \
    --pretrained-checkpoint "${MEGATRON_CKPT}" \
    --tp "${TP}" --pp "${PP}" \
    --train-iters "${TRAIN_ITERS}" \
    --output-dir "${OUTPUT_DIR}" \
    --block-size "${BLOCK_SIZE}" \
    --eps "${EPS}" \
    "${EXTRA_ARGS[@]}"
