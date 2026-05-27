#!/bin/bash
# Llama 3 8B INT4 QOFT load-only debug.
#
# This validates the expensive startup path first:
#   - build on meta device
#   - load the Megatron distributed checkpoint
#   - apply OFT wrappers
#   - print any remaining bias tensors / meta bias tensors
#
# No training step is run.

if command -v module >/dev/null 2>&1; then
    module load "${CUDA_MODULE:-cuda/13.2}" || true
fi
if command -v module >/dev/null 2>&1; then
    module load "${NCCL_MODULE:-nccl}" || true
fi
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -n "${CUDNN_HOME:-}" ]]; then
    export LD_LIBRARY_PATH="${CUDNN_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

set -euo pipefail

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/3rdparty/Megatron-LM:${PYTHONPATH:-}"

HF_MODEL_PATH="${HF_MODEL_PATH:-${HF_MODEL_ROOT:-${HOME}/hf_models}/Meta-Llama-3-8B-Instruct-W4A16-compressed-tensors-test}"
MEGATRON_CKPT="${MEGATRON_CKPT:-${MEGATRON_CKPT_ROOT:-${PWD}/checkpoints}/Llama-3-8B-Instruct-W4A16/iter_0000000}"
NUM_GPUS="${NUM_GPUS:-1}"
TP="${TP:-1}"
PP="${PP:-1}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
REPORT_LIMIT="${REPORT_LIMIT:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/llama3_8b_qoft_int4_load_debug}"

echo "======================================"
echo "Llama 3 8B INT4 QOFT Load-Only Debug"
echo "  GPUs: ${NUM_GPUS}, TP: ${TP}, PP: ${PP}"
echo "  OFT Block Size: ${BLOCK_SIZE}"
echo "  HF Model: ${HF_MODEL_PATH}"
echo "  Megatron Checkpoint: ${MEGATRON_CKPT}"
echo "  Output: ${OUTPUT_DIR}"
echo "======================================"

torchrun --nproc_per_node="${NUM_GPUS}" \
    examples/models/llama3_8b/load_qoft_int4_debug.py \
    --pretrained-checkpoint "${MEGATRON_CKPT}" \
    --hf-model-path "${HF_MODEL_PATH}" \
    --tp "${TP}" --pp "${PP}" \
    --block-size "${BLOCK_SIZE}" \
    --report-limit "${REPORT_LIMIT}" \
    --output-dir "${OUTPUT_DIR}"
