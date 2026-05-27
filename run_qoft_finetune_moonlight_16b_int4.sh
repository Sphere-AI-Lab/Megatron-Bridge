#!/bin/bash
# Moonlight-16B-A3B QOFT finetuning - INT4 base weights + BF16 OFT adapters.
#
# Suggested flow:
#   1. Quantize BF16 HF experts to INT4 triplets:
#        bash test_to_kimi_int4_conversion.sh
#   2. Convert HF INT4 checkpoint to Megatron INT4:
#        bash convert_int4_checkpoint_direct.sh \
#            ${HF_MODEL_ROOT:-${HOME}/hf_models}/Moonlight-16B-A3B-INT4 \
#            ./checkpoints/Moonlight-16B-A3B-INT4
#   3. Finetune with this script:
#        bash run_qoft_finetune_moonlight_16b_int4.sh
#
# Environment variables:
#   MEGATRON_CKPT      - Path to INT4 Megatron checkpoint
#   HF_MODEL_PATH      - HF model/tokenizer path
#   NUM_GPUS           - Number of GPUs (default: 1)
#   TP / EP / PP       - Parallel sizes (default: 1 / 1 / 1)
#   SP                 - Set to 1 to enable sequence parallel
#   TRAIN_ITERS        - Training iterations (default: 100)
#   GLOBAL_BATCH_SIZE  - Global batch size (default: 32)
#   MICRO_BATCH_SIZE   - Micro batch size (default: 1)
#   SEQ_LENGTH         - Sequence length (default: 2048)
#   OUTPUT_DIR         - Output directory
#   BLOCK_SIZE         - OFT block size (default: 32)
#   SKIP_TRAIN         - Set to 1 for load-only smoke mode
#   PROFILE_MEMORY     - Set to 1 to print CUDA memory around the first train step
#   PROFILE_MEMORY_STEPS - Number of initial steps to profile (default: 1)
#   DEBUG_NAN          - Set to 1 to trace where non-finite values first appear
#   DEBUG_NAN_STEPS    - Number of initial steps to inspect for NaNs (default: 3)
#   MEMORY_SMOKE_TEST  - Set to 1 for a one-step train run with memory profiling,
#                        no checkpoint save, and no validation
#   LOG_INTERVAL       - Training log interval (default: 10, or 1 in memory smoke mode)
#   INT4_ACTIVE_EXPERT_CHUNK_SIZE - Active-expert chunk size for grouped INT4 execution
#                                   (default: 4, set 0 to disable)
#   INT4_GROUPED_CHUNK_BACKEND - Grouped INT4 chunk execution backend
#                                (default: python, set to te to enable TE grouped GEMM)

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

export CUDA_DEVICE_MAX_CONNECTIONS=1

set -euo pipefail

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/3rdparty/Megatron-LM:${PYTHONPATH:-}"

MEGATRON_CKPT="${MEGATRON_CKPT:-${SCRIPT_DIR}/checkpoints/Moonlight-16B-A3B-INT4}"
HF_MODEL_PATH="${HF_MODEL_PATH:-${HF_MODEL_ROOT:-${HOME}/hf_models}/Moonlight-16B-A3B}"
NUM_GPUS="${NUM_GPUS:-8}"
TP="${TP:-1}"
EP="${EP:-1}"
PP="${PP:-1}"
SP="${SP:-0}"
TRAIN_ITERS="${TRAIN_ITERS:-100}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/moonlight_16b_qoft_int4}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
PROFILE_MEMORY="${PROFILE_MEMORY:-0}"
PROFILE_MEMORY_STEPS="${PROFILE_MEMORY_STEPS:-1}"
DEBUG_NAN="${DEBUG_NAN:-0}"
DEBUG_NAN_STEPS="${DEBUG_NAN_STEPS:-3}"
MEMORY_SMOKE_TEST="${MEMORY_SMOKE_TEST:-0}"
INT4_ACTIVE_EXPERT_CHUNK_SIZE="${INT4_ACTIVE_EXPERT_CHUNK_SIZE:-4}"
INT4_GROUPED_CHUNK_BACKEND="${INT4_GROUPED_CHUNK_BACKEND:-python}"
DEFAULT_LOG_INTERVAL=10
if [ "${MEMORY_SMOKE_TEST}" = "1" ]; then
    DEFAULT_LOG_INTERVAL=1
fi
LOG_INTERVAL="${LOG_INTERVAL:-${DEFAULT_LOG_INTERVAL}}"

EXTRA_ARGS=()
if [ "${SKIP_TRAIN}" = "1" ]; then
    EXTRA_ARGS+=(--skip-train)
fi
if [ "${SP}" = "1" ]; then
    EXTRA_ARGS+=(--sp)
fi
if [ "${PROFILE_MEMORY}" = "1" ]; then
    EXTRA_ARGS+=(--profile-memory --profile-memory-steps "${PROFILE_MEMORY_STEPS}")
fi
if [ "${DEBUG_NAN}" = "1" ]; then
    EXTRA_ARGS+=(--debug-nan --debug-nan-steps "${DEBUG_NAN_STEPS}")
fi
if [ "${MEMORY_SMOKE_TEST}" = "1" ]; then
    EXTRA_ARGS+=(--memory-smoke-test)
fi

if [ "${NUM_GPUS}" != "1" ] || [ "${TP}" != "1" ] || [ "${EP}" != "1" ] || [ "${PP}" != "1" ]; then
    if [ "${NUM_GPUS}" = "2" ] \
        && [ "${TP}" = "1" ] \
        && [ "${EP}" = "2" ] \
        && [ "${PP}" = "1" ]; then
        :
    elif [ "${NUM_GPUS}" = "4" ] \
        && [ "${TP}" = "2" ] \
        && [ "${EP}" = "2" ] \
        && [ "${PP}" = "1" ]; then
        :
    else
        echo "Current direct INT4 Moonlight checkpoints currently support:"
        echo "Supported normal mode: NUM_GPUS=1 TP=1 EP=1 PP=1"
        echo "Supported EP-sharded mode: NUM_GPUS=2 TP=1 EP=2 PP=1"
        echo "Supported mixed TP+EP mode: NUM_GPUS=4 TP=2 EP=2 PP=1"
        echo "Your values: NUM_GPUS=${NUM_GPUS} TP=${TP} EP=${EP} PP=${PP}"
        exit 1
    fi
fi

if [ "${TP}" != "1" ] && [ "${SP}" != "1" ]; then
    echo "Sequence parallelism is required for Moonlight MoE runs with TP>1."
    echo "Use: SP=1"
    echo "Your values: TP=${TP} SP=${SP}"
    exit 1
fi

echo "======================================"
echo "Moonlight-16B-A3B QOFT Fine-Tuning"
echo "  INT4 base weights (test model)"
echo "  GPUs: ${NUM_GPUS}, TP: ${TP}, EP: ${EP}, PP: ${PP}"
echo "  Sequence Parallel: ${SP}"
echo "  OFT Block Size: ${BLOCK_SIZE}"
echo "  INT4 Active Expert Chunk Size: ${INT4_ACTIVE_EXPERT_CHUNK_SIZE}"
echo "  INT4 Grouped Chunk Backend: ${INT4_GROUPED_CHUNK_BACKEND}"
echo "  Log Interval: ${LOG_INTERVAL}"
echo "  Megatron Checkpoint: ${MEGATRON_CKPT}"
echo "  HF Model Path: ${HF_MODEL_PATH}"
echo "  Output: ${OUTPUT_DIR}"
if [ "${MEMORY_SMOKE_TEST}" = "1" ]; then
    echo "  Mode: memory smoke test (1 train step, no save, no eval)"
fi
if [ "${NUM_GPUS}" = "2" ] && [ "${TP}" = "1" ] && [ "${EP}" = "2" ] && [ "${PP}" = "1" ]; then
    echo "  Mode: 2-GPU EP-sharded run"
fi
if [ "${NUM_GPUS}" = "4" ] && [ "${TP}" = "2" ] && [ "${EP}" = "2" ] && [ "${PP}" = "1" ]; then
    echo "  Mode: 4-GPU mixed TP+EP run"
fi
echo "======================================"

cd "${SCRIPT_DIR}"

torchrun --nproc_per_node="${NUM_GPUS}" \
    examples/models/moonlight_16b/finetune_qoft_int4.py \
    --pretrained-checkpoint "${MEGATRON_CKPT}" \
    --hf-model-path "${HF_MODEL_PATH}" \
    --tp "${TP}" --ep "${EP}" --pp "${PP}" \
    --train-iters "${TRAIN_ITERS}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --micro-batch-size "${MICRO_BATCH_SIZE}" \
    --seq-length "${SEQ_LENGTH}" \
    --output-dir "${OUTPUT_DIR}" \
    --block-size "${BLOCK_SIZE}" \
    --log-interval "${LOG_INTERVAL}" \
    --int4-active-expert-chunk-size "${INT4_ACTIVE_EXPERT_CHUNK_SIZE}" \
    --int4-grouped-chunk-backend "${INT4_GROUPED_CHUNK_BACKEND}" \
    "${EXTRA_ARGS[@]}"
