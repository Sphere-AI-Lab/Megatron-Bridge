#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Generic quantized-base QOFT launcher for FP8, INT4, and NVFP4 checkpoints.
# The architecture is detected from HF_MODEL_PATH. Run from any directory.
#
# Required: QUANT, HF_MODEL_PATH, MEGATRON_CKPT, NUM_GPUS
# Common: TP, EP, PP, SP, TRAIN_ITERS, GLOBAL_BATCH_SIZE,
# MICRO_BATCH_SIZE, SEQ_LENGTH, OUTPUT_DIR, OFT_TYPE, BLOCK_SIZE, TARGET_MODULES
# Advanced: DISTRIBUTED_TIMEOUT_MINUTES, LOG_INTERVAL, COFT, EPS, BLOCK_SHARE,
# MODULE_DROPOUT, GROUP_SIZE, INT4_ACTIVE_EXPERT_CHUNK_SIZE,
# INT4_GROUPED_CHUNK_BACKEND, SAVE_CHECKPOINTS, SAVE_INTERVAL, SKIP_TRAIN,
# SKIP_EVAL, PROFILE_MEMORY, PROFILE_MEMORY_STEPS, DEBUG_NAN, DEBUG_NAN_STEPS,
# MEMORY_SMOKE_TEST
#
# Pass extra entrypoint arguments after an optional `--`, for example:
#   bash scripts/orbit/run_qoft_finetune.sh -- --output-dir "/runs/INT4 smoke test"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if command -v module >/dev/null 2>&1; then
    module load "${CUDA_MODULE:-cuda/13.2}" || true
    module load "${NCCL_MODULE:-nccl}" || true
fi
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -n "${CUDNN_HOME:-}" ]]; then
    export LD_LIBRARY_PATH="${CUDNN_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

# Set WANDB_API_KEY in the environment or run `wandb login`; never store a key here.

: "${QUANT:?set QUANT=fp8|int4|nvfp4}"
: "${HF_MODEL_PATH:?set HF_MODEL_PATH to the HF model directory or id}"
: "${MEGATRON_CKPT:?set MEGATRON_CKPT to the converted Megatron checkpoint}"
: "${NUM_GPUS:?set NUM_GPUS to the positive number of local launcher processes}"

if [[ ! "${NUM_GPUS}" =~ ^0*[1-9][0-9]*$ ]]; then
    echo "NUM_GPUS must be a positive integer; received '${NUM_GPUS}'." >&2
    exit 1
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    echo "EXTRA_ARGS cannot preserve shell argument boundaries; pass extra arguments after -- instead." >&2
    exit 1
fi
[[ "${1:-}" == "--" ]] && shift

is_true() {
    case "$1" in
        1 | true | TRUE | True | yes | YES | Yes | y | Y | on | ON | On) return 0 ;;
        *) return 1 ;;
    esac
}

args=(
    --quant "${QUANT}"
    --hf-model-path "${HF_MODEL_PATH}"
    --pretrained-checkpoint "${MEGATRON_CKPT}"
)

[[ -n "${TP:-}" ]] && args+=(--tp "${TP}")
[[ -n "${EP:-}" ]] && args+=(--ep "${EP}")
[[ -n "${PP:-}" ]] && args+=(--pp "${PP}")
if [[ -n "${SP:-}" ]]; then
    if is_true "${SP}"; then
        args+=(--sp)
    else
        args+=(--no-sp)
    fi
fi
[[ -n "${DISTRIBUTED_TIMEOUT_MINUTES:-}" ]] && args+=(--distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES}")

[[ -n "${TRAIN_ITERS:-}" ]] && args+=(--train-iters "${TRAIN_ITERS}")
[[ -n "${GLOBAL_BATCH_SIZE:-}" ]] && args+=(--global-batch-size "${GLOBAL_BATCH_SIZE}")
[[ -n "${MICRO_BATCH_SIZE:-}" ]] && args+=(--micro-batch-size "${MICRO_BATCH_SIZE}")
[[ -n "${SEQ_LENGTH:-}" ]] && args+=(--seq-length "${SEQ_LENGTH}")
[[ -n "${LOG_INTERVAL:-}" ]] && args+=(--log-interval "${LOG_INTERVAL}")

[[ -n "${OFT_TYPE:-}" ]] && args+=(--oft-type "${OFT_TYPE}")
[[ -n "${BLOCK_SIZE:-}" ]] && args+=(--block-size "${BLOCK_SIZE}")
is_true "${COFT:-0}" && args+=(--coft)
[[ -n "${EPS:-}" ]] && args+=(--eps "${EPS}")
is_true "${BLOCK_SHARE:-0}" && args+=(--block-share)
[[ -n "${MODULE_DROPOUT:-}" ]] && args+=(--module-dropout "${MODULE_DROPOUT}")
[[ -n "${TARGET_MODULES:-}" ]] && args+=(--target-modules "${TARGET_MODULES}")

[[ -n "${GROUP_SIZE:-}" ]] && args+=(--group-size "${GROUP_SIZE}")
[[ -n "${INT4_ACTIVE_EXPERT_CHUNK_SIZE:-}" ]] && args+=(--int4-active-expert-chunk-size "${INT4_ACTIVE_EXPERT_CHUNK_SIZE}")
[[ -n "${INT4_GROUPED_CHUNK_BACKEND:-}" ]] && args+=(--int4-grouped-chunk-backend "${INT4_GROUPED_CHUNK_BACKEND}")

[[ -n "${OUTPUT_DIR:-}" ]] && args+=(--output-dir "${OUTPUT_DIR}")
if is_true "${SAVE_CHECKPOINTS:-0}"; then
    args+=(--save-checkpoints)
    [[ -n "${SAVE_INTERVAL:-}" ]] && args+=(--save-interval "${SAVE_INTERVAL}")
fi
is_true "${SKIP_TRAIN:-0}" && args+=(--skip-train)
is_true "${SKIP_EVAL:-0}" && args+=(--skip-eval)
if is_true "${PROFILE_MEMORY:-0}"; then
    args+=(--profile-memory)
    [[ -n "${PROFILE_MEMORY_STEPS:-}" ]] && args+=(--profile-memory-steps "${PROFILE_MEMORY_STEPS}")
fi
if is_true "${DEBUG_NAN:-0}"; then
    args+=(--debug-nan)
    [[ -n "${DEBUG_NAN_STEPS:-}" ]] && args+=(--debug-nan-steps "${DEBUG_NAN_STEPS}")
fi
is_true "${MEMORY_SMOKE_TEST:-0}" && args+=(--memory-smoke-test)

echo "======================================"
echo "QOFT finetuning (${QUANT})"
echo "  HF model:   ${HF_MODEL_PATH}"
echo "  checkpoint: ${MEGATRON_CKPT}"
echo "  GPUs:       ${NUM_GPUS}"
echo "======================================"

exec uv run --project "${REPO_ROOT}" python -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" scripts/orbit/finetune_qoft.py "${args[@]}" "$@"
