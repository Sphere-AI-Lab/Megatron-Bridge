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

# Generic PEFT finetuning launcher.
#
# Replaces run_lora_finetune.sh / run_oft_finetune.sh (which differed only in
# their target .py) and covers any model whose recipe is "_peft_common() plus an
# HF path" -- see scripts/orbit/finetune_peft.py.
#
# Usage:
#   bash scripts/orbit/run_peft_finetune.sh                       # oft, Llama-3.2-1B
#   PEFT=lora bash scripts/orbit/run_peft_finetune.sh
#   PEFT=oft QUANT=nvfp4 HF_MODEL=Qwen/Qwen3-14B NUM_GPUS=8 TP=1 \
#       MEGATRON_CKPT=./checkpoints/Qwen3-14B-NVFP4 \
#       bash scripts/orbit/run_peft_finetune.sh
#
# Environment variables:
#   PEFT           - oft | lora | dora | none            (default: oft)
#   QUANT          - none | fp8 | mxfp8 | nvfp4          (default: none)
#   HF_MODEL       - HF model id or local path           (default: meta-llama/Llama-3.2-1B)
#   MEGATRON_CKPT  - Megatron checkpoint to finetune from
#                    (default: ./checkpoints/<basename of HF_MODEL>)
#   NUM_GPUS       - local distributed process count     (default: 1)
#   TP / PP / EP / CP - parallelism sizes                (default: 1)
#   OUTPUT_DIR     - run directory                       (default: chosen by the entrypoint)
#
# Pass extra entrypoint arguments after an optional `--`, for example:
#   bash scripts/orbit/run_peft_finetune.sh -- --train-iters 20 \
#       --output-dir "/runs/OFT smoke test"
#
# INT4 is not covered here: its direct-checkpoint runtime needs the Orbit
# load-patch stack. Use run_qoft_finetune.sh with QUANT=int4.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# CUDA / cuDNN / NCCL paths
if command -v module >/dev/null 2>&1; then
    module load "${CUDA_MODULE:-cuda/13.2}" || true
    module load "${NCCL_MODULE:-nccl}" || true
fi
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -n "${CUDNN_HOME:-}" ]]; then
    export LD_LIBRARY_PATH="${CUDNN_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# W&B API key: set WANDB_API_KEY in the environment or run wandb login beforehand.
# Never commit a key into this file.

PEFT="${PEFT:-oft}"
QUANT="${QUANT:-none}"
HF_MODEL="${HF_MODEL:-meta-llama/Llama-3.2-1B}"
MEGATRON_CKPT="${MEGATRON_CKPT:-./checkpoints/$(basename "${HF_MODEL}")}"
NUM_GPUS="${NUM_GPUS:-1}"
TP="${TP:-1}"; PP="${PP:-1}"; EP="${EP:-1}"; CP="${CP:-1}"

if [[ ! "${NUM_GPUS}" =~ ^0*[1-9][0-9]*$ ]]; then
    echo "NUM_GPUS must be a positive integer; received '${NUM_GPUS}'." >&2
    exit 1
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    echo "EXTRA_ARGS cannot preserve shell argument boundaries; pass extra arguments after -- instead." >&2
    exit 1
fi
[[ "${1:-}" == "--" ]] && shift

args=(
    --model-path "${HF_MODEL}"
    --pretrained-checkpoint "${MEGATRON_CKPT}"
    --peft "${PEFT}"
    --quant "${QUANT}"
    --tp "${TP}" --pp "${PP}" --ep "${EP}" --cp "${CP}"
)
if [[ -n "${OUTPUT_DIR:-}" ]]; then
    args+=(--output-dir "${OUTPUT_DIR}")
fi

echo "Starting ${PEFT} finetuning of ${HF_MODEL} (quant=${QUANT}) on ${NUM_GPUS} GPU(s)..."
echo "  checkpoint: ${MEGATRON_CKPT}"
echo "  TP=${TP} PP=${PP} EP=${EP} CP=${CP}"

exec uv run --project "${REPO_ROOT}" python -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
    scripts/orbit/finetune_peft.py \
    "${args[@]}" \
    "$@"
