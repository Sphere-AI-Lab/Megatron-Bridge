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

# Compare tensor metadata between a source HuggingFace checkpoint and a converted Megatron checkpoint.
# Inspect only trusted checkpoints; checkpoint metadata can use Python serialization.
#
# Usage:
#   bash compare_converted_model_metadata.sh <hf_model> <megatron_path> [extra args...]

set -euo pipefail

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/src:${SCRIPT_DIR}/3rdparty/Megatron-LM:${PYTHONPATH:-}"

# modelopt / megatron dependencies live in the rl_infra uv project.
UV_PROJECT="${UV_PROJECT:-${MEGATRON_BRIDGE_ROOT:-${PWD}}}"

if [ $# -lt 2 ]; then
    echo "Usage: bash compare_converted_model_metadata.sh <hf_model> <megatron_path> [extra args...]"
    echo ""
    echo "Arguments:"
    echo "  hf_model       Path to the source HuggingFace checkpoint"
    echo "  megatron_path  Path to the converted Megatron checkpoint directory"
    echo ""
    echo "Extra args are forwarded to scripts/orbit/conversion/compare_model_metadata.py"
    exit 1
fi

HF_MODEL="$1"
MEGATRON_PATH="$2"
shift 2

echo "=============================================="
echo "HF -> Megatron Metadata Comparison"
echo "=============================================="
echo "Source HF model:      ${HF_MODEL}"
echo "Megatron checkpoint:  ${MEGATRON_PATH}"
echo "=============================================="

uv run --project "${UV_PROJECT}" python scripts/orbit/conversion/compare_model_metadata.py \
    --hf-model-path "${HF_MODEL}" \
    --megatron-path "${MEGATRON_PATH}" \
    "$@"
