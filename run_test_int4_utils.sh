#!/bin/bash
# Run the INT4 checkpoint-loading unit tests with the cluster CUDA/NCCL env.
#
# Usage:
#   bash run_test_int4_utils.sh
#   bash run_test_int4_utils.sh -k swiglu -vv

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

cd "${SCRIPT_DIR}"

pytest tests/unit_tests/peft/test_int4_utils.py -q "$@"
