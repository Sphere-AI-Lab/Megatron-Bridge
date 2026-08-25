#!/bin/bash
# Convert HuggingFace INT4 model to Megatron format, preserving INT4.
#
# Expert weights stay in INT4 (4x compression).  Non-expert weights
# (attention, norms, embeddings) stay in BF16.
#
# Usage:
#   bash convert_int4_checkpoint.sh <hf_model> [megatron_path]
#
# Examples:
#   bash convert_int4_checkpoint.sh ${HF_MODEL_ROOT:-${HOME}/hf_models}/Kimi-K2.5
#   bash convert_int4_checkpoint.sh /path/to/Kimi-K2.5 ./checkpoints/Kimi-K2.5-INT4

set -euo pipefail

# CUDA setup
if command -v module >/dev/null 2>&1; then
    module load "${CUDA_MODULE:-cuda/13.2}" || true
fi
if command -v module >/dev/null 2>&1; then
    module load "${NCCL_MODULE:-nccl}" || true
fi
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH

# Library paths
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -n "${CUDNN_HOME:-}" ]]; then
    export LD_LIBRARY_PATH="${CUDNN_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

REPO=${MEGATRON_BRIDGE_ROOT:-${PWD}}
IN=${HF_MODEL_ROOT:-${HOME}/hf_models}/Moonlight-16B-A3B
OUT=${HF_MODEL_ROOT:-${HOME}/hf_models}/Moonlight-16B-A3B-INT4

# PYTHONPATH=$REPO/src:$REPO/3rdparty/Megatron-LM:$PYTHONPATH \
python $REPO/examples/conversion/quantize_to_int4.py \
  --input "$IN" \
  --output "$OUT" \
  --group-size 32