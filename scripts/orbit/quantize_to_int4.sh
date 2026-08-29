#!/bin/bash
# Quantize a BF16 HuggingFace model's expert weights to Kimi-K2 native INT4 format.
#
# Usage:
#   bash quantize_to_int4.sh <input_model> [output_model]
#
# Examples:
#   bash quantize_to_int4.sh ${HF_MODEL_ROOT:-${HOME}/hf_models}/Moonlight-16B-A3B
#   bash quantize_to_int4.sh /path/to/model /path/to/model-INT4

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

if [ $# -lt 1 ]; then
    echo "Usage: bash quantize_to_int4.sh <input_model> [output_model]"
    echo ""
    echo "Quantizes expert MLP weights to INT4 (Kimi-K2 native format)."
    echo "Non-expert weights (attention, norms, embeddings) stay in BF16."
    exit 1
fi

INPUT="$1"

if [ $# -ge 2 ]; then
    OUTPUT="$2"
else
    OUTPUT="${INPUT}-INT4"
fi

echo "======================================"
echo "BF16 -> INT4 Expert Quantization"
echo "======================================"
echo "Input:  ${INPUT}"
echo "Output: ${OUTPUT}"
echo "======================================"

python scripts/orbit/conversion/quantize_to_int4.py \
    --input "${INPUT}" \
    --output "${OUTPUT}"

echo "======================================"
echo "Done. INT4 model saved to: ${OUTPUT}"
echo ""
echo "Next: convert to Megatron format:"
echo "  bash convert_int4_checkpoint_direct.sh ${OUTPUT}"
echo "======================================"
