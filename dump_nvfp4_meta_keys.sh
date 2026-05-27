#!/bin/bash
# Dump the sharded_state_dict keys of a modelopt-NVFP4-wrapped Megatron meta model.
#
# Usage:
#   bash dump_nvfp4_meta_keys.sh [hf_model] [layer_match_regex] [extra python args...]
#
# Pass --dump-quantizers as an extra arg to also probe weight_quantizer/
# input_quantizer module buffers. Example:
#   bash dump_nvfp4_meta_keys.sh \
#       ${HF_MODEL_ROOT:-${HOME}/hf_models}/Qwen3-14B-NVFP4 \
#       'decoder\.layers\.1\.' \
#       --dump-quantizers

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/src:${SCRIPT_DIR}/3rdparty/Megatron-LM:${PYTHONPATH:-}"

UV_PROJECT="${UV_PROJECT:-${MEGATRON_BRIDGE_ROOT:-${PWD}}}"

HF_MODEL="${1:-${HF_MODEL_ROOT:-${HOME}/hf_models}/Kimi-K2.5-NVFP4}"
LAYER_MATCH="${2:-decoder\.layers\.1\.}"
if [ $# -ge 2 ]; then shift 2
elif [ $# -eq 1 ]; then shift 1
fi

echo "======================================"
echo "Dump NVFP4 meta sharded_state_dict keys"
echo "======================================"
echo "HF model:     ${HF_MODEL}"
echo "Layer match:  ${LAYER_MATCH}"
echo "======================================"

uv run --project "${UV_PROJECT}" python "${SCRIPT_DIR}/examples/conversion/dump_nvfp4_meta_keys.py" \
    --hf-model-path "${HF_MODEL}" \
    --layer-match "${LAYER_MATCH}" \
    "$@"
