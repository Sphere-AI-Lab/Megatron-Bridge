#!/usr/bin/env bash

set -xeuo pipefail

WORKSPACE=${WORKSPACE:-$(pwd)}
HF_MODEL_PATH=${HF_MODEL_PATH:-${HF_MODEL_ROOT:-${HOME}/hf_models}/Qwen3-14B-NVFP4}
MODEL_NAME=${MODEL_NAME:-Qwen3-14B-NVFP4}
MEGATRON_PATH=${MEGATRON_PATH:-${WORKSPACE}/checkpoints/${MODEL_NAME}}

bash convert_nvfp4_checkpoint_direct.sh "${HF_MODEL_PATH}" "${MEGATRON_PATH}"
