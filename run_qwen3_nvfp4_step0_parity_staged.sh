#!/usr/bin/env bash
# Stage Qwen3-30B-A3B-NVFP4 locally, then run VERL NVFP4 step-0 parity.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'USAGE'
Usage:
  bash run_qwen3_nvfp4_step0_parity_staged.sh [--dry-run]

This stages the HF NVFP4 model and converted Megatron checkpoint to local
storage, then runs:
  verl/scripts/check_nvfp4_step0_parity.py
against the staged paths.

Common environment overrides:
  HF_MODEL_SRC              Source HF model path.
  MEGATRON_DIST_SRC         Source Megatron checkpoint path.
  VERL_ROOT                 VERL checkout containing scripts/check_nvfp4_step0_parity.py.
  LOCAL_STAGE_ROOT          Stage root, usually _CONDOR_SCRATCH_DIR or TMPDIR.
  STAGE_HF_MODEL_TO         Staged HF destination.
  STAGE_MEGATRON_CKPT_TO    Staged Megatron destination.
  FORCE_STAGE_HF_MODEL=1    Refresh staged HF copy.
  FORCE_STAGE_MEGATRON_CKPT=1
  PROMPT_FILE               Prompt JSONL path.
  REPORT_JSON               Runtime parity JSON output.
  VERL_FP8_PARITY_OUTPUT_DEBUG=1
  VERL_SGLANG_DISABLE_FLASHINFER_AUTOTUNE=1
  VERL_SGLANG_DISABLE_CUDA_GRAPH=1
  VERL_SGLANG_MAX_START_WAIT_TIME=900
  VERL_SGLANG_TIMEOUT=120
USAGE
}

is_true() {
    case "${1,,}" in
        1 | true | yes | y | on) return 0 ;;
        *) return 1 ;;
    esac
}

resolve_local_stage_root() {
    if [[ -n "${_CONDOR_SCRATCH_DIR:-}" && -w "${_CONDOR_SCRATCH_DIR}" ]]; then
        printf '%s\n' "${_CONDOR_SCRATCH_DIR}"
    else
        printf '%s\n' "${TMPDIR:-/tmp}/megatron-bridge-${USER:-user}"
    fi
}

write_default_prompts() {
    local prompt_file="$1"
    mkdir -p "$(dirname "${prompt_file}")"
    cat >"${prompt_file}" <<'JSONL'
{"prompt":"Solve step by step: 19 * 23 ="}
{"prompt":"If a train travels 180 km in 3 hours, its average speed is"}
{"prompt":"Complete the sequence: 3, 9, 27,"}
{"prompt":"Write a Python expression for the sum of squares from 1 to 5:"}
{"prompt":"Explain in one sentence why the sky looks blue:"}
JSONL
}

DRY_RUN="${DRY_RUN:-0}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

HF_MODEL_SRC="${HF_MODEL_SRC:-${HF_MODEL_ROOT:-${HOME}/hf_models}/Qwen3-30B-A3B-NVFP4}"
MEGATRON_DIST_SRC="${MEGATRON_DIST_SRC:-${SCRIPT_DIR}/checkpoints/Qwen3-30B-A3B-NVFP4}"
VERL_ROOT="${VERL_ROOT:-${SCRIPT_DIR}/../verl}"
LOCAL_STAGE_ROOT="${LOCAL_STAGE_ROOT:-$(resolve_local_stage_root)}"
STAGE_HF_MODEL_TO="${STAGE_HF_MODEL_TO:-${LOCAL_STAGE_ROOT}/hf_models/Qwen3-30B-A3B-NVFP4}"
STAGE_MEGATRON_CKPT_TO="${STAGE_MEGATRON_CKPT_TO:-${LOCAL_STAGE_ROOT}/Megatron-Bridge/checkpoints/Qwen3-30B-A3B-NVFP4}"
STAGE_ENV_FILE="${STAGE_ENV_FILE:-/tmp/qwen3_30b_a3b_nvfp4_staged_paths.env}"
PROMPT_FILE="${PROMPT_FILE:-/tmp/qwen3_30b_a3b_nvfp4_step0_prompts.jsonl}"
REPORT_JSON="${REPORT_JSON:-${LOG_DIR:-${HOME}/log}/qwen3_nvfp4_step0_parity_report.json}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
VERL_FP8_PARITY_OUTPUT_DEBUG="${VERL_FP8_PARITY_OUTPUT_DEBUG:-1}"
VERL_SGLANG_DISABLE_FLASHINFER_AUTOTUNE="${VERL_SGLANG_DISABLE_FLASHINFER_AUTOTUNE:-1}"
VERL_SGLANG_DISABLE_CUDA_GRAPH="${VERL_SGLANG_DISABLE_CUDA_GRAPH:-1}"
VERL_SGLANG_MAX_START_WAIT_TIME="${VERL_SGLANG_MAX_START_WAIT_TIME:-900}"
VERL_SGLANG_TIMEOUT="${VERL_SGLANG_TIMEOUT:-120}"
REFRESH_PROMPTS="${REFRESH_PROMPTS:-1}"

if [[ ! -d "${VERL_ROOT}" ]]; then
    echo "Missing VERL_ROOT: ${VERL_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${VERL_ROOT}/scripts/check_nvfp4_step0_parity.py" ]]; then
    echo "Missing parity script: ${VERL_ROOT}/scripts/check_nvfp4_step0_parity.py" >&2
    exit 1
fi

STAGE_ARGS=(
    --hf-model "${HF_MODEL_SRC}"
    --megatron-dist "${MEGATRON_DIST_SRC}"
    --stage-root "${LOCAL_STAGE_ROOT}"
    --stage-hf-to "${STAGE_HF_MODEL_TO}"
    --stage-megatron-to "${STAGE_MEGATRON_CKPT_TO}"
    --env-file "${STAGE_ENV_FILE}"
)
if is_true "${FORCE_STAGE_HF_MODEL:-0}"; then
    STAGE_ARGS+=(--force-hf)
fi
if is_true "${FORCE_STAGE_MEGATRON_CKPT:-0}"; then
    STAGE_ARGS+=(--force-megatron)
fi
if is_true "${DRY_RUN}"; then
    STAGE_ARGS+=(--dry-run)
fi

bash "${SCRIPT_DIR}/stage_nvfp4_checkpoint_pair.sh" "${STAGE_ARGS[@]}"

# shellcheck source=/dev/null
source "${STAGE_ENV_FILE}"
STAGED_HF_MODEL="${HF_MODEL}"
STAGED_MEGATRON_DIST="${MEGATRON_DIST}"

if [[ ! -f "${PROMPT_FILE}" || "${REFRESH_PROMPTS}" == "1" ]]; then
    if is_true "${DRY_RUN}"; then
        echo "DRY_RUN=1: would write prompts to ${PROMPT_FILE}"
    else
        write_default_prompts "${PROMPT_FILE}"
    fi
fi

PARITY_CMD=(
    python scripts/check_nvfp4_step0_parity.py
    --hf-model "${STAGED_HF_MODEL}"
    --megatron-dist "${STAGED_MEGATRON_DIST}"
    --prompt-file "${PROMPT_FILE}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}"
    --report-json "${REPORT_JSON}"
)
if is_true "${VERL_SGLANG_DISABLE_FLASHINFER_AUTOTUNE}"; then
    PARITY_CMD+=(--sglang-disable-flashinfer-autotune)
fi
if is_true "${VERL_SGLANG_DISABLE_CUDA_GRAPH}"; then
    PARITY_CMD+=(--sglang-disable-cuda-graph)
fi
if [[ -n "${VERL_SGLANG_MAX_START_WAIT_TIME}" ]]; then
    PARITY_CMD+=(--sglang-max-start-wait-time "${VERL_SGLANG_MAX_START_WAIT_TIME}")
fi
if [[ -n "${VERL_SGLANG_TIMEOUT}" ]]; then
    PARITY_CMD+=(--sglang-timeout "${VERL_SGLANG_TIMEOUT}")
fi

echo "======================================"
echo "Running staged Qwen3 NVFP4 parity"
echo "  VERL_ROOT=${VERL_ROOT}"
echo "  HF=${STAGED_HF_MODEL}"
echo "  Megatron=${STAGED_MEGATRON_DIST}"
echo "  Prompt file=${PROMPT_FILE}"
echo "  Report JSON=${REPORT_JSON}"
echo "  Disable FlashInfer autotune=${VERL_SGLANG_DISABLE_FLASHINFER_AUTOTUNE}"
echo "  Disable CUDA graph=${VERL_SGLANG_DISABLE_CUDA_GRAPH}"
echo "  SGLang max start wait=${VERL_SGLANG_MAX_START_WAIT_TIME}"
echo "  SGLang timeout=${VERL_SGLANG_TIMEOUT}"
echo "======================================"

if is_true "${DRY_RUN}"; then
    printf "DRY_RUN=1: cd %q && VERL_FP8_PARITY_OUTPUT_DEBUG=%q " "${VERL_ROOT}" "${VERL_FP8_PARITY_OUTPUT_DEBUG}"
    printf "%q " "${PARITY_CMD[@]}"
    printf "\n"
    echo "For later training, reuse:"
    printf "  export HF_MODEL=%q\n" "${STAGED_HF_MODEL}"
    printf "  export MEGATRON_DIST=%q\n" "${STAGED_MEGATRON_DIST}"
    exit 0
fi

cd "${VERL_ROOT}"
export VERL_FP8_PARITY_OUTPUT_DEBUG
export VERL_SGLANG_DISABLE_FLASHINFER_AUTOTUNE
export VERL_SGLANG_DISABLE_CUDA_GRAPH
export VERL_SGLANG_MAX_START_WAIT_TIME
export VERL_SGLANG_TIMEOUT
"${PARITY_CMD[@]}"
