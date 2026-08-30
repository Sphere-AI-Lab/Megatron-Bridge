#!/usr/bin/env bash
# Stage an HF NVFP4 model and its Megatron dist checkpoint to node-local storage.
#
# Examples:
#   bash stage_nvfp4_checkpoint_pair.sh
#   bash stage_nvfp4_checkpoint_pair.sh --hf-only
#   bash stage_nvfp4_checkpoint_pair.sh \
#     --hf-model ${HF_MODEL_ROOT:-${HOME}/hf_models}/Qwen3-30B-A3B-NVFP4 \
#     --megatron-dist ${MEGATRON_CKPT_ROOT:-${PWD}/checkpoints}/Qwen3-30B-A3B-NVFP4 \
#     --env-file /tmp/qwen3_30b_a3b_nvfp4_staged_paths.env

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

DEFAULT_HF_MODEL="${HF_MODEL_ROOT:-${HOME}/hf_models}/Kimi-K2.5-NVFP4"
DEFAULT_MEGATRON_DIST="${SCRIPT_DIR}/checkpoints/Kimi-K2.5-NVFP4"

usage() {
    cat <<USAGE
Usage:
  bash stage_nvfp4_checkpoint_pair.sh [options]

Stages checkpoint trees with rsync, reusing valid existing staged copies unless
forced. By default this stages the Kimi-K2.5-NVFP4 HF model and the matching
Megatron checkpoint under this Megatron-Bridge checkout.

Options:
  --hf-model PATH          HF model source directory.
                           Default: ${DEFAULT_HF_MODEL}
  --megatron-dist PATH     Megatron checkpoint source directory.
                           Default: ${DEFAULT_MEGATRON_DIST}
  --stage-root PATH        Local stage root.
                           Default: _CONDOR_SCRATCH_DIR if writable,
                           otherwise ${TMPDIR:-/tmp}/megatron-bridge-$USER.
  --stage-hf-to PATH       HF model stage destination.
  --stage-megatron-to PATH Megatron checkpoint stage destination.
  --hf-only                Stage only the HF model.
  --megatron-only          Stage only the Megatron checkpoint.
  --force                  Refresh both staged copies.
  --force-hf               Refresh only the staged HF copy.
  --force-megatron         Refresh only the staged Megatron copy.
  --dry-run                Print rsync actions without copying.
  --env-file PATH          Write shell exports for staged paths to PATH.
  -h, --help               Show this help.

Environment overrides:
  HF_MODEL, MEGATRON_DIST, LOCAL_STAGE_ROOT, STAGE_HF_MODEL_TO,
  STAGE_MEGATRON_CKPT_TO, FORCE_STAGE_HF_MODEL, FORCE_STAGE_MEGATRON_CKPT,
  DRY_RUN, STAGE_ENV_FILE.
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

model_dir_has_hf_weights() {
    local model_dir="$1"
    [[ -d "${model_dir}" ]] || return 1
    [[ -f "${model_dir}/config.json" ]] || return 1
    [[ -f "${model_dir}/model.safetensors.index.json" ]] && return 0
    compgen -G "${model_dir}/*.safetensors" >/dev/null
}

resolve_megatron_checkpoint_dir() {
    local ckpt_dir="$1"
    local latest_iter
    local candidate

    if [[ -f "${ckpt_dir}/metadata.json" || -f "${ckpt_dir}/.metadata" ]]; then
        printf '%s\n' "${ckpt_dir}"
        return
    fi

    if [[ -f "${ckpt_dir}/latest_checkpointed_iteration.txt" ]]; then
        latest_iter="$(<"${ckpt_dir}/latest_checkpointed_iteration.txt")"
        if [[ "${latest_iter}" =~ ^[0-9]+$ ]]; then
            printf -v candidate '%s/iter_%07d' "${ckpt_dir}" "${latest_iter}"
            if [[ -f "${candidate}/metadata.json" || -f "${candidate}/.metadata" ]]; then
                printf '%s\n' "${candidate}"
                return
            fi
        elif [[ "${latest_iter}" == "release" ]]; then
            candidate="${ckpt_dir}/release"
            if [[ -f "${candidate}/metadata.json" || -f "${candidate}/.metadata" ]]; then
                printf '%s\n' "${candidate}"
                return
            fi
        fi
    fi

    printf '%s\n' "${ckpt_dir}"
}

megatron_checkpoint_tree_has_dist_checkpoint() {
    local ckpt_tree="$1"
    local resolved_ckpt_dir

    [[ -d "${ckpt_tree}" ]] || return 1
    resolved_ckpt_dir="$(resolve_megatron_checkpoint_dir "${ckpt_tree}")"
    [[ -d "${resolved_ckpt_dir}" ]] || return 1
    [[ -f "${resolved_ckpt_dir}/metadata.json" || -f "${resolved_ckpt_dir}/.metadata" ]]
}

require_existing_path() {
    local path_name="$1"
    local path_value="$2"
    if [[ ! -e "${path_value}" ]]; then
        echo "Missing required ${path_name}: ${path_value}" >&2
        exit 1
    fi
}

run_rsync_tree() {
    local source_dir="$1"
    local dest_dir="$2"

    if is_true "${DRY_RUN:-0}"; then
        echo "DRY_RUN=1: rsync -ah --info=progress2 --delete ${source_dir}/ ${dest_dir}/"
        return
    fi

    mkdir -p "${dest_dir}"
    rsync -ah --info=progress2 --delete "${source_dir}/" "${dest_dir}/"
}

stage_hf_model() {
    require_existing_path "HF_MODEL" "${HF_MODEL}"

    local hf_model_real
    local stage_hf_real
    hf_model_real="$(realpath -m "${HF_MODEL}")"
    stage_hf_real="$(realpath -m "${STAGE_HF_MODEL_TO}")"

    if [[ "${hf_model_real}" == "${stage_hf_real}" ]]; then
        echo "Skipping HF model staging: HF_MODEL already points to ${STAGE_HF_MODEL_TO}"
    elif ! is_true "${FORCE_STAGE_HF_MODEL:-0}" && model_dir_has_hf_weights "${STAGE_HF_MODEL_TO}"; then
        echo "Using existing staged HF model at ${STAGE_HF_MODEL_TO} (set FORCE_STAGE_HF_MODEL=1 to refresh)"
    else
        echo "Staging HF model from ${HF_MODEL} to ${STAGE_HF_MODEL_TO}"
        run_rsync_tree "${HF_MODEL}" "${STAGE_HF_MODEL_TO}"
    fi

    HF_MODEL="${STAGE_HF_MODEL_TO}"
}

stage_megatron_checkpoint() {
    require_existing_path "MEGATRON_DIST" "${MEGATRON_DIST}"

    local megatron_dist_real
    local stage_megatron_real
    megatron_dist_real="$(realpath -m "${MEGATRON_DIST}")"
    stage_megatron_real="$(realpath -m "${STAGE_MEGATRON_CKPT_TO}")"

    if [[ "${megatron_dist_real}" == "${stage_megatron_real}" ]]; then
        echo "Skipping Megatron checkpoint staging: MEGATRON_DIST already points to ${STAGE_MEGATRON_CKPT_TO}"
    elif ! is_true "${FORCE_STAGE_MEGATRON_CKPT:-0}" && megatron_checkpoint_tree_has_dist_checkpoint "${STAGE_MEGATRON_CKPT_TO}"; then
        echo "Using existing staged Megatron checkpoint at ${STAGE_MEGATRON_CKPT_TO} (set FORCE_STAGE_MEGATRON_CKPT=1 to refresh)"
    else
        echo "Staging Megatron checkpoint from ${MEGATRON_DIST} to ${STAGE_MEGATRON_CKPT_TO}"
        run_rsync_tree "${MEGATRON_DIST}" "${STAGE_MEGATRON_CKPT_TO}"
    fi

    MEGATRON_DIST="${STAGE_MEGATRON_CKPT_TO}"
}

HF_MODEL="${HF_MODEL:-${DEFAULT_HF_MODEL}}"
MEGATRON_DIST="${MEGATRON_DIST:-${DEFAULT_MEGATRON_DIST}}"
LOCAL_STAGE_ROOT="${LOCAL_STAGE_ROOT:-$(resolve_local_stage_root)}"
STAGE_HF=1
STAGE_MEGATRON=1
FORCE_STAGE_HF_MODEL="${FORCE_STAGE_HF_MODEL:-0}"
FORCE_STAGE_MEGATRON_CKPT="${FORCE_STAGE_MEGATRON_CKPT:-0}"
DRY_RUN="${DRY_RUN:-0}"
STAGE_ENV_FILE="${STAGE_ENV_FILE:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf-model)
            HF_MODEL="$2"
            shift 2
            ;;
        --megatron-dist | --megatron-ckpt)
            MEGATRON_DIST="$2"
            shift 2
            ;;
        --stage-root)
            LOCAL_STAGE_ROOT="$2"
            shift 2
            ;;
        --stage-hf-to)
            STAGE_HF_MODEL_TO="$2"
            shift 2
            ;;
        --stage-megatron-to)
            STAGE_MEGATRON_CKPT_TO="$2"
            shift 2
            ;;
        --hf-only)
            STAGE_HF=1
            STAGE_MEGATRON=0
            shift
            ;;
        --megatron-only)
            STAGE_HF=0
            STAGE_MEGATRON=1
            shift
            ;;
        --force)
            FORCE_STAGE_HF_MODEL=1
            FORCE_STAGE_MEGATRON_CKPT=1
            shift
            ;;
        --force-hf)
            FORCE_STAGE_HF_MODEL=1
            shift
            ;;
        --force-megatron)
            FORCE_STAGE_MEGATRON_CKPT=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --env-file)
            STAGE_ENV_FILE="$2"
            shift 2
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

HF_MODEL_NAME="$(basename "${HF_MODEL}")"
MEGATRON_CKPT_NAME="$(basename "${MEGATRON_DIST}")"
STAGE_HF_MODEL_TO="${STAGE_HF_MODEL_TO:-${LOCAL_STAGE_ROOT}/hf_models/${HF_MODEL_NAME}}"
STAGE_MEGATRON_CKPT_TO="${STAGE_MEGATRON_CKPT_TO:-${LOCAL_STAGE_ROOT}/Megatron-Bridge/checkpoints/${MEGATRON_CKPT_NAME}}"

echo "======================================"
echo "NVFP4 checkpoint staging"
echo "  stage root:       ${LOCAL_STAGE_ROOT}"
echo "  stage HF:         ${STAGE_HF}"
echo "  stage Megatron:   ${STAGE_MEGATRON}"
echo "  dry run:          ${DRY_RUN}"
echo "======================================"

if is_true "${STAGE_HF}"; then
    stage_hf_model
fi

if is_true "${STAGE_MEGATRON}"; then
    stage_megatron_checkpoint
fi

echo "======================================"
echo "Staged paths"
if is_true "${STAGE_HF}"; then
    echo "  HF_MODEL=${HF_MODEL}"
fi
if is_true "${STAGE_MEGATRON}"; then
    echo "  MEGATRON_DIST=${MEGATRON_DIST}"
    echo "  MEGATRON_DIST_RESOLVED=$(resolve_megatron_checkpoint_dir "${MEGATRON_DIST}")"
fi
echo "======================================"
echo "To reuse in parity commands:"
if is_true "${STAGE_HF}"; then
    printf "  export HF_MODEL=%q\n" "${HF_MODEL}"
fi
if is_true "${STAGE_MEGATRON}"; then
    printf "  export MEGATRON_DIST=%q\n" "${MEGATRON_DIST}"
fi

if [[ -n "${STAGE_ENV_FILE}" ]]; then
    mkdir -p "$(dirname "${STAGE_ENV_FILE}")"
    : >"${STAGE_ENV_FILE}"
    if is_true "${STAGE_HF}"; then
        printf "export HF_MODEL=%q\n" "${HF_MODEL}" >>"${STAGE_ENV_FILE}"
    fi
    if is_true "${STAGE_MEGATRON}"; then
        printf "export MEGATRON_DIST=%q\n" "${MEGATRON_DIST}" >>"${STAGE_ENV_FILE}"
        printf "export MEGATRON_DIST_RESOLVED=%q\n" "$(resolve_megatron_checkpoint_dir "${MEGATRON_DIST}")" >>"${STAGE_ENV_FILE}"
    fi
    echo "Wrote staged path exports to ${STAGE_ENV_FILE}"
fi
