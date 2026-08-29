#!/bin/bash
# Kimi-K2.5 QOFT finetuning — INT4 base weights + BF16 OFT adapters
#
# Two steps:
#   1. Convert HF INT4 checkpoint to Megatron format:
#        bash convert_int4_checkpoint_direct.sh \
#            /path/to/Kimi-K2.5 \
#            ./checkpoints/Kimi-K2.5-INT4
#
#   2. Finetune (this script):
#        bash run_qoft_finetune_kimi_k25_int4.sh
#
# Environment variables:
#   MEGATRON_CKPT  - Path to INT4 Megatron checkpoint (from step 1)
#   NUM_GPUS       - Number of GPUs (default: 8)
#   TP / EP        - Tensor / Expert parallel size (default: 2 / 4)
#   PP             - Pipeline parallel size (default: 1)
#   HF_MODEL_PATH  - HF model/tokenizer path (default: moonshotai/Kimi-K2.5)
#   TRAIN_ITERS    - Training iterations (default: 2000)
#   SEQ_LENGTH     - Sequence length / packed sequence size (default: 16384)
#   GLOBAL_BATCH_SIZE - Global batch size (default: 32)
#   MICRO_BATCH_SIZE  - Micro batch size per GPU (default: 1)
#   DISTRIBUTED_TIMEOUT_MINUTES - Process-group timeout for slow first-time dataset packing (default: 60)
#   OUTPUT_DIR     - Output directory
#   BLOCK_SIZE     - OFT block size (default: 32)
#   TARGET_MODULES - Comma/space separated OFT target modules
#   STAGE_HF_MODEL_TO - Local staged HF model path (set empty to disable)
#   STAGE_MEGATRON_CKPT_TO - Local staged Megatron checkpoint path (set empty to disable)
#   FORCE_STAGE_HF_MODEL / FORCE_STAGE_MEGATRON_CKPT - Set to 1 to refresh staged copies
#   SAVE_CHECKPOINTS - Set to 1 to save/resume run checkpoints (default: 0)
#   SAVE_INTERVAL  - Checkpoint save interval when SAVE_CHECKPOINTS=1 (default: 500)
#   PROFILE_MEMORY - Set to 1 to print model storage and CUDA memory around training steps
#   PROFILE_MEMORY_STEPS - Number of initial training steps to profile (default: 1)
#   SKIP_TRAIN     - Set to 1 to load/setup the model and exit without training
#   SKIP_EVAL      - Set to 1 to disable validation/evaluation after training

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

if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.wandb_key" && "${ENABLE_WANDB:-auto}" != "0" ]]; then
    export WANDB_API_KEY
    WANDB_API_KEY="$(<"${HOME}/.wandb_key")"
fi

# NCCL optimizations
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

# Ensure megatron.legacy is importable from the submodule source tree
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/3rdparty/Megatron-LM:${PYTHONPATH:-}"

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
    compgen -G "${model_dir}/*.safetensors" > /dev/null
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

stage_hf_model_if_requested() {
    if [[ -z "${STAGE_HF_MODEL_TO:-}" ]]; then
        return
    fi

    require_existing_path "HF_MODEL_PATH" "${HF_MODEL_PATH}"
    local hf_model_real
    local stage_hf_real
    hf_model_real="$(realpath -m "${HF_MODEL_PATH}")"
    stage_hf_real="$(realpath -m "${STAGE_HF_MODEL_TO}")"
    if [[ "${hf_model_real}" == "${stage_hf_real}" ]]; then
        echo "Skipping HF model staging: HF_MODEL_PATH already points to ${STAGE_HF_MODEL_TO}"
    elif ! is_true "${FORCE_STAGE_HF_MODEL:-0}" && model_dir_has_hf_weights "${STAGE_HF_MODEL_TO}"; then
        echo "Using existing staged HF model at ${STAGE_HF_MODEL_TO} (set FORCE_STAGE_HF_MODEL=1 to refresh)"
    else
        echo "Staging HF model from ${HF_MODEL_PATH} to ${STAGE_HF_MODEL_TO}"
        mkdir -p "${STAGE_HF_MODEL_TO}"
        rsync -ah --info=progress2 --delete "${HF_MODEL_PATH}/" "${STAGE_HF_MODEL_TO}/"
    fi
    HF_MODEL_PATH="${STAGE_HF_MODEL_TO}"
}

stage_megatron_checkpoint_if_requested() {
    if [[ -z "${STAGE_MEGATRON_CKPT_TO:-}" ]]; then
        return
    fi

    require_existing_path "MEGATRON_CKPT" "${MEGATRON_CKPT}"
    local megatron_ckpt_real
    local stage_megatron_real
    megatron_ckpt_real="$(realpath -m "${MEGATRON_CKPT}")"
    stage_megatron_real="$(realpath -m "${STAGE_MEGATRON_CKPT_TO}")"
    if [[ "${megatron_ckpt_real}" == "${stage_megatron_real}" ]]; then
        echo "Skipping Megatron checkpoint staging: MEGATRON_CKPT already points to ${STAGE_MEGATRON_CKPT_TO}"
    elif ! is_true "${FORCE_STAGE_MEGATRON_CKPT:-0}" && megatron_checkpoint_tree_has_dist_checkpoint "${STAGE_MEGATRON_CKPT_TO}"; then
        echo "Using existing staged Megatron checkpoint at ${STAGE_MEGATRON_CKPT_TO} (set FORCE_STAGE_MEGATRON_CKPT=1 to refresh)"
    else
        echo "Staging Megatron checkpoint from ${MEGATRON_CKPT} to ${STAGE_MEGATRON_CKPT_TO}"
        mkdir -p "${STAGE_MEGATRON_CKPT_TO}"
        rsync -ah --info=progress2 --delete "${MEGATRON_CKPT}/" "${STAGE_MEGATRON_CKPT_TO}/"
    fi
    MEGATRON_CKPT="${STAGE_MEGATRON_CKPT_TO}"
}

MEGATRON_CKPT="${MEGATRON_CKPT:-${MEGATRON_CKPT_ROOT:-${PWD}/checkpoints}/Kimi-K2.5}"
NUM_GPUS="${NUM_GPUS:-8}"
TP="${TP:-2}"
EP="${EP:-8}"
PP="${PP:-1}"
HF_MODEL_PATH="${HF_MODEL_PATH:-${HF_MODEL_ROOT:-${HOME}/hf_models}/Kimi-K2.5}"
TRAIN_ITERS="${TRAIN_ITERS:-2000}"
SEQ_LENGTH="${SEQ_LENGTH:-16384}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-60}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/kimi_k25_qoft_int4}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
TARGET_MODULES="${TARGET_MODULES:-linear_q_down_proj,linear_q_up_proj,linear_kv_down_proj,linear_kv_up_proj,linear_proj,linear_fc1,linear_fc2}"
LOCAL_STAGE_ROOT="${LOCAL_STAGE_ROOT:-$(resolve_local_stage_root)}"
STAGE_HF_MODEL_TO="${STAGE_HF_MODEL_TO-${LOCAL_STAGE_ROOT}/Kimi-K2.5}"
STAGE_MEGATRON_CKPT_TO="${STAGE_MEGATRON_CKPT_TO-${LOCAL_STAGE_ROOT}/Megatron-Bridge/checkpoints/Kimi-K2.5}"
FORCE_STAGE_HF_MODEL="${FORCE_STAGE_HF_MODEL:-0}"
FORCE_STAGE_MEGATRON_CKPT="${FORCE_STAGE_MEGATRON_CKPT:-0}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
PROFILE_MEMORY="${PROFILE_MEMORY:-0}"
PROFILE_MEMORY_STEPS="${PROFILE_MEMORY_STEPS:-1}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

EXTRA_ARGS=()
if [ "${SAVE_CHECKPOINTS}" = "1" ]; then
    EXTRA_ARGS+=(--save-checkpoints --save-interval "${SAVE_INTERVAL}")
fi
if [ "${PROFILE_MEMORY}" = "1" ]; then
    EXTRA_ARGS+=(--profile-memory --profile-memory-steps "${PROFILE_MEMORY_STEPS}")
fi
if [ "${SKIP_TRAIN}" = "1" ]; then
    EXTRA_ARGS+=(--skip-train)
fi
if [ "${SKIP_EVAL}" = "1" ]; then
    EXTRA_ARGS+=(--skip-eval)
fi

stage_hf_model_if_requested
stage_megatron_checkpoint_if_requested

echo "======================================"
echo "Kimi-K2.5 QOFT Fine-Tuning"
echo "  INT4 base weights (no BF16 alloc)"
echo "  GPUs: ${NUM_GPUS}, TP: ${TP}, EP: ${EP}, PP: ${PP}"
echo "  Seq Length: ${SEQ_LENGTH}"
echo "  Global Batch Size: ${GLOBAL_BATCH_SIZE}"
echo "  Micro Batch Size: ${MICRO_BATCH_SIZE}"
echo "  Distributed Timeout Minutes: ${DISTRIBUTED_TIMEOUT_MINUTES}"
echo "  OFT Block Size: ${BLOCK_SIZE}"
echo "  OFT Target Modules: ${TARGET_MODULES}"
echo "  Save Checkpoints: ${SAVE_CHECKPOINTS}"
echo "  Save Interval: ${SAVE_INTERVAL}"
echo "  Profile Memory: ${PROFILE_MEMORY}"
echo "  Profile Memory Steps: ${PROFILE_MEMORY_STEPS}"
echo "  Skip Train: ${SKIP_TRAIN}"
echo "  Skip Eval: ${SKIP_EVAL}"
echo "  Megatron Checkpoint: ${MEGATRON_CKPT}"
echo "  HF Model Path: ${HF_MODEL_PATH}"
echo "  Local Stage Root: ${LOCAL_STAGE_ROOT}"
echo "  Stage HF Model To: ${STAGE_HF_MODEL_TO:-<disabled>}"
echo "  Stage Megatron Checkpoint To: ${STAGE_MEGATRON_CKPT_TO:-<disabled>}"
echo "  Output: ${OUTPUT_DIR}"
echo "======================================"

torchrun --nproc_per_node="${NUM_GPUS}" \
    scripts/orbit/models/kimi_k25/finetune_qoft_int4.py \
    --pretrained-checkpoint "${MEGATRON_CKPT}" \
    --hf-model-path "${HF_MODEL_PATH}" \
    --tp "${TP}" --ep "${EP}" --pp "${PP}" \
    --train-iters "${TRAIN_ITERS}" \
    --seq-length "${SEQ_LENGTH}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --micro-batch-size "${MICRO_BATCH_SIZE}" \
    --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES}" \
    --output-dir "${OUTPUT_DIR}" \
    --block-size "${BLOCK_SIZE}" \
    --target-modules "${TARGET_MODULES}" \
    "${EXTRA_ARGS[@]}"
