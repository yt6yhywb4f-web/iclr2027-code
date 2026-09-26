#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 CONDITION SEED [OUTPUT_ROOT]" >&2
    exit 2
fi

CONDITION="$1"
SEED="$2"
OUTPUT_ROOT="${3:-outputs/llama60_reconstruction}"

case "${CONDITION}" in
    carry) ABLATION=none ;;
    reset_m) ABLATION=reset_m ;;
    reset_v) ABLATION=reset_v ;;
    reset_all) ABLATION=reset_all ;;
    local_v_age) ABLATION=reset_v_local_v_age ;;
    *)
        echo "Unknown condition: ${CONDITION}" >&2
        exit 2
        ;;
esac

RUN_NAME="llama60_${CONDITION}_seed${SEED}"
SAVE_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
if [[ -e "${SAVE_DIR}" && -z "${CONTINUE_FROM:-}" ]]; then
    echo "Refusing to overwrite ${SAVE_DIR}" >&2
    exit 2
fi
if [[ -n "${CONTINUE_FROM:-}" && ! -d "${CONTINUE_FROM}" ]]; then
    echo "Checkpoint directory does not exist: ${CONTINUE_FROM}" >&2
    exit 2
fi

mkdir -p "${SAVE_DIR}"
cp configs/llama_60m_reconstruction.json "${SAVE_DIR}/scientific_config.json"

RUN_NAME="${RUN_NAME}" \
SAVE_DIR="${SAVE_DIR}" \
SEED="${SEED}" \
NUM_TRAINING_STEPS=10000 \
WARMUP_STEPS=1000 \
EVAL_EVERY=1000 \
SAVE_EVERY=10000 \
WORKERS=8 \
OPTIMIZER_STATE_ABLATION="${ABLATION}" \
EXPERIMENT_CONDITION="${CONDITION}" \
PYTHON_BIN="${PYTHON_BIN:-python3}" \
    bash scripts/benchmark_c4/llama_60m.sh
