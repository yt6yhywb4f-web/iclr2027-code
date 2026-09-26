#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_NAME="${RUN_NAME:-llama60_galore}"
SAVE_DIR="${SAVE_DIR:?SAVE_DIR must be set}"
SEED="${SEED:-42}"
NUM_TRAINING_STEPS="${NUM_TRAINING_STEPS:-10000}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
WORKERS="${WORKERS:-8}"
OPTIMIZER_STATE_ABLATION="${OPTIMIZER_STATE_ABLATION:-none}"
REFRESH_V_GAMMA="${REFRESH_V_GAMMA:-}"
REFRESH_EVENTS_CSV="${REFRESH_EVENTS_CSV:-}"
REFRESH_LR_MULTIPLIER="${REFRESH_LR_MULTIPLIER:-1.0}"
REFRESH_LR_MULTIPLIERS="${REFRESH_LR_MULTIPLIERS:-}"
REFRESH_LR_WINDOW="${REFRESH_LR_WINDOW:-0}"
REFRESH_LR_SCHEDULE_JSON="${REFRESH_LR_SCHEDULE_JSON:-}"
REFRESH_LR_SCHEDULE_KIND="${REFRESH_LR_SCHEDULE_KIND:-spike}"
REFRESH_LR_CONTROL_CSV="${REFRESH_LR_CONTROL_CSV:-}"
TRAINING_METRICS_JSONL="${TRAINING_METRICS_JSONL:-}"
STATE_CHECKSUMS_JSON="${STATE_CHECKSUMS_JSON:-}"
RNG_CHECKSUMS_JSON="${RNG_CHECKSUMS_JSON:-}"
EXPERIMENT_CONDITION="${EXPERIMENT_CONDITION:-${RUN_NAME}}"
PROJECTOR_SAVE_DIR="${PROJECTOR_SAVE_DIR:-}"
PROJECTOR_LOAD_DIR="${PROJECTOR_LOAD_DIR:-}"
FREEZE_PROJECTOR_AFTER_INITIAL="${FREEZE_PROJECTOR_AFTER_INITIAL:-false}"

OPTIONAL_ARGS=()
[[ -n "${CONTINUE_FROM:-}" ]] && OPTIONAL_ARGS+=(--continue_from "${CONTINUE_FROM}")
[[ -n "${STOP_AFTER_UPDATES:-}" ]] && OPTIONAL_ARGS+=(--stop_after_updates "${STOP_AFTER_UPDATES}")
[[ -n "${REFRESH_V_GAMMA}" ]] && OPTIONAL_ARGS+=(--refresh_v_gamma "${REFRESH_V_GAMMA}")
[[ -n "${REFRESH_EVENTS_CSV}" ]] && OPTIONAL_ARGS+=(--refresh_events_csv "${REFRESH_EVENTS_CSV}")
[[ -n "${REFRESH_LR_MULTIPLIERS}" ]] && OPTIONAL_ARGS+=(--refresh_lr_multipliers "${REFRESH_LR_MULTIPLIERS}")
[[ -n "${REFRESH_LR_SCHEDULE_JSON}" ]] && OPTIONAL_ARGS+=(--refresh_lr_schedule_json "${REFRESH_LR_SCHEDULE_JSON}")
[[ -n "${REFRESH_LR_CONTROL_CSV}" ]] && OPTIONAL_ARGS+=(--refresh_lr_control_csv "${REFRESH_LR_CONTROL_CSV}")
[[ -n "${TRAINING_METRICS_JSONL}" ]] && OPTIONAL_ARGS+=(--training_metrics_jsonl "${TRAINING_METRICS_JSONL}")
[[ -n "${STATE_CHECKSUMS_JSON}" ]] && OPTIONAL_ARGS+=(--state_checksums_json "${STATE_CHECKSUMS_JSON}")
[[ -n "${RNG_CHECKSUMS_JSON}" ]] && OPTIONAL_ARGS+=(--rng_checksums_json "${RNG_CHECKSUMS_JSON}")
[[ -n "${PROJECTOR_SAVE_DIR}" ]] && OPTIONAL_ARGS+=(--projector_save_dir "${PROJECTOR_SAVE_DIR}")
[[ -n "${PROJECTOR_LOAD_DIR}" ]] && OPTIONAL_ARGS+=(--projector_load_dir "${PROJECTOR_LOAD_DIR}")
case "${FREEZE_PROJECTOR_AFTER_INITIAL}" in
    [Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss]) OPTIONAL_ARGS+=(--freeze_projector_after_initial) ;;
    [Ff][Aa][Ll][Ss][Ee]|0|[Nn][Oo]) ;;
    *) echo "FREEZE_PROJECTOR_AFTER_INITIAL must be true or false" >&2; exit 2 ;;
esac

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=1 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --lr 0.01 \
    --galore_scale 0.25 \
    --rank 128 \
    --update_proj_gap 200 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps "${NUM_TRAINING_STEPS}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every "${EVAL_EVERY}" \
    --save_every "${SAVE_EVERY}" \
    --workers "${WORKERS}" \
    --seed "${SEED}" \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-6 \
    --optimizer galore_adamw \
    --optimizer_state_ablation "${OPTIMIZER_STATE_ABLATION}" \
    --refresh_lr_multiplier "${REFRESH_LR_MULTIPLIER}" \
    --refresh_lr_window "${REFRESH_LR_WINDOW}" \
    --refresh_lr_schedule_kind "${REFRESH_LR_SCHEDULE_KIND}" \
    --experiment_condition "${EXPERIMENT_CONDITION}" \
    --name "${RUN_NAME}" \
    --save_dir "${SAVE_DIR}" \
    "${OPTIONAL_ARGS[@]}" \
    "$@"
