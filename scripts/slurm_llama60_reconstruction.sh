#!/usr/bin/env bash
#SBATCH --job-name=galore-reconstruction
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${CONDITION:?Set CONDITION to carry, reset_m, reset_v, reset_all, or local_v_age}"
: "${SEED:?Set SEED to 0, 2, or 42}"

mkdir -p logs
bash scripts/run_llama60_reconstruction.sh "${CONDITION}" "${SEED}"

