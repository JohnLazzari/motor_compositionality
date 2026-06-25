#!/usr/bin/env bash
set -euo pipefail

JOBS=(
  train_rnn256_softplus_reach
  train_rnn256_softplus_clkcurvedreach
  train_rnn256_softplus_cclkcurvedreach
  train_rnn256_softplus_sinusoid
  train_rnn256_softplus_invsinusoid
  train_rnn256_softplus_reachback
  train_rnn256_softplus_clkcycle
  train_rnn256_softplus_cclkcycle
  train_rnn256_softplus_figure8
  train_rnn256_softplus_invfigure8
  train_rnn256_softplus_reach_nofeedback
  train_rnn256_softplus_clkcurvedreach_nofeedback
  train_rnn256_softplus_cclkcurvedreach_nofeedback
  train_rnn256_softplus_sinusoid_nofeedback
  train_rnn256_softplus_invsinusoid_nofeedback
  train_rnn256_softplus_reachback_nofeedback
  train_rnn256_softplus_clkcycle_nofeedback
  train_rnn256_softplus_cclkcycle_nofeedback
  train_rnn256_softplus_figure8_nofeedback
  train_rnn256_softplus_invfigure8_nofeedback
)

LOG_DIR="${LOG_DIR:-logs/single_task_training}"
JOB_PREFIX="${JOB_PREFIX:-single_task}"
SBATCH_TIME="${SBATCH_TIME:-24:00:00}"
SBATCH_CPUS="${SBATCH_CPUS:-1}"
SBATCH_MEM="${SBATCH_MEM:-8G}"
SBATCH_PARTITION="${SBATCH_PARTITION:-}"
SBATCH_EXTRA="${SBATCH_EXTRA:-}"

list_jobs() {
  for i in "${!JOBS[@]}"; do
    printf '%02d %s\n' "$i" "${JOBS[$i]}"
  done
}

submit_job() {
  local idx="$1"
  local job="${JOBS[$idx]}"
  local sbatch_args=(
    --job-name "${JOB_PREFIX}_${idx}_${job}"
    --output "${LOG_DIR}/%x_%j.out"
    --error "${LOG_DIR}/%x_%j.err"
    --time "${SBATCH_TIME}"
    --cpus-per-task "${SBATCH_CPUS}"
    --mem "${SBATCH_MEM}"
  )

  if [[ -n "${SBATCH_PARTITION}" ]]; then
    sbatch_args+=(--partition "${SBATCH_PARTITION}")
  fi

  if [[ -n "${SBATCH_EXTRA}" ]]; then
    # shellcheck disable=SC2206
    sbatch_args+=(${SBATCH_EXTRA})
  fi

  echo "Submitting ${idx}: ${job}"
  sbatch "${sbatch_args[@]}" --wrap "cd '${PWD}' && python experiments/run_train.py --experiment '${job}'"
}

if [[ "${1:-}" == "--list" ]]; then
  list_jobs
  exit 0
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  list_jobs
  exit 0
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found. This script submits jobs to SLURM and must be run on a SLURM login node." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

if [[ -n "${1:-}" ]]; then
  submit_job "$1"
  exit 0
fi

for i in "${!JOBS[@]}"; do
  submit_job "$i"
done
