#!/bin/bash
#
# Slurm-Job: comparison_eval
# Protocol A: stratifizierte Subsets à 1820 (avg über 2 Subsets)
# Protocol B: voller Test-Split (N=3668)
#
# Start:
#   sbatch jobs/comparison_eval.sh
#   sbatch --export=DATASET_RUN_NAME=...,TRAINING_RUN_DIR=...,AE_PAIR_RUN_DIR=...,AE_GENRE_RUN_DIR=... jobs/comparison_eval.sh
#

#SBATCH --job-name=comparison_eval
#SBATCH --partition=paula
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=0-6:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Jobs liegen unter <domain>/jobs/ → Repo-Root ist ../..
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/configs" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
fi
cd "$REPO_ROOT" || { echo "FEHLER: cd nach $REPO_ROOT fehlgeschlagen." >&2; exit 1; }
# shellcheck disable=SC1091
source "$REPO_ROOT/configs/cluster_env.sh"
mkdir -p "$REPO_ROOT/logs"

module purge
module load Python/3.11.5-GCCcore-13.2.0

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "FEHLER: Venv nicht gefunden ($VENV_ACTIVATE)." >&2
  exit 1
fi
source "$VENV_ACTIVATE"

echo "Hostname: $(hostname)"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "DATASET_RUN_NAME: ${DATASET_RUN_NAME:-<neuester Run>}"
echo "TRAINING_RUN_DIR: ${TRAINING_RUN_DIR:-<unset>}"
echo "AE_PAIR_RUN_DIR: ${AE_PAIR_RUN_DIR:-<unset>}"
echo "AE_GENRE_RUN_DIR: ${AE_GENRE_RUN_DIR:-<unset>}"
echo "COMPARISON_TARGET_TOTAL: ${COMPARISON_TARGET_TOTAL:-1820}"
echo "COMPARISON_PER_GENRE: ${COMPARISON_PER_GENRE:-<auto>}"
echo "COMPARISON_SUBSET_SEED: ${COMPARISON_SUBSET_SEED:-42}"

export PYTHONUNBUFFERED=1
python3 "$REPO_ROOT/eval/comparison_eval.py"
exit $?

