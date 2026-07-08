#!/bin/bash
#
# Slurm-Job: comparison_eval (Stewart-kompatible Pool-Size Evaluation)
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

#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/comparison_eval_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/comparison_eval_%j.err

set -euo pipefail

WORK_ROOT="/work2/ra39oxet-DatasetAudioSetSubset"
mkdir -p "$WORK_ROOT/logs"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT" || { echo "FEHLER: cd nach $REPO_ROOT fehlgeschlagen." >&2; exit 1; }

module purge
module load Python/3.11.5-GCCcore-13.2.0

if [[ ! -f "$HOME/venv/ba/bin/activate" ]]; then
  echo "FEHLER: Venv nicht gefunden ($HOME/venv/ba)." >&2
  exit 1
fi
source "$HOME/venv/ba/bin/activate"

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
python3 "$REPO_ROOT/pipeline/comparison_eval.py"
exit $?

