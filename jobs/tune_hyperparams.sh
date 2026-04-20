#!/bin/bash
#
# Slurm-Job: Hyperparameter-Tuning (Pair -> Genre).
# Ressourcen wie bei Trainingsjobs, optional mit mehreren Tuning-Workern.
#
# Nutzung (vom Repo-Root auf dem Cluster):
#   sbatch jobs/tune_hyperparams.sh
#   sbatch --export=DATASET_RUN_NAME=2026-03-13_18-12-31_audioset jobs/tune_hyperparams.sh
#   sbatch --export=DATASET_RUN_NAME=...,HP_TUNE_WORKERS=2,HP_BATCH_SIZE=64,HP_MAX_EPOCHS=20,HP_PATIENCE=3 jobs/tune_hyperparams.sh
#

#SBATCH --job-name=tuning
#SBATCH --partition=paula
#SBATCH --cpus-per-task=8
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00

#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/tuning_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/tuning_%j.err

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

export PYTHONUNBUFFERED=1
export HP_BATCH_SIZE="${HP_BATCH_SIZE:-64}"
export HP_MAX_EPOCHS="${HP_MAX_EPOCHS:-20}"
export HP_PATIENCE="${HP_PATIENCE:-3}"
export HP_TUNE_WORKERS="${HP_TUNE_WORKERS:-12}"

echo "Hostname: $(hostname)"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "DATASET_RUN_NAME: ${DATASET_RUN_NAME:-<neuester Run>}"
echo "HP_BATCH_SIZE=$HP_BATCH_SIZE | HP_MAX_EPOCHS=$HP_MAX_EPOCHS | HP_PATIENCE=$HP_PATIENCE | HP_TUNE_WORKERS=$HP_TUNE_WORKERS"
echo "Starte run_tune_hyperparams.sh …"

bash "$REPO_ROOT/run_tune_hyperparams.sh" "${DATASET_RUN_NAME:-}"
exit $?
