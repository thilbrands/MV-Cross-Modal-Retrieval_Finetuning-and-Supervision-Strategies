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
#SBATCH --time=2-00:00:00

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

export PYTHONUNBUFFERED=1
export HP_MAX_EPOCHS="${HP_MAX_EPOCHS:-20}"
export HP_PATIENCE="${HP_PATIENCE:-3}"
# Nur 1 Worker auf 1 GPU — parallele Trials führen zu OOM (Slurm mem=24GB).
export HP_TUNE_WORKERS="${HP_TUNE_WORKERS:-12}"
export HP_SEED="${HP_SEED:-42}"

echo "Hostname: $(hostname)"
echo "Slurm Job ID: $SLURM_JOB_ID"
if [[ -z "${DATASET_RUN_NAME:-}" ]]; then
  DATASET_RUN_NAME="$(python3 -c "import sys; sys.path.insert(0,'configs'); import config; print(config.get_latest_run_name() or '')")"
  export DATASET_RUN_NAME
fi
if [[ -z "${DATASET_RUN_NAME:-}" ]]; then
  echo "FEHLER: DATASET_RUN_NAME nicht gesetzt und kein Run unter DATASETS_ROOT gefunden." >&2
  exit 1
fi
echo "DATASET_RUN_NAME: $DATASET_RUN_NAME"
echo "HP_MAX_EPOCHS=$HP_MAX_EPOCHS | HP_PATIENCE=$HP_PATIENCE | HP_TUNE_WORKERS=$HP_TUNE_WORKERS | HP_SEED=$HP_SEED"
echo "Starte tuning_pair + tuning_genre …"

echo "========== 1/2 Pair-Tuning (task-aligned: Protokoll A) =========="
TRAINING_TYPE=pair python3 "$REPO_ROOT/training/tune_hyperparams.py"

echo ""
echo "========== 2/2 Genre-Tuning (task-aligned: Protokoll B) =========="
TRAINING_TYPE=genre python3 "$REPO_ROOT/training/tune_hyperparams.py"

echo ""
echo "========== Tuning abgeschlossen =========="
echo "Dataset-Run: $DATASET_RUN_NAME"
exit $?
