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
export HP_MAX_EPOCHS="${HP_MAX_EPOCHS:-20}"
export HP_PATIENCE="${HP_PATIENCE:-3}"
# Nur 1 Worker auf 1 GPU — parallele Trials führen zu OOM (Slurm mem=24GB).
export HP_TUNE_WORKERS="${HP_TUNE_WORKERS:-1}"
export HP_SEED="${HP_SEED:-42}"

echo "Hostname: $(hostname)"
echo "Slurm Job ID: $SLURM_JOB_ID"
if [[ -z "${DATASET_RUN_NAME:-}" ]]; then
  DATASET_RUN_NAME="$(python3 -c "import sys; sys.path.insert(0,'.'); import config; print(config.get_latest_run_name() or '')")"
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
TRAINING_TYPE=pair python3 "$REPO_ROOT/pipeline/tune_hyperparams.py"

echo ""
echo "========== 2/2 Genre-Tuning (task-aligned: Protokoll B) =========="
TRAINING_TYPE=genre python3 "$REPO_ROOT/pipeline/tune_hyperparams.py"

echo ""
echo "========== Tuning abgeschlossen =========="
echo "Dataset-Run: $DATASET_RUN_NAME"
exit $?
