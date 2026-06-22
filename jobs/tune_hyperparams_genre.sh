#!/bin/bash
#
# Slurm-Job: Hyperparameter-Tuning nur für Genre-Training (Protokoll B).
# out_dim ist im Tuning fest auf 512 gesetzt (siehe tune_hyperparams.py).
# Pair-Tuning läuft hier nicht — dafür jobs/tune_hyperparams.sh oder TRAINING_TYPE=pair.
#
# Nutzung (vom Repo-Root auf dem Cluster):
#   sbatch jobs/tune_hyperparams_genre.sh
#   sbatch --export=DATASET_RUN_NAME=2026-05-05_11-42-52_audioset jobs/tune_hyperparams_genre.sh
#   sbatch --export=DATASET_RUN_NAME=...,HP_TUNE_WORKERS=1,HP_MAX_EPOCHS=20 jobs/tune_hyperparams_genre.sh
#

#SBATCH --job-name=tuning_genre
#SBATCH --partition=paula
#SBATCH --cpus-per-task=12
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00

#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/tuning_genre_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/tuning_genre_%j.err

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
export TRAINING_TYPE=genre
export HP_MAX_EPOCHS="${HP_MAX_EPOCHS:-20}"
export HP_PATIENCE="${HP_PATIENCE:-3}"
# Genre-Trials sind leichtgewichtig (nur Projektionsköpfe + gecachte Embeddings,
# keine CLIP/Wav2CLIP-Modelle) → mehrere parallele Worker sind unkritisch.
export HP_TUNE_WORKERS="${HP_TUNE_WORKERS:-12}"
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
echo "TRAINING_TYPE: $TRAINING_TYPE"
echo "HP_MAX_EPOCHS=$HP_MAX_EPOCHS | HP_PATIENCE=$HP_PATIENCE | HP_TUNE_WORKERS=$HP_TUNE_WORKERS | HP_SEED=$HP_SEED"
echo "Starte Genre-Tuning (task-aligned: Protokoll B) …"

python3 "$REPO_ROOT/pipeline/tune_hyperparams.py"

echo ""
echo "========== Genre-Tuning abgeschlossen =========="
echo "Ergebnisse unter: $WORK_ROOT/training_runs/tuning_genre_*"
echo "Dataset-Run: $DATASET_RUN_NAME"
exit $?
