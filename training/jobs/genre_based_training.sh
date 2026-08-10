#!/bin/bash
#
# Slurm-Job: Genre-basiertes Training (Projektions-Heads, Supervised Contrastive über Labels).
# Nutzt train_val_test_split.csv + Embeddings eines Dataset-Runs, speichert unter PROJECTION_HEADS_GENRE_PATH.
#
# Nutzung (vom Repo-Root auf dem Cluster):
#   sbatch jobs/genre_based_training.sh
#   sbatch --export=DATASET_RUN_NAME=2026-05-26_15-59-27_audioset jobs/genre_based_training.sh
#   sbatch --export=DATASET_RUN_NAME=...,HP_LR=1e-3,HP_OUT_DIM=512,HP_TEMP=0.05,HP_HEAD_TYPE=mlp,HP_HIDDEN_DIM=128,HP_BATCH_SIZE=64 jobs/genre_based_training.sh
#

#SBATCH --job-name=training_genre
#SBATCH --partition=paula
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


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
echo "Starte training/genre_based_training.py …"

export PYTHONUNBUFFERED=1
# Defaults aus Tuning Trial 758 (tuning_genre_2026-07-03_12-11-17, MRR-Selection)
export HP_LR="${HP_LR:-1e-3}"
export HP_OUT_DIM="${HP_OUT_DIM:-512}"
export HP_TEMP="${HP_TEMP:-0.05}"
export HP_HEAD_TYPE="${HP_HEAD_TYPE:-mlp}"
export HP_HIDDEN_DIM="${HP_HIDDEN_DIM:-128}"
export HP_BATCH_SIZE="${HP_BATCH_SIZE:-64}"
export HP_MAX_EPOCHS="${HP_MAX_EPOCHS:-20}"
export HP_PATIENCE="${HP_PATIENCE:-3}"
export HP_SEED="${HP_SEED:-42}"
python3 "$REPO_ROOT/training/genre_based_training.py"
exit $?

