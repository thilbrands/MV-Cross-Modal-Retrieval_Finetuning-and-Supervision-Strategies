#!/bin/bash
#
# Slurm-Job: Genre-basiertes Training (Projektions-Heads, Supervised Contrastive über Labels).
# Nutzt train_val_test_split.csv + Embeddings eines Dataset-Runs, speichert unter PROJECTION_HEADS_GENRE_PATH.
#
# Nutzung (vom Repo-Root auf dem Cluster):
#   sbatch jobs/genre_based_training.sh
#   sbatch --export=DATASET_RUN_NAME=2026-03-13_18-02-30_audioset jobs/genre_based_training.sh
#

#SBATCH --job-name=training_genre
#SBATCH --partition=paula
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00

#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/training_genre_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/training_genre_%j.err

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
echo "Starte pipeline/genre_based_training.py …"

export PYTHONUNBUFFERED=1
# Defaults aus Tuning Trial 70 (tuning_genre_2026-06-15_10-28-04)
export HP_LR="${HP_LR:-1e-3}"
export HP_OUT_DIM="${HP_OUT_DIM:-64}"
export HP_TEMP="${HP_TEMP:-1.5}"
export HP_HEAD_TYPE="${HP_HEAD_TYPE:-mlp}"
export HP_HIDDEN_DIM="${HP_HIDDEN_DIM:-512}"
export HP_BATCH_SIZE="${HP_BATCH_SIZE:-64}"
export HP_MAX_EPOCHS="${HP_MAX_EPOCHS:-20}"
export HP_PATIENCE="${HP_PATIENCE:-3}"
export HP_SEED="${HP_SEED:-42}"
python3 "$REPO_ROOT/pipeline/genre_based_training.py"
exit $?

