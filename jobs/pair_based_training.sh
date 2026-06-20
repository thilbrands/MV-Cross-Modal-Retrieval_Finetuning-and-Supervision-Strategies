#!/bin/bash
#
# Slurm-Job: Pair-basiertes Training (Projektions-Heads, InfoNCE).
# Nutzt train_val_test_split.csv + Embeddings eines Dataset-Runs, speichert unter PROJECTION_HEADS_PATH.
#
# Nutzung (vom Repo-Root auf dem Cluster):
#   sbatch jobs/pair_based_training.sh
#   sbatch --export=DATASET_RUN_NAME=2026-03-13_18-02-30_audioset jobs/pair_based_training.sh
#

#SBATCH --job-name=training
#SBATCH --partition=paula
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00

#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/training_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/training_%j.err

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
echo "Starte pipeline/pair_based_training.py …"

export PYTHONUNBUFFERED=1
# Defaults aus Tuning Trial 1110 (tuning_pair_2026-06-11_13-44-13)
export HP_LR="${HP_LR:-1e-4}"
export HP_OUT_DIM="${HP_OUT_DIM:-256}"
export HP_TEMP="${HP_TEMP:-0.1}"
export HP_HEAD_TYPE="${HP_HEAD_TYPE:-mlp}"
export HP_HIDDEN_DIM="${HP_HIDDEN_DIM:-512}"
export HP_BATCH_SIZE="${HP_BATCH_SIZE:-1024}"
export HP_MAX_EPOCHS="${HP_MAX_EPOCHS:-20}"
export HP_PATIENCE="${HP_PATIENCE:-3}"
export HP_SEED="${HP_SEED:-42}"
python3 "$REPO_ROOT/pipeline/pair_based_training.py"
exit $?
