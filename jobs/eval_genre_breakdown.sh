#!/bin/bash
#
# Slurm-Job: Genre-Breakdown-Evaluation (Metriken pro Genre, Seen vs. Unseen).
#
#SBATCH --job-name=eval_genre_breakdown
#SBATCH --partition=paula
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-4:00:00
#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/eval_genre_breakdown_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/eval_genre_breakdown_%j.err

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
echo "TRAINING_RUN_DIR: ${TRAINING_RUN_DIR:-<nicht gesetzt>}"
echo "TRAIN_GENRES:     ${TRAIN_GENRES:-<alle>}"
echo "Starte pipeline/eval_genre_breakdown.py …"

export PYTHONUNBUFFERED=1
python3 "$REPO_ROOT/pipeline/eval_genre_breakdown.py"
exit $?
