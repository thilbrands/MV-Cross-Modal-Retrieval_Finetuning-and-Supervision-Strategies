#!/bin/bash
#
# Slurm-Job: Genre-Centroid-Similarity (Audio + Video, je Modalität) für E1–E3b.
# Pro Modell ein eigenes PDF — projiziertes Gegenstück zu genre_similarity.png.
#
# Nutzung (vom Repo-Root auf dem Cluster):
#   sbatch jobs/plot_model_genre_similarity.sh
#   sbatch --export=DATASET_RUN_NAME=2026-05-26_15-59-27_audioset,TRAINING_RUN_DIR=/work2/ra39oxet-DatasetAudioSetSubset/training_runs/2026-07-04_19-06 jobs/plot_model_genre_similarity.sh
#

#SBATCH --job-name=plot_model_genre_sim
#SBATCH --partition=paula
#SBATCH --cpus-per-task=8
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-01:00:00

#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/plot_model_genre_similarity_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/plot_model_genre_similarity_%j.err

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
echo "TRAINING_RUN_DIR: ${TRAINING_RUN_DIR:-<auto>}"
echo "PLOT_OUTPUT_DIR:  ${PLOT_OUTPUT_DIR:-<TRAINING_RUN_DIR>}"
echo "Starte plot_model_genre_similarity.py …"

export PYTHONUNBUFFERED=1
python3 "$REPO_ROOT/plot_model_genre_similarity.py"
exit $?
