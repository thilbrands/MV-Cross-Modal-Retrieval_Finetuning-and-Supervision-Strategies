#!/bin/bash
#
# Cross-modaler Genre-Centroid-Margin Δ_i = s_true − s_nearest_other
# + Korrelation mit Protocol-B-Rang. Modelle: E1 / E2 / E3a / E3b
# Aggregation: pro Genre × Richtung; ALL = Macro über Genres.
#
#   sbatch --export=DATASET_RUN_NAME=2026-05-26_15-59-27_audioset,TRAINING_RUN_DIR=/work2/ra39oxet-DatasetAudioSetSubset/training_runs/2026-07-04_19-06 jobs/analyze_soft_assignment.sh
#
# Optional: AE_PAIR_RUN_DIR, AE_GENRE_RUN_DIR, PLOT_OUTPUT_DIR
#

#SBATCH --job-name=centroid_margin
#SBATCH --partition=paula
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-01:00:00
#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/analyze_soft_assignment_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/analyze_soft_assignment_%j.err

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

echo "Hostname: $(hostname) | Job: ${SLURM_JOB_ID:-local}"
echo "DATASET_RUN_NAME: ${DATASET_RUN_NAME:-<neuester Run>}"
echo "TRAINING_RUN_DIR: ${TRAINING_RUN_DIR:-<auto>}"
echo "Starte analyze_soft_assignment.py (Δ_i + nearest-other confusion) …"

python3 "$REPO_ROOT/analyze_soft_assignment.py"
exit $?
