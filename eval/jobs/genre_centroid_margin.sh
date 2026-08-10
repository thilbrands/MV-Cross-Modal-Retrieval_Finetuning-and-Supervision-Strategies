#!/bin/bash
#
# Cross-modaler Genre-Centroid-Margin Δ_i = s_true − s_nearest_other
# + Korrelation mit Protocol-B-Rang. Modelle: E1 / E2 / E3a / E3b
# Aggregation: pro Genre × Richtung; ALL = Macro über Genres.
#
#   sbatch --export=DATASET_RUN_NAME=2026-05-26_15-59-27_audioset,TRAINING_RUN_DIR=/work2/ra39oxet-DatasetAudioSetSubset/training_runs/2026-07-04_19-06 jobs/genre_centroid_margin.sh
#
# Optional: AE_PAIR_RUN_DIR, AE_GENRE_RUN_DIR, PLOT_OUTPUT_DIR
#

#SBATCH --job-name=genre_centroid_margin
#SBATCH --partition=paula
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-01:00:00
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

echo "Hostname: $(hostname) | Job: ${SLURM_JOB_ID:-local}"
echo "DATASET_RUN_NAME: ${DATASET_RUN_NAME:-<neuester Run>}"
echo "TRAINING_RUN_DIR: ${TRAINING_RUN_DIR:-<auto>}"
echo "Starte eval/genre_centroid_margin.py (Δ_i + nearest-other confusion) …"

python3 "$REPO_ROOT/eval/genre_centroid_margin.py"
exit $?
