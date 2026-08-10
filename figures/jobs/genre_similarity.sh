#!/bin/bash
#
# Slurm-Job: Genre-Centroid-Similarity Heatmap (frozen CLIP / Wav2CLIP, Val).
# Speichert genre_similarity.png/.pdf im Dataset-Run.
#
#   sbatch --export=DATASET_RUN_NAME=... jobs/genre_similarity.sh
#

#SBATCH --job-name=genre_sim
#SBATCH --partition=paula
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Jobs liegen unter <domain>/jobs/ → Repo-Root ist ../..
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/configs" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
fi
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/configs/cluster_env.sh"
mkdir -p "$REPO_ROOT/logs"
module purge && module load Python/3.11.5-GCCcore-13.2.0
source "$VENV_ACTIVATE"
export PYTHONUNBUFFERED=1
echo "DATASET_RUN_NAME: ${DATASET_RUN_NAME:-<neuester Run>}"
python3 "$REPO_ROOT/figures/genre_similarity.py"
