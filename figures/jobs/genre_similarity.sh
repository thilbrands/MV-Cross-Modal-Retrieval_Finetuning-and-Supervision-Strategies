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
#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/genre_similarity_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/genre_similarity_%j.err

set -euo pipefail
WORK_ROOT="/work2/ra39oxet-DatasetAudioSetSubset"
mkdir -p "$WORK_ROOT/logs"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Jobs liegen unter <domain>/jobs/ → Repo-Root ist ../..
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/configs" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
fi
cd "$REPO_ROOT"
module purge && module load Python/3.11.5-GCCcore-13.2.0
source "$HOME/venv/ba/bin/activate"
export PYTHONUNBUFFERED=1
echo "DATASET_RUN_NAME: ${DATASET_RUN_NAME:-<neuester Run>}"
python3 "$REPO_ROOT/figures/genre_similarity.py"
