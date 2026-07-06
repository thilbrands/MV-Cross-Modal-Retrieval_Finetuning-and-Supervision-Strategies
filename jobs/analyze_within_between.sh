#!/bin/bash
#
# Within/Between-Genre-Analyse (projiziert, E1/E2, Audio/Video).
#
#   sbatch --export=DATASET_RUN_NAME=2026-05-26_15-59-27_audioset,TRAINING_RUN_DIR=/work2/ra39oxet-DatasetAudioSetSubset/training_runs/2026-07-04_19-06 jobs/analyze_within_between.sh
#

#SBATCH --job-name=within_between
#SBATCH --partition=paula
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-01:00:00
#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/analyze_within_between_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/analyze_within_between_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
module purge && module load Python/3.11.5-GCCcore-13.2.0
source "$HOME/venv/ba/bin/activate"
export PYTHONUNBUFFERED=1
python3 "$REPO_ROOT/analyze_within_between.py"
