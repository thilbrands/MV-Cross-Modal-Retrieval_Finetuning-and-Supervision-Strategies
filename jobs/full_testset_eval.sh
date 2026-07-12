#!/bin/bash
#
# Slurm-Job: Full-Testset-Evaluation (Protocol A + B, gesamter Test-Split)
#
# Nutzung:
#   sbatch jobs/full_testset_eval.sh
#   sbatch --export=DATASET_RUN_NAME=...,TRAINING_RUN_DIR=...,AE_PAIR_RUN_DIR=...,AE_GENRE_RUN_DIR=... jobs/full_testset_eval.sh
#

#SBATCH --job-name=full_testset_eval
#SBATCH --partition=paula
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=0-4:00:00

#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/full_testset_eval_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/full_testset_eval_%j.err

set -euo pipefail

WORK_ROOT="/work2/ra39oxet-DatasetAudioSetSubset"
mkdir -p "$WORK_ROOT/logs"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT" || { echo "FEHLER: cd nach $REPO_ROOT fehlgeschlagen." >&2; exit 1; }

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "$HOME/venv/ba/bin/activate"

export PYTHONUNBUFFERED=1
python3 "$REPO_ROOT/pipeline/full_testset_eval.py"
