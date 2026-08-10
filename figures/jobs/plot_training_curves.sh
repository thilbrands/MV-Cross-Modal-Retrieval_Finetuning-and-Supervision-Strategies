#!/bin/bash
#
# Slurm-Job: Training-Curves-Plot (E1/E2/E3a/E3b) aus results_*.csv.
#
#   sbatch --export=TRAINING_RUN_DIR=... jobs/plot_training_curves.sh
#

#SBATCH --job-name=train_curves
#SBATCH --partition=paula
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=0-00:20:00
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
echo "TRAINING_RUN_DIR: ${TRAINING_RUN_DIR:-<auto>}"
python3 "$REPO_ROOT/figures/plot_training_curves.py"
