#!/bin/bash
#
# Slurm-Job: Train/Val/Test-Split (stratifiziert) für einen Dataset-Run.
# Liest segments_balanced.csv + Embeddings, schreibt train_val_test_split.csv.
#
# Nutzung (vom Repo-Root auf dem Cluster):
#   sbatch jobs/train_val_test_split.sh
#     → nutzt automatisch den neuesten Run (Default)
#   sbatch --export=DATASET_RUN_NAME=2026-03-13_18-02-30_audioset jobs/train_val_test_split.sh
#     → nutzt den angegebenen Run
#

#SBATCH --job-name=split
#SBATCH --partition=paula
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=0-00:30:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


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

echo "Hostname: $(hostname)"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "DATASET_RUN_NAME: ${DATASET_RUN_NAME:-<neuester Run>}"
echo "Starte data/train_val_test_split.py …"

export PYTHONUNBUFFERED=1
python3 "$REPO_ROOT/data/train_val_test_split.py"
exit $?
