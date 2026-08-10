#!/bin/bash
#
# Frame-Strips für KEEP_LOW-Beispiele (Thesis-Figuren).
# Output: datasets/<RUN>/keep_low_examples/
#
#   sbatch jobs/export_keep_low_examples.sh
#   sbatch --export=DATASET_RUN_NAME=2026-05-05_11-42-52_audioset jobs/export_keep_low_examples.sh
#

#SBATCH --job-name=rm_examples
#SBATCH --partition=paula
#SBATCH --cpus-per-task=2
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

export PYTHONUNBUFFERED=1

echo "Hostname: $(hostname) | Job: $SLURM_JOB_ID"
echo "DATASET_RUN_NAME: ${DATASET_RUN_NAME:-<neuester Run>}"

python3 "$REPO_ROOT/figures/export_keep_low_frame_examples.py"
exit $?
