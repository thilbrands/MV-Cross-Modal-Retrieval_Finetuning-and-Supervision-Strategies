#!/bin/bash
#
# Slurm-Job: Einmalige Extraktion roher Audiowaveforms für Audio-Encoder-Training.
# Kein GPU nötig.
#
# Nutzung:
#   sbatch jobs/extract_raw_audio.sh
#   sbatch --export=DATASET_RUN_NAME=2026-03-13_18-12-31_audioset jobs/extract_raw_audio.sh
#

#SBATCH --job-name=extract_raw_audio
#SBATCH --partition=paula
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=0-02:00:00

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

source "$VENV_ACTIVATE"

export PYTHONUNBUFFERED=1

if [[ -z "${DATASET_RUN_NAME:-}" ]]; then
    DATASET_RUN_NAME="$(python3 -c "import sys; sys.path.insert(0,'configs'); import config; print(config.get_latest_run_name() or '')")"
    export DATASET_RUN_NAME
fi
if [[ -z "${DATASET_RUN_NAME:-}" ]]; then
    echo "FEHLER: DATASET_RUN_NAME nicht gesetzt." >&2
    exit 1
fi

echo "Hostname: $(hostname)"
echo "DATASET_RUN_NAME: $DATASET_RUN_NAME"

python3 "$REPO_ROOT/data/extract_raw_audio.py"
