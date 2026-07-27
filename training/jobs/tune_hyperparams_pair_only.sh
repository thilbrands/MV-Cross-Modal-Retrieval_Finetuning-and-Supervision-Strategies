#!/bin/bash
#
# Slurm-Job: Hyperparameter-Tuning nur für Pair (InfoNCE), Selection = MRR Protocol A.
#
# Nutzung (vom Repo-Root auf dem Cluster):
#   sbatch jobs/tune_hyperparams_pair_only.sh
#   sbatch --export=DATASET_RUN_NAME=2026-05-26_15-59-27_audioset jobs/tune_hyperparams_pair_only.sh
#

#SBATCH --job-name=tuning_pair
#SBATCH --partition=paula
#SBATCH --cpus-per-task=8
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00

#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/tuning_pair_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/tuning_pair_%j.err

set -euo pipefail

WORK_ROOT="/work2/ra39oxet-DatasetAudioSetSubset"
mkdir -p "$WORK_ROOT/logs"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Jobs liegen unter <domain>/jobs/ → Repo-Root ist ../..
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/configs" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
fi
cd "$REPO_ROOT" || { echo "FEHLER: cd nach $REPO_ROOT fehlgeschlagen." >&2; exit 1; }

module purge
module load Python/3.11.5-GCCcore-13.2.0

if [[ ! -f "$HOME/venv/ba/bin/activate" ]]; then
  echo "FEHLER: Venv nicht gefunden ($HOME/venv/ba)." >&2
  exit 1
fi
source "$HOME/venv/ba/bin/activate"

export PYTHONUNBUFFERED=1
export HP_MAX_EPOCHS="${HP_MAX_EPOCHS:-20}"
export HP_PATIENCE="${HP_PATIENCE:-3}"
export HP_TUNE_WORKERS="${HP_TUNE_WORKERS:-12}"
export HP_SEED="${HP_SEED:-42}"

if [[ -z "${DATASET_RUN_NAME:-}" ]]; then
  DATASET_RUN_NAME="$(python3 -c "import sys; sys.path.insert(0,'configs'); import config; print(config.get_latest_run_name() or '')")"
  export DATASET_RUN_NAME
fi
if [[ -z "${DATASET_RUN_NAME:-}" ]]; then
  echo "FEHLER: DATASET_RUN_NAME nicht gesetzt und kein Run unter DATASETS_ROOT gefunden." >&2
  exit 1
fi

echo "Hostname: $(hostname)"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "DATASET_RUN_NAME: $DATASET_RUN_NAME"
echo "HP_MAX_EPOCHS=$HP_MAX_EPOCHS | HP_PATIENCE=$HP_PATIENCE | HP_TUNE_WORKERS=$HP_TUNE_WORKERS"
echo "Starte Pair-Tuning (MRR, Protocol A) …"

TRAINING_TYPE=pair python3 "$REPO_ROOT/training/tune_hyperparams.py"
exit $?
