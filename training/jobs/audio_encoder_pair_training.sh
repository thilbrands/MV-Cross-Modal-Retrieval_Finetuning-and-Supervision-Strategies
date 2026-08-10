#!/bin/bash
#
# Slurm-Job: Pair-Training mit trainierbarem Audio-Encoder (Wav2CLIP unfrozen).
# Hyperparameter per Env-Variablen setzen (aus Tuning-Ergebnissen).
#
# Nutzung:
#   sbatch jobs/audio_encoder_pair_training.sh
#   sbatch --export=DATASET_RUN_NAME=...,HP_LR=1e-4,HP_LR_ENCODER=1e-5,HP_OUT_DIM=512,HP_TEMP=0.1,HP_HEAD_TYPE=linear,HP_HIDDEN_DIM=32,HP_BATCH_SIZE=128 jobs/audio_encoder_pair_training.sh
#

#SBATCH --job-name=ae_pair_training
#SBATCH --partition=paula
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-06:00:00

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
# Defaults aus Tuning Trial 1772; Partial Unfreeze: layer4 + transform
# HP_ENCODER_UNFREEZE: layer4_transform (default) | layer3_4_transform | full
export HP_ENCODER_UNFREEZE="${HP_ENCODER_UNFREEZE:-layer4_transform}"
export HP_LR="${HP_LR:-1e-4}"
# HP_LR_ENCODER optional: ohne Setzen → lr/3 (partial) bzw. lr/10 (full) im Python-Skript
export HP_OUT_DIM="${HP_OUT_DIM:-512}"
export HP_TEMP="${HP_TEMP:-0.1}"
export HP_HEAD_TYPE="${HP_HEAD_TYPE:-linear}"
export HP_HIDDEN_DIM="${HP_HIDDEN_DIM:-32}"
export HP_BATCH_SIZE="${HP_BATCH_SIZE:-128}"
export HP_MAX_EPOCHS="${HP_MAX_EPOCHS:-20}"
export HP_PATIENCE="${HP_PATIENCE:-3}"
export HP_SEED="${HP_SEED:-42}"

if [[ -z "${DATASET_RUN_NAME:-}" ]]; then
    DATASET_RUN_NAME="$(python3 -c "import sys; sys.path.insert(0,'configs'); import config; print(config.get_latest_run_name() or '')")"
    export DATASET_RUN_NAME
fi
if [[ -z "${DATASET_RUN_NAME:-}" ]]; then
    echo "FEHLER: DATASET_RUN_NAME nicht gesetzt." >&2
    exit 1
fi

echo "Hostname: $(hostname)"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "DATASET_RUN_NAME: $DATASET_RUN_NAME"
echo "HP_ENCODER_UNFREEZE=${HP_ENCODER_UNFREEZE:-default} | HP_LR=${HP_LR:-default} | HP_LR_ENCODER=${HP_LR_ENCODER:-auto} | HP_BATCH_SIZE=$HP_BATCH_SIZE | HP_MAX_EPOCHS=$HP_MAX_EPOCHS | HP_PATIENCE=$HP_PATIENCE"

python3 "$REPO_ROOT/training/audio_encoder_pair_training.py"
