#!/bin/bash
#
# Slurm-Job: Genre-Training mit trainierbarem Audio-Encoder (Wav2CLIP unfrozen).
# Hyperparameter per Env-Variablen setzen (aus Tuning-Ergebnissen).
#
# Nutzung:
#   sbatch jobs/audio_encoder_genre_training.sh
#   sbatch --export=DATASET_RUN_NAME=...,HP_LR=1e-3,HP_LR_ENCODER=1e-4,HP_OUT_DIM=512,HP_TEMP=0.05,HP_HEAD_TYPE=mlp,HP_HIDDEN_DIM=128,HP_BATCH_SIZE=64 jobs/audio_encoder_genre_training.sh
#

#SBATCH --job-name=ae_genre_training
#SBATCH --partition=paula
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-06:00:00

#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/ae_genre_training_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/ae_genre_training_%j.err

set -euo pipefail

WORK_ROOT="/work2/ra39oxet-DatasetAudioSetSubset"
mkdir -p "$WORK_ROOT/logs"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT" || { echo "FEHLER: cd nach $REPO_ROOT fehlgeschlagen." >&2; exit 1; }

module purge
module load Python/3.11.5-GCCcore-13.2.0

source "$HOME/venv/ba/bin/activate"

export PYTHONUNBUFFERED=1
# Defaults aus Tuning Trial 758; Partial Unfreeze: layer4 + transform
# HP_ENCODER_UNFREEZE: layer4_transform (default) | layer3_4_transform | full
export HP_ENCODER_UNFREEZE="${HP_ENCODER_UNFREEZE:-layer4_transform}"
export HP_LR="${HP_LR:-1e-3}"
# HP_LR_ENCODER optional: ohne Setzen → lr/3 (partial) bzw. lr/10 (full) im Python-Skript
export HP_OUT_DIM="${HP_OUT_DIM:-512}"
export HP_TEMP="${HP_TEMP:-0.05}"
export HP_HEAD_TYPE="${HP_HEAD_TYPE:-mlp}"
export HP_HIDDEN_DIM="${HP_HIDDEN_DIM:-128}"
export HP_BATCH_SIZE="${HP_BATCH_SIZE:-64}"
export HP_MAX_EPOCHS="${HP_MAX_EPOCHS:-20}"
export HP_PATIENCE="${HP_PATIENCE:-3}"
export HP_SEED="${HP_SEED:-42}"

if [[ -z "${DATASET_RUN_NAME:-}" ]]; then
    DATASET_RUN_NAME="$(python3 -c "import sys; sys.path.insert(0,'.'); import config; print(config.get_latest_run_name() or '')")"
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

python3 "$REPO_ROOT/pipeline/audio_encoder_genre_training.py"
