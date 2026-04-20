#!/bin/bash
#
# Hyperparameter-Tuning: Pair (Protokoll A) -> Genre (Protokoll B)
#
# Nutzung (vom Repo-Root):
#   bash run_tune_hyperparams.sh
#   bash run_tune_hyperparams.sh 2026-03-13_18-12-31_audioset
#   HP_BATCH_SIZE=64 HP_MAX_EPOCHS=20 HP_PATIENCE=3 HP_TUNE_WORKERS=12 bash run_tune_hyperparams.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

# Dataset-Run: 1) erstes CLI-Argument 2) ENV 3) neuester Run
if [[ $# -ge 1 && -n "${1:-}" ]]; then
  export DATASET_RUN_NAME="$1"
fi

if [[ -z "${DATASET_RUN_NAME:-}" ]]; then
  DATASET_RUN_NAME="$(python3 -c "import sys; sys.path.insert(0,'.'); import config; print(config.get_latest_run_name() or '')")"
  export DATASET_RUN_NAME
fi

if [[ -z "${DATASET_RUN_NAME:-}" ]]; then
  echo "FEHLER: DATASET_RUN_NAME nicht gesetzt und kein Run unter DATASETS_ROOT gefunden." >&2
  exit 1
fi

# Defaults (können über ENV überschrieben werden)
export HP_BATCH_SIZE="${HP_BATCH_SIZE:-64}"
export HP_MAX_EPOCHS="${HP_MAX_EPOCHS:-20}"
export HP_PATIENCE="${HP_PATIENCE:-3}"
export HP_TUNE_WORKERS="${HP_TUNE_WORKERS:-12}"

echo "========== Hyperparameter-Tuning =========="
echo "Dataset-Run: $DATASET_RUN_NAME"
echo "HP_BATCH_SIZE=$HP_BATCH_SIZE | HP_MAX_EPOCHS=$HP_MAX_EPOCHS | HP_PATIENCE=$HP_PATIENCE | HP_TUNE_WORKERS=$HP_TUNE_WORKERS"
echo ""

echo "========== 1/2 Pair-Tuning (task-aligned: Protokoll A) =========="
TRAINING_TYPE=pair python3 pipeline/tune_hyperparams.py

echo ""
echo "========== 2/2 Genre-Tuning (task-aligned: Protokoll B) =========="
TRAINING_TYPE=genre python3 pipeline/tune_hyperparams.py

echo ""
echo "========== Tuning abgeschlossen =========="
echo "Dataset-Run: $DATASET_RUN_NAME"
