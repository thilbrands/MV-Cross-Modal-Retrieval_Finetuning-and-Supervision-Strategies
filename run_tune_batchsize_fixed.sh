#!/bin/bash
#
# Kleine Batchsize-Tuning-Pipeline:
# - Pair: beste Konfiguration + bestes MLP aus PAIR_RESULTS_CSV
# - Genre: beste Konfiguration + bestes MLP aus GENRE_RESULTS_CSV
# und für beide ein Sweep über BATCH_SIZES.
#
# Nutzung:
#   PAIR_RESULTS_CSV=/work2/.../tuning_pair_.../results.csv \
#   GENRE_RESULTS_CSV=/work2/.../tuning_genre_.../results.csv \
#   bash run_tune_batchsize_fixed.sh
#   # oder ohne Parameter: nimmt automatisch die neuesten tuning_pair_*/tuning_genre_* results.csv
#   bash run_tune_batchsize_fixed.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

if [[ -z "${PAIR_RESULTS_CSV:-}" ]]; then
  PAIR_RESULTS_CSV="$(python3 - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, ".")
import config
root = config.TRAINING_RUNS_ROOT
paths = [d / "results.csv" for d in root.glob("tuning_pair_*") if (d / "results.csv").exists()]
paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
print(str(paths[0]) if paths else "")
PY
)"
  export PAIR_RESULTS_CSV
fi
if [[ -z "${GENRE_RESULTS_CSV:-}" ]]; then
  GENRE_RESULTS_CSV="$(python3 - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, ".")
import config
root = config.TRAINING_RUNS_ROOT
paths = [d / "results.csv" for d in root.glob("tuning_genre_*") if (d / "results.csv").exists()]
paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
print(str(paths[0]) if paths else "")
PY
)"
  export GENRE_RESULTS_CSV
fi
if [[ -z "${PAIR_RESULTS_CSV:-}" || -z "${GENRE_RESULTS_CSV:-}" ]]; then
  echo "FEHLER: Keine results.csv gefunden. Setze PAIR_RESULTS_CSV und GENRE_RESULTS_CSV explizit." >&2
  exit 1
fi

if [[ -z "${DATASET_RUN_NAME:-}" ]]; then
  DATASET_RUN_NAME="$(python3 -c "import sys; sys.path.insert(0,'.'); import config; print(config.get_latest_run_name() or '')")"
  export DATASET_RUN_NAME
fi
if [[ -z "${DATASET_RUN_NAME:-}" ]]; then
  echo "FEHLER: DATASET_RUN_NAME nicht gesetzt und kein Run unter DATASETS_ROOT gefunden." >&2
  exit 1
fi

export BATCH_SIZES="${BATCH_SIZES:-32 64 128 256}"
export HP_MAX_EPOCHS="${HP_MAX_EPOCHS:-20}"
export HP_PATIENCE="${HP_PATIENCE:-3}"

echo "========== Fixed Batchsize Tuning =========="
echo "Dataset-Run: $DATASET_RUN_NAME"
echo "BATCH_SIZES=$BATCH_SIZES | HP_MAX_EPOCHS=$HP_MAX_EPOCHS | HP_PATIENCE=$HP_PATIENCE"
echo "PAIR_RESULTS_CSV=$PAIR_RESULTS_CSV"
echo "GENRE_RESULTS_CSV=$GENRE_RESULTS_CSV"
echo ""

echo "========== 1/2 Pair =========="
TRAINING_TYPE=pair RESULTS_CSV="$PAIR_RESULTS_CSV" python3 pipeline/tune_batchsize_fixed.py

echo ""
echo "========== 2/2 Genre =========="
TRAINING_TYPE=genre RESULTS_CSV="$GENRE_RESULTS_CSV" python3 pipeline/tune_batchsize_fixed.py

echo ""
echo "========== Fertig =========="
