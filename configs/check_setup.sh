#!/bin/bash
# Sanity-check for cluster setup. Run from repo root:
#   bash configs/check_setup.sh
#
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/configs/cluster_env.sh"

echo "Repo:          $REPO_ROOT"
echo "WORK_ROOT:     $WORK_ROOT"
echo "VENV_ACTIVATE: $VENV_ACTIVATE"
echo ""

err=0
warn=0

if [[ ! -d "$WORK_ROOT" ]]; then
  echo "ERROR: WORK_ROOT does not exist: $WORK_ROOT"
  echo "       Edit configs/cluster_env.sh or: export WORK_ROOT=/work2/your-dir"
  err=1
elif [[ ! -w "$WORK_ROOT" ]]; then
  echo "ERROR: WORK_ROOT is not writable: $WORK_ROOT"
  err=1
else
  echo "OK   WORK_ROOT writable"
fi

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "ERROR: venv activate script missing: $VENV_ACTIVATE"
  echo "       module load Python/3.11.5-GCCcore-13.2.0"
  echo "       python3 -m venv ~/venv/ba && source ~/venv/ba/bin/activate"
  echo "       pip install -r requirements.txt"
  err=1
else
  echo "OK   venv found"
fi

DATA_CSV="$WORK_ROOT/AudioSetData/unbalanced_train_segments-2.csv"
ONTOLOGY="$WORK_ROOT/AudioSetData/ontology.json"
if [[ -f "$DATA_CSV" && -f "$ONTOLOGY" ]]; then
  echo "OK   AudioSet metadata present"
else
  echo "WARN AudioSet metadata missing (needed for download/dataset pipeline):"
  echo "       $DATA_CSV"
  echo "       $ONTOLOGY"
  warn=1
fi

# Any dataset run with embeddings?
emb_ok=0
if [[ -d "$WORK_ROOT/datasets" ]]; then
  for d in "$WORK_ROOT/datasets"/*/embeddings; do
    if [[ -d "$d" ]]; then
      emb_ok=1
      echo "OK   embeddings found: $d"
      break
    fi
  done
fi
if [[ "$emb_ok" -eq 0 ]]; then
  echo "WARN no embeddings under $WORK_ROOT/datasets/*/embeddings"
  echo "       Run: bash pipelines/run_dataset_pipeline.sh"
  echo "       (or copy an existing dataset run into WORK_ROOT/datasets/)"
  warn=1
fi

mkdir -p "$REPO_ROOT/logs"
echo "OK   repo logs/ ready (submit jobs from repo root)"
echo ""

if [[ "$err" -ne 0 ]]; then
  echo "Setup incomplete — fix ERROR lines above."
  exit 1
fi
if [[ "$warn" -ne 0 ]]; then
  echo "Setup usable with warnings — full training needs embeddings + (for download) AudioSet CSVs."
  exit 0
fi
echo "Setup OK."
exit 0
