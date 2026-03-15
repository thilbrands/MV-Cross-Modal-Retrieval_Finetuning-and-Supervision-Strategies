#!/bin/bash
#
# Data-Preparation-Pipeline: Download → Extract+Embed → Train/Val/Test-Split.
# Startet nacheinander drei Slurm-Jobs und wartet nach jedem auf Abschluss.
# Am Ende liegt ein neuer Dataset-Run mit train_val_test_split.csv vor.
# (Spätere Schritte wie Training o. Ä. kommen in eigene Pipelines.)
#
# Nutzung (vom Repo-Root auf dem Cluster):
#   bash run_data_preparation_pipeline.sh
#
# Voraussetzung: Venv mit allen Abhängigkeiten, config.py mit korrektem WORK_ROOT.
#
set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

echo "========== 1/3 Downloader (neuer Run) =========="
JOB1=$(sbatch --parsable jobs/downloader.sh)
echo "Job gestartet: $JOB1"
echo "Warte auf Abschluss …"
while squeue -j "$JOB1" 2>/dev/null | grep -q "$JOB1"; do sleep 60; done
echo "Downloader beendet."

echo ""
echo "========== 2/3 Extract + Embed (neuester Run) =========="
export DATASET_RUN_NAME
DATASET_RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); import config; print(config.get_latest_run_name() or '')")
if [[ -z "$DATASET_RUN_NAME" ]]; then
  echo "FEHLER: Kein Run gefunden (get_latest_run_name)." >&2
  exit 1
fi
echo "Nutze Run: $DATASET_RUN_NAME"
JOB2=$(sbatch --parsable --export=DATASET_RUN_NAME jobs/extract_and_embed_videos.sh)
echo "Job gestartet: $JOB2"
echo "Warte auf Abschluss …"
while squeue -j "$JOB2" 2>/dev/null | grep -q "$JOB2"; do sleep 60; done
echo "Extract+Embed beendet."

echo ""
echo "========== 3/3 Train/Val/Test-Split =========="
JOB3=$(sbatch --parsable --export=DATASET_RUN_NAME jobs/train_val_test_split.sh)
echo "Job gestartet: $JOB3"
echo "Warte auf Abschluss …"
while squeue -j "$JOB3" 2>/dev/null | grep -q "$JOB3"; do sleep 60; done
echo "Split beendet."

echo ""
echo "========== Data-Preparation-Pipeline fertig =========="
echo "Run: $DATASET_RUN_NAME"
echo "  - datasets/$DATASET_RUN_NAME/train_val_test_split.csv"
echo "  - datasets/$DATASET_RUN_NAME/embeddings/"
