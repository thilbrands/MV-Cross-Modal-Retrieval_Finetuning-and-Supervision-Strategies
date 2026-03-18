#!/bin/bash
#
# Training + Evaluation: Pair-Training → Genre-Training → Evaluation.
# Startet nacheinander drei Slurm-Jobs; jeder nutzt den gleichen Dataset-Run.
# Heads landen in training_runs/<Datum_Uhrzeit>/; Evaluation nutzt die neuesten.
#
# Nutzung (vom Repo-Root auf dem Cluster):
#   bash run_train_and_eval.sh
#   DATASET_RUN_NAME=2026-03-13_18-12-31_audioset bash run_train_and_eval.sh
#
set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

# Dataset-Run: aus Umgebung oder neuester
export DATASET_RUN_NAME
if [[ -z "$DATASET_RUN_NAME" ]]; then
  DATASET_RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); import config; print(config.get_latest_run_name() or '')")
fi
if [[ -z "$DATASET_RUN_NAME" ]]; then
  echo "FEHLER: DATASET_RUN_NAME nicht gesetzt und kein Run unter datasets/." >&2
  exit 1
fi
echo "Dataset-Run: $DATASET_RUN_NAME"
echo ""

echo "========== 1/3 Pair-basiertes Training =========="
JOB1=$(sbatch --parsable --export=DATASET_RUN_NAME jobs/pair_based_training.sh)
echo "Job gestartet: $JOB1"
echo "Warte auf Abschluss …"
while squeue -j "$JOB1" 2>/dev/null | grep -q "$JOB1"; do sleep 60; done
echo "Pair-Training beendet."

echo ""
echo "========== 2/3 Genre-basiertes Training =========="
JOB2=$(sbatch --parsable --export=DATASET_RUN_NAME jobs/genre_based_training.sh)
echo "Job gestartet: $JOB2"
echo "Warte auf Abschluss …"
while squeue -j "$JOB2" 2>/dev/null | grep -q "$JOB2"; do sleep 60; done
echo "Genre-Training beendet."

echo ""
echo "========== 3/3 Evaluation =========="
JOB3=$(sbatch --parsable --export=DATASET_RUN_NAME jobs/evaluation.sh)
echo "Job gestartet: $JOB3"
echo "Warte auf Abschluss …"
while squeue -j "$JOB3" 2>/dev/null | grep -q "$JOB3"; do sleep 60; done
echo "Evaluation beendet."

echo ""
echo "========== Train+Eval-Pipeline fertig =========="
echo "Dataset-Run: $DATASET_RUN_NAME"
echo "Heads + meta: WORK_ROOT/training_runs/<Datum_Uhrzeit>/"
echo "Logs: WORK_ROOT/logs/"
