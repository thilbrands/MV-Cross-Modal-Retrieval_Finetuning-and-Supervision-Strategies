#!/bin/bash
#
# Training + Evaluation: Pair → Genre → Audio-Encoder Pair → Audio-Encoder Genre → Evaluation.
# Erstellt einen gemeinsamen Ordner training_runs/<Datum_Uhrzeit>/; alle Schritte
# speichern darin (Heads, Meta, Evaluation-Ausgabe).
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

# Ein gemeinsamer Training-Run-Ordner für alle drei Schritte
export TRAINING_RUN_DIR
TRAINING_RUN_DIR=$(python3 -c "import sys; sys.path.insert(0,'.'); import config; d=config.get_new_training_run_dir(); print(d)")
echo "Dataset-Run: $DATASET_RUN_NAME"
echo "Training-Run (alle Ausgaben): $TRAINING_RUN_DIR"
echo ""

echo "========== 1/3 Pair-basiertes Training =========="
JOB1=$(sbatch --parsable --export=DATASET_RUN_NAME,TRAINING_RUN_DIR \
  --output="$TRAINING_RUN_DIR/training_pair.out" --error="$TRAINING_RUN_DIR/training_pair.err" \
  jobs/pair_based_training.sh)
echo "Job gestartet: $JOB1"
echo "Warte auf Abschluss …"
while squeue -j "$JOB1" 2>/dev/null | grep -q "$JOB1"; do sleep 60; done
echo "Pair-Training beendet."

echo ""
echo "========== 2/3 Genre-basiertes Training =========="
JOB2=$(sbatch --parsable --export=DATASET_RUN_NAME,TRAINING_RUN_DIR \
  --output="$TRAINING_RUN_DIR/training_genre.out" --error="$TRAINING_RUN_DIR/training_genre.err" \
  jobs/genre_based_training.sh)
echo "Job gestartet: $JOB2"
echo "Warte auf Abschluss …"
while squeue -j "$JOB2" 2>/dev/null | grep -q "$JOB2"; do sleep 60; done
echo "Genre-Training beendet."

echo ""
echo "========== 3/4 Audio-Encoder Pair-Training =========="
JOB3=$(sbatch --parsable --export=DATASET_RUN_NAME,TRAINING_RUN_DIR \
  --output="$TRAINING_RUN_DIR/ae_pair_training.out" --error="$TRAINING_RUN_DIR/ae_pair_training.err" \
  jobs/audio_encoder_pair_training.sh)
echo "Job gestartet: $JOB3"
echo "Warte auf Abschluss …"
while squeue -j "$JOB3" 2>/dev/null | grep -q "$JOB3"; do sleep 60; done
echo "Audio-Encoder Pair-Training beendet."

echo ""
echo "========== 4/4 Audio-Encoder Genre-Training =========="
JOB4=$(sbatch --parsable --export=DATASET_RUN_NAME,TRAINING_RUN_DIR \
  --output="$TRAINING_RUN_DIR/ae_genre_training.out" --error="$TRAINING_RUN_DIR/ae_genre_training.err" \
  jobs/audio_encoder_genre_training.sh)
echo "Job gestartet: $JOB4"
echo "Warte auf Abschluss …"
while squeue -j "$JOB4" 2>/dev/null | grep -q "$JOB4"; do sleep 60; done
echo "Audio-Encoder Genre-Training beendet."

echo ""
echo "========== 5/5 Evaluation =========="
export AE_PAIR_RUN_DIR="$TRAINING_RUN_DIR"
export AE_GENRE_RUN_DIR="$TRAINING_RUN_DIR"
JOB5=$(sbatch --parsable --export=DATASET_RUN_NAME,TRAINING_RUN_DIR,AE_PAIR_RUN_DIR,AE_GENRE_RUN_DIR \
  --output="$TRAINING_RUN_DIR/evaluation.out" --error="$TRAINING_RUN_DIR/evaluation.err" \
  jobs/evaluation.sh)
echo "Job gestartet: $JOB5"
echo "Warte auf Abschluss …"
while squeue -j "$JOB5" 2>/dev/null | grep -q "$JOB5"; do sleep 60; done
echo "Evaluation beendet."

echo ""
echo "========== Train+Eval-Pipeline fertig =========="
echo "Dataset-Run: $DATASET_RUN_NAME"
echo "Alles in einem Ordner: $TRAINING_RUN_DIR"
echo "  (projection_heads_*.pt, audio_encoder_*.pt, meta_*.json, *.out/.err, evaluation_output.txt)"
