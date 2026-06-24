#!/bin/bash
#
# Slurm-Job: E4-Interpolation — Training auf 7 Genres, Evaluation auf allen 10.
# Unseen Genres (Interpolation): Electronic music, Funk, Reggae.
# Seen Genres (Training): Blues, Classical music, Country, Hip hop music, Jazz, Pop music, Rock music.
#
#SBATCH --job-name=e4_interpolation
#SBATCH --partition=paula
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=0-24:00:00
#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/e4_interpolation_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/e4_interpolation_%j.err
#
# Nutzung (vom Repo-Root auf dem Cluster):
#   bash run_e4_interpolation.sh
#   sbatch run_e4_interpolation.sh
#   DATASET_RUN_NAME=2026-03-13_18-12-31_audioset sbatch run_e4_interpolation.sh
#
set -e
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$REPO_ROOT" || { echo "FEHLER: cd nach $REPO_ROOT fehlgeschlagen." >&2; exit 1; }

# Dataset-Run: aus Umgebung oder neuester
export DATASET_RUN_NAME
if [[ -z "$DATASET_RUN_NAME" ]]; then
  DATASET_RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); import config; print(config.get_latest_run_name() or '')")
fi
if [[ -z "$DATASET_RUN_NAME" ]]; then
  echo "FEHLER: DATASET_RUN_NAME nicht gesetzt und kein Run unter datasets/." >&2
  exit 1
fi

# Fester Ordner für E4-Interpolation
export TRAINING_RUN_DIR="/work2/ra39oxet-DatasetAudioSetSubset/training_runs/e4_interpolation"
mkdir -p "$TRAINING_RUN_DIR"
mkdir -p "/work2/ra39oxet-DatasetAudioSetSubset/logs"

# 7 Trainings-Genres — Unseen: Electronic music, Funk, Reggae
export TRAIN_GENRES="Blues,Classical music,Country,Hip hop music,Jazz,Pop music,Rock music"

echo "Dataset-Run:      $DATASET_RUN_NAME"
echo "Training-Run-Dir: $TRAINING_RUN_DIR"
echo "Train-Genres:     $TRAIN_GENRES"
echo ""

echo "========== 1/5 Pair-basiertes Training (7 Genres) =========="
JOB1=$(sbatch --parsable --export=DATASET_RUN_NAME,TRAINING_RUN_DIR,TRAIN_GENRES \
  --output="$TRAINING_RUN_DIR/training_pair.out" --error="$TRAINING_RUN_DIR/training_pair.err" \
  jobs/pair_based_training.sh)
echo "Job gestartet: $JOB1"
echo "Warte auf Abschluss …"
while squeue -j "$JOB1" 2>/dev/null | grep -q "$JOB1"; do sleep 60; done
echo "Pair-Training beendet."

echo ""
echo "========== 2/5 Genre-basiertes Training (7 Genres) =========="
JOB2=$(sbatch --parsable --export=DATASET_RUN_NAME,TRAINING_RUN_DIR,TRAIN_GENRES \
  --output="$TRAINING_RUN_DIR/training_genre.out" --error="$TRAINING_RUN_DIR/training_genre.err" \
  jobs/genre_based_training.sh)
echo "Job gestartet: $JOB2"
echo "Warte auf Abschluss …"
while squeue -j "$JOB2" 2>/dev/null | grep -q "$JOB2"; do sleep 60; done
echo "Genre-Training beendet."

echo ""
echo "========== 3/5 Audio-Encoder Pair-Training (7 Genres) =========="
JOB3=$(sbatch --parsable --export=DATASET_RUN_NAME,TRAINING_RUN_DIR,TRAIN_GENRES \
  --output="$TRAINING_RUN_DIR/ae_pair_training.out" --error="$TRAINING_RUN_DIR/ae_pair_training.err" \
  jobs/audio_encoder_pair_training.sh)
echo "Job gestartet: $JOB3"
echo "Warte auf Abschluss …"
while squeue -j "$JOB3" 2>/dev/null | grep -q "$JOB3"; do sleep 60; done
echo "Audio-Encoder Pair-Training beendet."

echo ""
echo "========== 4/5 Audio-Encoder Genre-Training (7 Genres) =========="
JOB4=$(sbatch --parsable --export=DATASET_RUN_NAME,TRAINING_RUN_DIR,TRAIN_GENRES \
  --output="$TRAINING_RUN_DIR/ae_genre_training.out" --error="$TRAINING_RUN_DIR/ae_genre_training.err" \
  jobs/audio_encoder_genre_training.sh)
echo "Job gestartet: $JOB4"
echo "Warte auf Abschluss …"
while squeue -j "$JOB4" 2>/dev/null | grep -q "$JOB4"; do sleep 60; done
echo "Audio-Encoder Genre-Training beendet."

echo ""
echo "========== 5/5 Evaluation (alle 10 Genres) =========="
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
echo "========== 6/6 Genre-Breakdown-Evaluation =========="
JOB6=$(sbatch --parsable --export=DATASET_RUN_NAME,TRAINING_RUN_DIR,AE_PAIR_RUN_DIR,AE_GENRE_RUN_DIR,TRAIN_GENRES \
  --output="$TRAINING_RUN_DIR/genre_breakdown.out" --error="$TRAINING_RUN_DIR/genre_breakdown.err" \
  jobs/eval_genre_breakdown.sh)
echo "Job gestartet: $JOB6"
echo "Warte auf Abschluss …"
while squeue -j "$JOB6" 2>/dev/null | grep -q "$JOB6"; do sleep 60; done
echo "Genre-Breakdown beendet."

echo ""
echo "========== E4-Interpolation fertig =========="
echo "Dataset-Run:  $DATASET_RUN_NAME"
echo "Ausgaben in:  $TRAINING_RUN_DIR"
echo "  (projection_heads_*.pt, audio_encoder_*.pt, meta_*.json, *.out/.err, evaluation_output.txt, genre_breakdown.txt, genre_breakdown_plot.png)"
