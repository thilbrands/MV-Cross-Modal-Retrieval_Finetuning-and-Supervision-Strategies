#!/bin/bash
#
# Nur Genre-Breakdown + Plot für beide E4-Varianten (kein Training).
# Voraussetzung: Checkpoints + audio_encoder_*_test_embeddings/ liegen bereits
# in training_runs/e4_exploration bzw. e4_interpolation.
#
# Nutzung (vom Repo-Root auf dem Cluster):
#   bash generalization_experiment/run_e4_genre_breakdown_only.sh
#   DATASET_RUN_NAME=2026-05-26_15-59-27_audioset bash generalization_experiment/run_e4_genre_breakdown_only.sh
#
set -e
_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Orchestratoren liegen unter generalization_experiment/ → Repo-Root ist eine Ebene höher
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/configs" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
elif [[ -d "$_SCRIPT_DIR/../configs" ]]; then
  REPO_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"
else
  REPO_ROOT="$_SCRIPT_DIR"
fi
cd "$REPO_ROOT" || { echo "FEHLER: cd nach $REPO_ROOT fehlgeschlagen." >&2; exit 1; }
# shellcheck disable=SC1091
source "$REPO_ROOT/configs/cluster_env.sh"
mkdir -p "$REPO_ROOT/logs"

WORK_ROOT=$(python3 -c "import sys; sys.path.insert(0,'configs'); import config; print(config.WORK_ROOT)")
mkdir -p "$WORK_ROOT/logs"

export DATASET_RUN_NAME
if [[ -z "$DATASET_RUN_NAME" ]]; then
  DATASET_RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'configs'); import config; print(config.get_latest_run_name() or '')")
fi
if [[ -z "$DATASET_RUN_NAME" ]]; then
  echo "FEHLER: DATASET_RUN_NAME nicht gesetzt und kein Run unter datasets/." >&2
  exit 1
fi

_run_eval() {
  local name="$1"
  local run_dir="$2"
  local train_genres="$3"

  if [[ ! -f "$run_dir/projection_heads_pair.pt" ]] || [[ ! -f "$run_dir/projection_heads_genre.pt" ]]; then
    echo "FEHLER: $name — fehlende Heads in $run_dir" >&2
    exit 1
  fi

  export TRAINING_RUN_DIR="$run_dir"
  export TRAIN_GENRES="$train_genres"
  export AE_PAIR_RUN_DIR="$run_dir"
  export AE_GENRE_RUN_DIR="$run_dir"

  echo ""
  echo "========== Genre-Breakdown: $name =========="
  echo "Dataset-Run:      $DATASET_RUN_NAME"
  echo "Training-Run-Dir: $TRAINING_RUN_DIR"
  echo "TRAIN_GENRES:     $TRAIN_GENRES"

  JOB=$(sbatch --parsable \
    --export=WORK_ROOT,VENV_ACTIVATE,DATASET_RUN_NAME,TRAINING_RUN_DIR,TRAIN_GENRES,AE_PAIR_RUN_DIR,AE_GENRE_RUN_DIR \
    --output="$TRAINING_RUN_DIR/genre_breakdown.out" \
    --error="$TRAINING_RUN_DIR/genre_breakdown.err" \
    generalization_experiment/jobs/eval_genre_breakdown.sh)
  echo "Job gestartet: $JOB"
  echo "Warte auf Abschluss …"
  while squeue -j "$JOB" 2>/dev/null | grep -q "$JOB"; do sleep 30; done
  echo "Fertig: $name → $run_dir/genre_breakdown_plot.pdf"
}

_run_eval "E4 Exploration" \
  "$WORK_ROOT/training_runs/e4_exploration" \
  "Blues,Electronic music,Funk,Jazz,Pop music,Reggae,Rock music"

_run_eval "E4 Interpolation" \
  "$WORK_ROOT/training_runs/e4_interpolation" \
  "Blues,Classical music,Country,Hip hop music,Jazz,Pop music,Rock music"

echo ""
echo "========== Beide E4 Genre-Breakdowns fertig =========="
echo "  $WORK_ROOT/training_runs/e4_exploration/genre_breakdown_plot.{pdf,png}"
echo "  $WORK_ROOT/training_runs/e4_interpolation/genre_breakdown_plot.{pdf,png}"
