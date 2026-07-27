#!/bin/bash
#
# Train + Eval + BA-Analysen/Plots (ein gemeinsamer Training-Run-Ordner).
#
# Ablauf:
#   1–4) E1 Pair → E2 Genre → E3a AE-Pair → E3b AE-Genre
#   5)   Evaluation (Protocol A/B + Bootstrap-CIs)
#   6)   Quality-Split (KEEP_HIGH vs KEEP_LOW)
#   7)   Comparison-Eval (related work)
#   8)   Genre-Centroid-Margin (+ Confusion)
#   9)   Plots + Thesis-Figuren (alles nach TRAINING_RUN_DIR / PLOT_OUTPUT_DIR)
#  10)   optional E4 Exploration + Interpolation + Genre-Breakdown  (RUN_E4=1, default)
#
# Nutzung:
#   bash pipelines/run_train_and_eval.sh
#   sbatch pipelines/run_train_and_eval.sh
#   DATASET_RUN_NAME=... RUN_E4=0 bash pipelines/run_train_and_eval.sh
#
#SBATCH --job-name=train_eval_orch
#SBATCH --partition=paula
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=0-72:00:00
#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/train_eval_orch_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/train_eval_orch_%j.err

set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Orchestratoren liegen unter pipelines/ → Repo-Root ist eine Ebene höher
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/configs" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
elif [[ -d "$_SCRIPT_DIR/../configs" ]]; then
  REPO_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"
else
  REPO_ROOT="$_SCRIPT_DIR"
fi
cd "$REPO_ROOT" || { echo "FEHLER: cd nach $REPO_ROOT fehlgeschlagen." >&2; exit 1; }
mkdir -p /work2/ra39oxet-DatasetAudioSetSubset/logs

wait_job() {
  local jid="$1"
  echo "Warte auf Job $jid …"
  while squeue -j "$jid" 2>/dev/null | grep -q "$jid"; do sleep 60; done
}

export DATASET_RUN_NAME
if [[ -z "${DATASET_RUN_NAME:-}" ]]; then
  DATASET_RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'configs'); import config; print(config.get_latest_run_name() or '')")
fi
if [[ -z "$DATASET_RUN_NAME" ]]; then
  echo "FEHLER: DATASET_RUN_NAME nicht gesetzt und kein Run unter datasets/." >&2
  exit 1
fi

export TRAINING_RUN_DIR
if [[ -z "${TRAINING_RUN_DIR:-}" ]]; then
  TRAINING_RUN_DIR=$(python3 -c "import sys; sys.path.insert(0,'configs'); import config; print(config.get_new_training_run_dir())")
fi
mkdir -p "$TRAINING_RUN_DIR"

export AE_PAIR_RUN_DIR="${AE_PAIR_RUN_DIR:-$TRAINING_RUN_DIR}"
export AE_GENRE_RUN_DIR="${AE_GENRE_RUN_DIR:-$TRAINING_RUN_DIR}"
export PLOT_OUTPUT_DIR="${PLOT_OUTPUT_DIR:-$TRAINING_RUN_DIR}"
# Eval schreibt standardmäßig in TRAINING_RUN_DIR; optional separater Unterordner:
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$TRAINING_RUN_DIR}"

RUN_E4="${RUN_E4:-1}"

echo "Dataset-Run:      $DATASET_RUN_NAME"
echo "Training-Run-Dir: $TRAINING_RUN_DIR"
echo "Eval-Output-Dir:  $EVAL_OUTPUT_DIR"
echo "RUN_E4:           $RUN_E4"
echo ""

echo "========== 1/9 Pair-Training (E1) =========="
JOB=$(sbatch --parsable --export=DATASET_RUN_NAME,TRAINING_RUN_DIR \
  --output="$TRAINING_RUN_DIR/training_pair.out" --error="$TRAINING_RUN_DIR/training_pair.err" \
  training/jobs/pair_based_training.sh)
echo "Job: $JOB"; wait_job "$JOB"

echo ""
echo "========== 2/9 Genre-Training (E2) =========="
JOB=$(sbatch --parsable --export=DATASET_RUN_NAME,TRAINING_RUN_DIR \
  --output="$TRAINING_RUN_DIR/training_genre.out" --error="$TRAINING_RUN_DIR/training_genre.err" \
  training/jobs/genre_based_training.sh)
echo "Job: $JOB"; wait_job "$JOB"

echo ""
echo "========== 3/9 Audio-Encoder Pair (E3a) =========="
JOB=$(sbatch --parsable --export=DATASET_RUN_NAME,TRAINING_RUN_DIR \
  --output="$TRAINING_RUN_DIR/ae_pair_training.out" --error="$TRAINING_RUN_DIR/ae_pair_training.err" \
  training/jobs/audio_encoder_pair_training.sh)
echo "Job: $JOB"; wait_job "$JOB"

echo ""
echo "========== 4/9 Audio-Encoder Genre (E3b) =========="
JOB=$(sbatch --parsable --export=DATASET_RUN_NAME,TRAINING_RUN_DIR \
  --output="$TRAINING_RUN_DIR/ae_genre_training.out" --error="$TRAINING_RUN_DIR/ae_genre_training.err" \
  training/jobs/audio_encoder_genre_training.sh)
echo "Job: $JOB"; wait_job "$JOB"

echo ""
echo "========== 5/9 Evaluation (A/B + Bootstrap) =========="
JOB=$(sbatch --parsable --export=DATASET_RUN_NAME,TRAINING_RUN_DIR,AE_PAIR_RUN_DIR,AE_GENRE_RUN_DIR,EVAL_OUTPUT_DIR \
  --output="$TRAINING_RUN_DIR/evaluation.out" --error="$TRAINING_RUN_DIR/evaluation.err" \
  eval/jobs/evaluation.sh)
echo "Job: $JOB"; wait_job "$JOB"

echo ""
echo "========== 6/9 Quality-Split Eval =========="
JOB=$(sbatch --parsable --export=DATASET_RUN_NAME,TRAINING_RUN_DIR,AE_PAIR_RUN_DIR,AE_GENRE_RUN_DIR \
  --output="$TRAINING_RUN_DIR/eval_quality_split.out" --error="$TRAINING_RUN_DIR/eval_quality_split.err" \
  eval/jobs/eval_quality_split.sh)
echo "Job: $JOB"; wait_job "$JOB"

echo ""
echo "========== 7/9 Comparison-Eval (related work) =========="
JOB=$(sbatch --parsable --export=DATASET_RUN_NAME,TRAINING_RUN_DIR,AE_PAIR_RUN_DIR,AE_GENRE_RUN_DIR \
  --output="$TRAINING_RUN_DIR/comparison_eval.out" --error="$TRAINING_RUN_DIR/comparison_eval.err" \
  eval/jobs/comparison_eval.sh)
echo "Job: $JOB"; wait_job "$JOB"

echo ""
echo "========== 8/9 Genre-Centroid-Margin =========="
JOB=$(sbatch --parsable --export=DATASET_RUN_NAME,TRAINING_RUN_DIR,AE_PAIR_RUN_DIR,AE_GENRE_RUN_DIR,PLOT_OUTPUT_DIR \
  --output="$TRAINING_RUN_DIR/genre_centroid_margin.out" --error="$TRAINING_RUN_DIR/genre_centroid_margin.err" \
  eval/jobs/genre_centroid_margin.sh)
echo "Job: $JOB"; wait_job "$JOB"

echo ""
echo "========== 9/9 Plots + Thesis-Figuren → TRAINING_RUN_DIR =========="
# Alle Figuren + Eval-Artefakte landen in TRAINING_RUN_DIR (PLOT_OUTPUT_DIR).
export PLOT_OUTPUT_DIR="$TRAINING_RUN_DIR"

JOB=$(sbatch --parsable --export=TRAINING_RUN_DIR,PLOT_OUTPUT_DIR \
  --output="$TRAINING_RUN_DIR/plot_training_curves.out" --error="$TRAINING_RUN_DIR/plot_training_curves.err" \
  figures/jobs/plot_training_curves.sh)
echo "Job Curves: $JOB"; wait_job "$JOB"

JOB=$(sbatch --parsable --export=DATASET_RUN_NAME,TRAINING_RUN_DIR,AE_PAIR_RUN_DIR,AE_GENRE_RUN_DIR,PLOT_OUTPUT_DIR \
  --output="$TRAINING_RUN_DIR/tsne_embeddings.out" --error="$TRAINING_RUN_DIR/tsne_embeddings.err" \
  figures/jobs/tsne_embeddings.sh)
echo "Job t-SNE Embeddings: $JOB"; wait_job "$JOB"

JOB=$(sbatch --parsable --export=DATASET_RUN_NAME,TRAINING_RUN_DIR,AE_PAIR_RUN_DIR,AE_GENRE_RUN_DIR,PLOT_OUTPUT_DIR \
  --output="$TRAINING_RUN_DIR/tsne_baseline.out" --error="$TRAINING_RUN_DIR/tsne_baseline.err" \
  figures/jobs/tsne_baseline.sh)
echo "Job t-SNE Baseline: $JOB"; wait_job "$JOB"

JOB=$(sbatch --parsable --export=DATASET_RUN_NAME,TRAINING_RUN_DIR,AE_PAIR_RUN_DIR,AE_GENRE_RUN_DIR,PLOT_OUTPUT_DIR \
  --output="$TRAINING_RUN_DIR/similarity_histograms_e1e2.out" --error="$TRAINING_RUN_DIR/similarity_histograms_e1e2.err" \
  figures/jobs/similarity_histograms_e1e2.sh)
echo "Job Hist E1/E2: $JOB"; wait_job "$JOB"

JOB=$(sbatch --parsable --export=DATASET_RUN_NAME,TRAINING_RUN_DIR,AE_PAIR_RUN_DIR,AE_GENRE_RUN_DIR,PLOT_OUTPUT_DIR \
  --output="$TRAINING_RUN_DIR/similarity_histograms.out" --error="$TRAINING_RUN_DIR/similarity_histograms.err" \
  figures/jobs/similarity_histograms.sh)
echo "Job Hist E3: $JOB"; wait_job "$JOB"

JOB=$(sbatch --parsable --export=DATASET_RUN_NAME,PLOT_OUTPUT_DIR \
  --output="$TRAINING_RUN_DIR/genre_similarity.out" --error="$TRAINING_RUN_DIR/genre_similarity.err" \
  figures/jobs/genre_similarity.sh)
echo "Job Genre-Similarity: $JOB"; wait_job "$JOB"

# Frame-Exports (falls in Dataset-Pipeline schon erzeugt: spiegeln; sonst neu erzeugen)
DS_DIR=$(python3 -c "import sys; sys.path.insert(0,'configs'); import config; print(config.DATASETS_ROOT / '$DATASET_RUN_NAME')")
for name in remove_examples keep_low_examples keep_high_examples; do
  if [[ -d "$DS_DIR/$name" ]]; then
    mkdir -p "$TRAINING_RUN_DIR/$name"
    cp -a "$DS_DIR/$name/." "$TRAINING_RUN_DIR/$name/"
    echo "Kopiert: $name → $TRAINING_RUN_DIR/$name"
  fi
done
if [[ ! -d "$TRAINING_RUN_DIR/remove_examples" ]]; then
  JOB=$(sbatch --parsable --export=DATASET_RUN_NAME,PLOT_OUTPUT_DIR,TRAINING_RUN_DIR \
    --output="$TRAINING_RUN_DIR/export_remove.out" --error="$TRAINING_RUN_DIR/export_remove.err" \
    figures/jobs/export_remove_examples.sh)
  echo "Job REMOVE-Exports: $JOB"; wait_job "$JOB"
fi
if [[ ! -d "$TRAINING_RUN_DIR/keep_low_examples" ]]; then
  JOB=$(sbatch --parsable --export=DATASET_RUN_NAME,PLOT_OUTPUT_DIR,TRAINING_RUN_DIR \
    --output="$TRAINING_RUN_DIR/export_keep_low.out" --error="$TRAINING_RUN_DIR/export_keep_low.err" \
    figures/jobs/export_keep_low_examples.sh)
  echo "Job KEEP_LOW-Exports: $JOB"; wait_job "$JOB"
fi
if [[ ! -d "$TRAINING_RUN_DIR/keep_high_examples" ]]; then
  JOB=$(sbatch --parsable --export=DATASET_RUN_NAME,PLOT_OUTPUT_DIR,TRAINING_RUN_DIR \
    --output="$TRAINING_RUN_DIR/export_keep_high.out" --error="$TRAINING_RUN_DIR/export_keep_high.err" \
    figures/jobs/export_keep_high_examples.sh)
  echo "Job KEEP_HIGH-Exports: $JOB"; wait_job "$JOB"
fi

if [[ "$RUN_E4" == "1" ]]; then
  echo ""
  echo "========== E4 Exploration =========="
  # run_e4_*.sh sind selbst Orchestratoren (sbatch intern) — hier sequentiell via bash
  DATASET_RUN_NAME="$DATASET_RUN_NAME" bash "$REPO_ROOT/generalization_experiment/run_e4_exploration.sh"
  echo ""
  echo "========== E4 Interpolation =========="
  DATASET_RUN_NAME="$DATASET_RUN_NAME" bash "$REPO_ROOT/generalization_experiment/run_e4_interpolation.sh"
fi

echo ""
echo "========== Kuratierte results/ packen =========="
PACKED=$(python3 -c "import sys; sys.path.insert(0,'configs'); import config; print(config.package_run_to_results(r'''$TRAINING_RUN_DIR'''))")
echo "Kuratiert: $PACKED/{checkpoints,results,outputs,meta}"

if [[ "$RUN_E4" == "1" ]]; then
  WORK_ROOT=$(python3 -c "import sys; sys.path.insert(0,'configs'); import config; print(config.WORK_ROOT)")
  for e4 in e4_exploration e4_interpolation; do
    E4_SRC="$WORK_ROOT/training_runs/$e4"
    if [[ -d "$E4_SRC" ]]; then
      E4_DEST=$(python3 -c "import sys; sys.path.insert(0,'configs'); import config; print(config.package_run_to_results(r'''$E4_SRC''', r'''$PACKED/$e4'''))")
      echo "Kuratiert E4: $E4_DEST/{checkpoints,results,outputs,meta}"
    else
      echo "Hinweis: $E4_SRC fehlt — E4 nicht kuratiert."
    fi
  done
fi

echo ""
echo "========== Train+Eval-Pipeline fertig =========="
echo "Dataset-Run:     $DATASET_RUN_NAME"
echo "Training-Prozess: $TRAINING_RUN_DIR"
echo "  (Checkpoints, Eval, Figures — flach im Run-Ordner)"
echo "Kuratierte Results: $PACKED"
echo "  checkpoints/  results/  outputs/  meta/"
if [[ "$RUN_E4" == "1" ]]; then
  echo "  e4_exploration/  e4_interpolation/"
fi
