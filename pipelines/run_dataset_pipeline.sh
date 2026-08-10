#!/bin/bash
#
# Dataset-Erstellungs-Pipeline (korrekt für die BA):
#   1) Download
#   2) VLM-Filter (KEEP_HIGH / KEEP_LOW / REMOVE)
#   3) Train/Val/Test-Split (stratifiziert, balanciert)
#   4) Extract + Embed (CLIP / Wav2CLIP)
#   5) Raw-Audio für E3
#   6) Frame-Beispiel-Exports (REMOVE / KEEP_LOW / KEEP_HIGH) + Genre-Similarity (Fig. 8)
#
# Nutzung (Repo-Root auf dem Cluster):
#   bash pipelines/run_dataset_pipeline.sh
#   sbatch pipelines/run_dataset_pipeline.sh
#   SKIP_EXPORTS=1 bash pipelines/run_dataset_pipeline.sh   # ohne Frame-Exports / Genre-Similarity
#
#SBATCH --job-name=dataset_orch
#SBATCH --partition=paula
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

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
# shellcheck disable=SC1091
source "$REPO_ROOT/configs/cluster_env.sh"
mkdir -p "$REPO_ROOT/logs"
bash "$REPO_ROOT/configs/check_setup.sh"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load Python/3.11.5-GCCcore-13.2.0 >/dev/null 2>&1 || true
fi
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_ACTIVATE"
fi

wait_job() {
  local jid="$1"
  echo "Warte auf Job $jid …"
  while squeue -j "$jid" 2>/dev/null | grep -q "$jid"; do sleep 60; done
}

resolve_latest_run() {
  python3 -c "import sys; sys.path.insert(0,'configs'); import config; print(config.get_latest_run_name() or '')"
}

SKIP_EXPORTS="${SKIP_EXPORTS:-0}"

echo "========== 1/6 Downloader (neuer Dataset-Run) =========="
JOB=$(sbatch --parsable --export=WORK_ROOT,VENV_ACTIVATE data/jobs/downloader.sh)
echo "Job: $JOB"
wait_job "$JOB"
echo "Downloader beendet."

export DATASET_RUN_NAME
DATASET_RUN_NAME="$(resolve_latest_run)"
if [[ -z "$DATASET_RUN_NAME" ]]; then
  echo "FEHLER: Kein Dataset-Run nach Download gefunden." >&2
  exit 1
fi
echo "Dataset-Run: $DATASET_RUN_NAME"

echo ""
echo "========== 2/6 VLM-Filter =========="
JOB=$(sbatch --parsable --export=WORK_ROOT,VENV_ACTIVATE,DATASET_RUN_NAME data/jobs/vlm_filter_balanced.sh)
echo "Job: $JOB"
wait_job "$JOB"
echo "VLM-Filter beendet."

echo ""
echo "========== 3/6 Train/Val/Test-Split =========="
JOB=$(sbatch --parsable --export=WORK_ROOT,VENV_ACTIVATE,DATASET_RUN_NAME data/jobs/train_val_test_split.sh)
echo "Job: $JOB"
wait_job "$JOB"
echo "Split beendet."

echo ""
echo "========== 4/6 Extract + Embed =========="
JOB=$(sbatch --parsable --export=WORK_ROOT,VENV_ACTIVATE,DATASET_RUN_NAME data/jobs/extract_and_embed_videos.sh)
echo "Job: $JOB"
wait_job "$JOB"
echo "Extract+Embed beendet."

echo ""
echo "========== 5/6 Raw-Audio (E3) =========="
JOB=$(sbatch --parsable --export=WORK_ROOT,VENV_ACTIVATE,DATASET_RUN_NAME data/jobs/extract_raw_audio.sh)
echo "Job: $JOB"
wait_job "$JOB"
echo "Raw-Audio beendet."

if [[ "$SKIP_EXPORTS" != "1" ]]; then
  echo ""
  echo "========== 6/6 Thesis-Figuren (Exports + Genre-Similarity) =========="
  JOB=$(sbatch --parsable --export=WORK_ROOT,VENV_ACTIVATE,DATASET_RUN_NAME figures/jobs/export_remove_examples.sh)
  echo "Job REMOVE-Exports: $JOB"; wait_job "$JOB"
  JOB=$(sbatch --parsable --export=WORK_ROOT,VENV_ACTIVATE,DATASET_RUN_NAME figures/jobs/export_keep_low_examples.sh)
  echo "Job KEEP_LOW-Exports: $JOB"; wait_job "$JOB"
  JOB=$(sbatch --parsable --export=WORK_ROOT,VENV_ACTIVATE,DATASET_RUN_NAME figures/jobs/export_keep_high_examples.sh)
  echo "Job KEEP_HIGH-Exports: $JOB"; wait_job "$JOB"
  JOB=$(sbatch --parsable --export=WORK_ROOT,VENV_ACTIVATE,DATASET_RUN_NAME figures/jobs/genre_similarity.sh)
  echo "Job Genre-Similarity: $JOB"; wait_job "$JOB"
  JOB=$(sbatch --parsable --export=WORK_ROOT,VENV_ACTIVATE,DATASET_RUN_NAME figures/jobs/tsne_baseline.sh)
  echo "Job t-SNE Baseline: $JOB"; wait_job "$JOB"
else
  echo ""
  echo "========== 6/6 übersprungen (SKIP_EXPORTS=1) =========="
fi

echo ""
echo "========== Dataset-Pipeline fertig =========="
echo "Dataset-Run: $DATASET_RUN_NAME"
echo "  datasets/$DATASET_RUN_NAME/"
echo "    segments_*, train_val_test_split.csv, downloads/, embeddings/, …"
