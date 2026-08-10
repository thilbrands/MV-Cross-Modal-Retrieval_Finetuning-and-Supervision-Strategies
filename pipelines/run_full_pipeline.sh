#!/bin/bash
#
# Full BA-Pipeline: Dataset-Erstellung → Train+Eval (+ Analysen/Plots, optional E4).
#
#   bash pipelines/run_full_pipeline.sh
#   sbatch pipelines/run_full_pipeline.sh
#   SKIP_EXPORTS=1 RUN_E4=0 bash pipelines/run_full_pipeline.sh
#
# Teilpipelines einzeln:
#   bash pipelines/run_dataset_pipeline.sh
#   bash pipelines/run_train_and_eval.sh
#
#SBATCH --job-name=full_pipeline
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

echo "############################################"
echo "# 1/2 Dataset-Pipeline"
echo "############################################"
bash "$REPO_ROOT/pipelines/run_dataset_pipeline.sh"

# Nach Dataset: aktuellen Run an Train+Eval durchreichen
export DATASET_RUN_NAME
if [[ -z "${DATASET_RUN_NAME:-}" ]]; then
  DATASET_RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'configs'); import config; print(config.get_latest_run_name() or '')")
fi
if [[ -z "$DATASET_RUN_NAME" ]]; then
  echo "FEHLER: Kein Dataset-Run nach Dataset-Pipeline." >&2
  exit 1
fi

echo ""
echo "############################################"
echo "# 2/2 Train + Eval + Analysen + Figures"
echo "############################################"
# Gemeinsamer Ausgabeordner für Checkpoints, Eval und alle Figuren
export TRAINING_RUN_DIR
if [[ -z "${TRAINING_RUN_DIR:-}" ]]; then
  TRAINING_RUN_DIR=$(python3 -c "import sys; sys.path.insert(0,'configs'); import config; print(config.get_new_training_run_dir())")
fi
mkdir -p "$TRAINING_RUN_DIR"
echo "Training/Eval/Figures-Dir: $TRAINING_RUN_DIR"

DATASET_RUN_NAME="$DATASET_RUN_NAME" TRAINING_RUN_DIR="$TRAINING_RUN_DIR" \
  bash "$REPO_ROOT/pipelines/run_train_and_eval.sh"

echo ""
echo "========== Full Pipeline fertig =========="
echo "Dataset-Run:      $DATASET_RUN_NAME"
echo "Training-Prozess: $TRAINING_RUN_DIR"
echo "Kuratierte Results:  …/results/$(basename "$TRAINING_RUN_DIR")/{checkpoints,results,outputs,meta[,e4_*]}"
