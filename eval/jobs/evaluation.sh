#!/bin/bash
#
# Slurm-Job: Evaluation (MRR, Recall@k, Mean Rank, V→A und A→V).
# Nutzt Test-Split + Embeddings eines Runs und Projektions-Heads aus PROJECTION_HEADS_PATH.
#
# Nutzung (vom Repo-Root auf dem Cluster):
#   sbatch jobs/evaluation.sh
#   sbatch --export=DATASET_RUN_NAME=...,TRAINING_RUN_DIR=... jobs/evaluation.sh
#   sbatch --export=DATASET_RUN_NAME=...,TRAINING_RUN_DIR=...,AE_PAIR_RUN_DIR=...,AE_GENRE_RUN_DIR=...,EVAL_OUTPUT_DIR=/work2/.../eval_bootstrap jobs/evaluation.sh
#

#SBATCH --job-name=evaluation
#SBATCH --partition=paula
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=0-4:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Jobs liegen unter <domain>/jobs/ → Repo-Root ist ../..
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/configs" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
fi
cd "$REPO_ROOT" || { echo "FEHLER: cd nach $REPO_ROOT fehlgeschlagen." >&2; exit 1; }
# shellcheck disable=SC1091
source "$REPO_ROOT/configs/cluster_env.sh"
mkdir -p "$REPO_ROOT/logs"

module purge
module load Python/3.11.5-GCCcore-13.2.0

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "FEHLER: Venv nicht gefunden ($VENV_ACTIVATE)." >&2
  exit 1
fi
source "$VENV_ACTIVATE"

echo "Hostname: $(hostname)"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "DATASET_RUN_NAME: ${DATASET_RUN_NAME:-<neuester Run>}"
echo "Starte eval/evaluation.py …"

export PYTHONUNBUFFERED=1
python3 "$REPO_ROOT/eval/evaluation.py"
exit $?
