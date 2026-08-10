#!/bin/bash
#
# Slurm-Job: Evaluation aufgesplittet nach VLM-Qualität (KEEP_HIGH vs. KEEP_LOW).
# Baseline + E1 + E2, MRR & Recall@10, Protocol A/B, V→A und A→V.
# Speichert results_quality_split.csv im Ausgabeordner.
#
# Nutzung (vom Repo-Root auf dem Cluster):
#   sbatch jobs/eval_quality_split.sh
#   sbatch --export=DATASET_RUN_NAME=2026-05-26_15-59-27_audioset,TRAINING_RUN_DIR=/work2/ra39oxet-DatasetAudioSetSubset/training_runs/2026-07-04_19-06 jobs/eval_quality_split.sh
#

#SBATCH --job-name=eval_quality_split
#SBATCH --partition=paula
#SBATCH --cpus-per-task=8
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00

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
echo "TRAINING_RUN_DIR: ${TRAINING_RUN_DIR:-<neuester Run mit projection_heads_*.pt>}"
echo "Starte eval/eval_quality_split.py …"

export PYTHONUNBUFFERED=1
python3 "$REPO_ROOT/eval/eval_quality_split.py"
exit $?
