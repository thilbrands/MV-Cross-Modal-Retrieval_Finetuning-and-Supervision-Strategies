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

#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/eval_quality_split_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/eval_quality_split_%j.err

WORK_ROOT="/work2/ra39oxet-DatasetAudioSetSubset"
mkdir -p "$WORK_ROOT/logs"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT" || { echo "FEHLER: cd nach $REPO_ROOT fehlgeschlagen." >&2; exit 1; }

module purge
module load Python/3.11.5-GCCcore-13.2.0

if [[ ! -f "$HOME/venv/ba/bin/activate" ]]; then
  echo "FEHLER: Venv nicht gefunden ($HOME/venv/ba)." >&2
  exit 1
fi
source "$HOME/venv/ba/bin/activate"

echo "Hostname: $(hostname)"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "DATASET_RUN_NAME: ${DATASET_RUN_NAME:-<neuester Run>}"
echo "TRAINING_RUN_DIR: ${TRAINING_RUN_DIR:-<neuester Run mit projection_heads_*.pt>}"
echo "Starte pipeline/eval_quality_split.py …"

export PYTHONUNBUFFERED=1
python3 "$REPO_ROOT/pipeline/eval_quality_split.py"
exit $?
