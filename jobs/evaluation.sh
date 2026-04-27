#!/bin/bash
#
# Slurm-Job: Evaluation (MRR, Recall@k, Mean Rank, V→A und A→V).
# Nutzt Test-Split + Embeddings eines Runs und Projektions-Heads aus PROJECTION_HEADS_PATH.
#
# Nutzung (vom Repo-Root auf dem Cluster):
#   sbatch jobs/evaluation.sh
#   sbatch --export=DATASET_RUN_NAME=2026-03-13_18-02-30_audioset jobs/evaluation.sh
#

#SBATCH --job-name=evaluation
#SBATCH --partition=paula
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=0-4:00:00

#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/evaluation_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/evaluation_%j.err

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
echo "Starte pipeline/evaluation.py …"

export PYTHONUNBUFFERED=1
python3 "$REPO_ROOT/pipeline/evaluation.py"
exit $?
