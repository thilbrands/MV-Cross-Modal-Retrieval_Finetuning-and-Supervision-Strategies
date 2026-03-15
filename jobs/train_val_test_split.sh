#!/bin/bash
#
# Slurm-Job: Train/Val/Test-Split (stratifiziert) für einen Dataset-Run.
# Liest segments_balanced.csv + Embeddings, schreibt train_val_test_split.csv.
#
# Nutzung (vom Repo-Root auf dem Cluster):
#   sbatch jobs/train_val_test_split.sh
#     → nutzt automatisch den neuesten Run (Default)
#   sbatch --export=DATASET_RUN_NAME=2026-03-13_18-02-30_audioset jobs/train_val_test_split.sh
#     → nutzt den angegebenen Run
#

#SBATCH --job-name=split
#SBATCH --partition=paula
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=0-00:30:00

#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/split_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/split_%j.err

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
echo "Starte pipeline/train_val_test_split.py …"

export PYTHONUNBUFFERED=1
python3 "$REPO_ROOT/pipeline/train_val_test_split.py"
exit $?
