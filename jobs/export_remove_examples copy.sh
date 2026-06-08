#!/bin/bash
#
# Frame-Strips für KEEP_LOW-Beispiele (Thesis-Figuren).
# Output: datasets/<RUN>/keep_low_examples/
#
#   sbatch jobs/export_keep_low_examples.sh
#   sbatch --export=DATASET_RUN_NAME=2026-05-05_11-42-52_audioset jobs/export_keep_low_examples.sh
#

#SBATCH --job-name=rm_examples
#SBATCH --partition=paula
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=0-00:30:00

#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/export_keep_low_examples_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/export_keep_low_examples_%j.err

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

export PYTHONUNBUFFERED=1

echo "Hostname: $(hostname) | Job: $SLURM_JOB_ID"
echo "DATASET_RUN_NAME: ${DATASET_RUN_NAME:-<neuester Run>}"

python3 "$REPO_ROOT/helper_scripts/export_keep_low_frame_examples.py"
exit $?
