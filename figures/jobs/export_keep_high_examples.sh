#!/bin/bash
#
# Frame-Strips für KEEP_HIGH-Beispiele (Thesis-Figuren).
# Output: datasets/<RUN>/keep_high_examples/
#
#   sbatch jobs/export_keep_high_examples.sh
#   sbatch --export=DATASET_RUN_NAME=2026-05-26_15-59-27_audioset jobs/export_keep_high_examples.sh
#

#SBATCH --job-name=kh_examples
#SBATCH --partition=paula
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=0-00:30:00

#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/export_keep_high_examples_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/export_keep_high_examples_%j.err

WORK_ROOT="/work2/ra39oxet-DatasetAudioSetSubset"
mkdir -p "$WORK_ROOT/logs"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Jobs liegen unter <domain>/jobs/ → Repo-Root ist ../..
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/configs" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
fi
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

python3 "$REPO_ROOT/figures/export_keep_high_frame_examples.py"
exit $?
