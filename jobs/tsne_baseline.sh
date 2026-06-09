#!/bin/bash
#
# Slurm-Job: t-SNE-Baseline-Visualisierung (frozen CLIP + Wav2CLIP, Test-Split).
# Liest die vorab berechneten Embeddings, berechnet t-SNE und speichert tsne_baseline.pdf.
#
# Nutzung (vom Repo-Root auf dem Cluster):
#   sbatch jobs/tsne_baseline.sh
#     → nutzt automatisch den neuesten Run (Default)
#   sbatch --export=DATASET_RUN_NAME=2026-05-26_15-59-27_audioset jobs/tsne_baseline.sh
#     → nutzt den angegebenen Run
#

#SBATCH --job-name=tsne_baseline
#SBATCH --partition=paula
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=0-00:30:00

#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/tsne_baseline_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/tsne_baseline_%j.err

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
echo "Starte tsne_baseline.py …"

export PYTHONUNBUFFERED=1
python3 "$REPO_ROOT/tsne_baseline.py"
exit $?
