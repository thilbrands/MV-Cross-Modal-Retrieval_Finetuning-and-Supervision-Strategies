#!/bin/bash
#
# Slurm-Job: Extraktion + Embedding (CLIP/Wav2CLIP) aus MP4s eines Dataset-Runs.
# Pro Video: Frames und Audio nur im RAM, sofort embedden, nur embeddings/ speichern.
#
# Nutzung (vom Repo-Root auf dem Cluster):
#   sbatch jobs/extract_and_embed_videos.sh
#     → nutzt automatisch den neuesten Dataset-Run (Default)
#   sbatch --export=DATASET_RUN_NAME=2026-03-13_18-02-30_audioset jobs/extract_and_embed_videos.sh
#     → nutzt den angegebenen Run
#

#SBATCH --job-name=extract_embed
#SBATCH --partition=paula
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00

#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/extract_embed_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/extract_embed_%j.err

WORK_ROOT="/work2/ra39oxet-DatasetAudioSetSubset"
mkdir -p "$WORK_ROOT/logs"

# DATASET_RUN_NAME optional: wenn nicht gesetzt, wählt das Skript den neuesten Run (Default)

# Repo-Root: Verzeichnis, aus dem sbatch aufgerufen wurde (Fallback: übergeordnetes Verzeichnis von jobs/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Jobs liegen unter <domain>/jobs/ → Repo-Root ist ../..
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/configs" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
fi
cd "$REPO_ROOT" || { echo "FEHLER: cd nach $REPO_ROOT fehlgeschlagen." >&2; exit 1; }

module purge
module load Python/3.11.5-GCCcore-13.2.0

# Venv „ba“ aktivieren (für numpy, torch, clip, wav2clip, cv2, av, librosa …)
if [[ ! -f "$HOME/venv/ba/bin/activate" ]]; then
  echo "FEHLER: Venv nicht gefunden ($HOME/venv/ba). Bitte zuerst Venv-Setup ausführen." >&2
  exit 1
fi
source "$HOME/venv/ba/bin/activate"

echo "Hostname: $(hostname)"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "WORK_ROOT: $WORK_ROOT"
echo "DATASET_RUN_NAME: $DATASET_RUN_NAME"
echo "Starte data/extract_and_embed_videos.py …"

export PYTHONUNBUFFERED=1
python3 "$REPO_ROOT/data/extract_and_embed_videos.py"
EXIT_CODE=$?

echo "Job beendet."
exit $EXIT_CODE
