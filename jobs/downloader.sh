#!/bin/bash
#
# Slurm-Job für AudioSet-Download auf dem Uni-Cluster.
# Alle Ausgabe (Logs, CSVs, Downloads) geht nach work2.
#
# Nutzung (vom Repo-Root auf dem Cluster):
#   sbatch jobs/downloader.sh
#

#SBATCH --job-name=audioset_download
#SBATCH --partition=paula
#SBATCH --cpus-per-task=48
#SBATCH --mem=128GB
#SBATCH --time=1-00:00:00

#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/audioset_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/audioset_%j.err

WORK_ROOT="/work2/ra39oxet-DatasetAudioSetSubset"
mkdir -p "$WORK_ROOT/logs"

# Repo-Root: Verzeichnis, aus dem sbatch aufgerufen wurde (Fallback: übergeordnetes Verzeichnis von jobs/)
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT" || { echo "FEHLER: cd nach $REPO_ROOT fehlgeschlagen." >&2; exit 1; }

module purge

# FFmpeg: Conda-Env mit HTTPS (Default $HOME/conda_ffmpeg)
if [[ -z "${CONDA_ENV_FFMPEG:-}" && -d "$HOME/conda_ffmpeg" ]]; then
  export CONDA_ENV_FFMPEG="$HOME/conda_ffmpeg"
fi
if [[ -n "${CONDA_ENV_FFMPEG:-}" ]]; then
  module load Anaconda3
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV_FFMPEG"
  echo "Conda-Env für FFmpeg: $CONDA_ENV_FFMPEG"
else
  module load Python/3.11.5-GCCcore-13.2.0
fi

# Venv „ba“ aktivieren (Python/yt-dlp; FFmpeg von Conda oder Modul/statisch)
if [[ ! -f "$HOME/venv/ba/bin/activate" ]]; then
  echo "FEHLER: Venv nicht gefunden ($HOME/venv/ba). Bitte zuerst Venv-Setup ausführen." >&2
  exit 1
fi
source "$HOME/venv/ba/bin/activate"

if [[ -x "$HOME/venv/ba/bin/ffmpeg" ]]; then
  echo "Nutze FFmpeg aus venv (ba)."
elif [[ -n "${CONDA_ENV_FFMPEG:-}" ]]; then
  :
else
  if [[ -n "${FFMPEG_STATIC:-}" && -x "${FFMPEG_STATIC}/ffmpeg" ]]; then
    export PATH="${FFMPEG_STATIC}:$PATH"
    echo "Nutze statisches FFmpeg: ${FFMPEG_STATIC}/ffmpeg"
  else
    module load FFmpeg/6.0-GCCcore-12.3.0
  fi
fi

export AUDIOSET_DOWNLOAD_WORKERS="${AUDIOSET_DOWNLOAD_WORKERS:-32}"

echo "Hostname: $(hostname)"
echo "ffmpeg: $(which ffmpeg 2>/dev/null || echo 'NICHT GEFUNDEN')"
echo "Download-Worker: $AUDIOSET_DOWNLOAD_WORKERS"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "WORK_ROOT: $WORK_ROOT"
echo "Starte pipeline/downloader.py …"

export PYTHONUNBUFFERED=1
python3 "$REPO_ROOT/pipeline/downloader.py"
EXIT_CODE=$?
echo "Job beendet."
exit $EXIT_CODE
