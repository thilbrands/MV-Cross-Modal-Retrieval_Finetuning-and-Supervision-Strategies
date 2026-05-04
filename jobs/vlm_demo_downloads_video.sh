#!/bin/bash
#
# Slurm: VLM auf der ersten *.mp4 in datasets/<neuester Run>/downloads/ — drei Frames wie extract_and_embed.
#
#   sbatch jobs/vlm_demo_downloads_video.sh
#

#SBATCH --job-name=vlm_demo_vid
#SBATCH --partition=paula
#SBATCH --cpus-per-task=2
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-01:00:00

#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/vlm_demo_vid_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/vlm_demo_vid_%j.err

WORK_ROOT="/work2/ra39oxet-DatasetAudioSetSubset"
mkdir -p "$WORK_ROOT/logs"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT" || { echo "FEHLER: cd nach $REPO_ROOT fehlgeschlagen." >&2; exit 1; }

export HF_HOME="${HF_HOME:-$WORK_ROOT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

module purge
module load Python/3.11.5-GCCcore-13.2.0

if [[ ! -f "$HOME/venv/ba/bin/activate" ]]; then
  echo "FEHLER: Venv nicht gefunden ($HOME/venv/ba)." >&2
  exit 1
fi
source "$HOME/venv/ba/bin/activate"

export PYTHONUNBUFFERED=1

echo "Hostname: $(hostname) | Job: $SLURM_JOB_ID"
echo "HF_HOME=$HF_HOME"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python3 "$REPO_ROOT/pipeline/vlm_demo_downloads_video.py"
exit $?
