#!/bin/bash
#
# Slurm: VLM-Scoring für segments_raw.csv (neuester Dataset-Run).
# Ergebnis:
#   - segments_raw_vlm_scored.csv (nur Zeilen mit vorhandener MP4 + vlm_score)
#
# Nutzung:
#   sbatch jobs/vlm_filter_balanced.sh
#

#SBATCH --job-name=vlm_filter
#SBATCH --partition=paula
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00:00

#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/vlm_filter_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/vlm_filter_%j.err

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

python3 "$REPO_ROOT/pipeline/vlm_filter_balanced.py"
exit $?

