#!/bin/bash
#
# Slurm: Qwen3-VL Smoke-Test (Modell laden + eine kurze Generation).
#
# Einmalig im venv (Login-Knoten, mit Netzwerk):
#   pip install -r requirements-vlm.txt
#   pip install "git+https://github.com/huggingface/transformers.git"
#
# Optional Modell vorab nach work2 cachen (spart Zeit im ersten GPU-Job):
#   export HF_HOME=/work2/ra39oxet-DatasetAudioSetSubset/.cache/huggingface
#   huggingface-cli download Qwen/Qwen3-VL-2B-Instruct
#
# Nutzung (Repo-Root):
#   sbatch jobs/vlm_smoke_test.sh
#

#SBATCH --job-name=vlm_smoke
#SBATCH --partition=paula
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00

#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/vlm_smoke_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/vlm_smoke_%j.err

WORK_ROOT="/work2/ra39oxet-DatasetAudioSetSubset"
mkdir -p "$WORK_ROOT/logs"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT" || { echo "FEHLER: cd nach $REPO_ROOT fehlgeschlagen." >&2; exit 1; }

# Hugging Face Cache auf work2 (nicht ins Home vollaufen lassen)
export HF_HOME="${HF_HOME:-$WORK_ROOT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
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

python3 "$REPO_ROOT/pipeline/vlm_smoke_test.py"
exit $?
