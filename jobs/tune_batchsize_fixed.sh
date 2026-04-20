#!/bin/bash
#
# Slurm-Job: Kleine Batchsize-Tuning-Pipeline (Pair + Genre, jeweils Top-1 + bestes MLP).
# Output wie gewohnt unter training_runs auf work2.
#
# Nutzung:
# sbatch --export=PAIR_RESULTS_CSV=/work2/ra39oxet-DatasetAudioSetSubset/training_runs/tuning_pair_YYYY-MM-DD_HH-MM-SS/results.csv,GENRE_RESULTS_CSV=/work2/ra39oxet-DatasetAudioSetSubset/training_runs/tuning_genre_YYYY-MM-DD_HH-MM-SS/results.csv jobs/tune_batchsize_fixed.sh
# oder ohne CSV-Pfade:
# sbatch jobs/tune_batchsize_fixed.sh
#

#SBATCH --job-name=tune_bs_fixed
#SBATCH --partition=paula
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00:00

#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/tune_bs_fixed_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/tune_bs_fixed_%j.err

set -euo pipefail

# Robust gegen unbound variables unter `set -u`
: "${PAIR_RESULTS_CSV:=}"
: "${GENRE_RESULTS_CSV:=}"
: "${DATASET_RUN_NAME:=}"

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
export BATCH_SIZES="${BATCH_SIZES:-32 64 128 256}"
export HP_MAX_EPOCHS="${HP_MAX_EPOCHS:-20}"
export HP_PATIENCE="${HP_PATIENCE:-3}"

echo "Hostname: $(hostname)"
echo "Slurm Job ID: $SLURM_JOB_ID"
if [[ -z "${PAIR_RESULTS_CSV:-}" ]]; then
  PAIR_RESULTS_CSV="$(python3 - <<'PY'
import sys
sys.path.insert(0, ".")
import config
root = config.TRAINING_RUNS_ROOT
paths = [d / "results.csv" for d in root.glob("tuning_pair_*") if (d / "results.csv").exists()]
paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
print(str(paths[0]) if paths else "")
PY
)"
  export PAIR_RESULTS_CSV
fi
if [[ -z "${GENRE_RESULTS_CSV:-}" ]]; then
  GENRE_RESULTS_CSV="$(python3 - <<'PY'
import sys
sys.path.insert(0, ".")
import config
root = config.TRAINING_RUNS_ROOT
paths = [d / "results.csv" for d in root.glob("tuning_genre_*") if (d / "results.csv").exists()]
paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
print(str(paths[0]) if paths else "")
PY
)"
  export GENRE_RESULTS_CSV
fi
if [[ -z "${PAIR_RESULTS_CSV:-}" || -z "${GENRE_RESULTS_CSV:-}" ]]; then
  echo "FEHLER: Keine results.csv gefunden. Setze PAIR_RESULTS_CSV und GENRE_RESULTS_CSV explizit." >&2
  exit 1
fi

if [[ -z "${DATASET_RUN_NAME:-}" ]]; then
  DATASET_RUN_NAME="$(python3 -c "import sys; sys.path.insert(0,'.'); import config; print(config.get_latest_run_name() or '')")"
  export DATASET_RUN_NAME
fi
if [[ -z "${DATASET_RUN_NAME:-}" ]]; then
  echo "FEHLER: DATASET_RUN_NAME nicht gesetzt und kein Run unter DATASETS_ROOT gefunden." >&2
  exit 1
fi

echo "DATASET_RUN_NAME: $DATASET_RUN_NAME"
echo "PAIR_RESULTS_CSV: $PAIR_RESULTS_CSV"
echo "GENRE_RESULTS_CSV: $GENRE_RESULTS_CSV"
echo "BATCH_SIZES=$BATCH_SIZES | HP_MAX_EPOCHS=$HP_MAX_EPOCHS | HP_PATIENCE=$HP_PATIENCE"
echo "Starte fixed batchsize tuning …"

echo "========== 1/2 Pair =========="
TRAINING_TYPE=pair RESULTS_CSV="$PAIR_RESULTS_CSV" python3 "$REPO_ROOT/pipeline/tune_batchsize_fixed.py"

echo ""
echo "========== 2/2 Genre =========="
TRAINING_TYPE=genre RESULTS_CSV="$GENRE_RESULTS_CSV" python3 "$REPO_ROOT/pipeline/tune_batchsize_fixed.py"

echo ""
echo "========== Fertig =========="
exit $?
