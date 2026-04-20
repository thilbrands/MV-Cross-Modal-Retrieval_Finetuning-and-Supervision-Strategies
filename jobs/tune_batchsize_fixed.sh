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
echo "DATASET_RUN_NAME: ${DATASET_RUN_NAME:-<neuester Run>}"
echo "PAIR_RESULTS_CSV: $PAIR_RESULTS_CSV"
echo "GENRE_RESULTS_CSV: $GENRE_RESULTS_CSV"
echo "BATCH_SIZES=$BATCH_SIZES | HP_MAX_EPOCHS=$HP_MAX_EPOCHS | HP_PATIENCE=$HP_PATIENCE"
echo "Starte run_tune_batchsize_fixed.sh …"

bash "$REPO_ROOT/run_tune_batchsize_fixed.sh"
exit $?
