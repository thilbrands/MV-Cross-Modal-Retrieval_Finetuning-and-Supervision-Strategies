#!/bin/bash
#
# Slurm-Job: Batchsize-Sweep auf Top-Konfigurationen aus einer vorhandenen results.csv.
# Entspricht dem Stil der bestehenden jobs/*.sh Skripte.
#
# Nutzung (vom Repo-Root auf dem Cluster):
#   sbatch --export=RESULTS_CSV=/work2/ra39oxet-DatasetAudioSetSubset/training_runs/tuning_genre_YYYY-MM-DD_HH-MM-SS/results.csv jobs/batchsize_sweep_top_configs.sh
#   sbatch --export=RESULTS_CSV=/work2/.../results.csv,DATASET_RUN_NAME=2026-03-25_13-12-06_audioset jobs/batchsize_sweep_top_configs.sh
#   sbatch --export=RESULTS_CSV=/work2/.../results.csv,BATCH_SIZES="32 64 128 256",HP_MAX_EPOCHS=20,HP_PATIENCE=3 jobs/batchsize_sweep_top_configs.sh
#

#SBATCH --job-name=batchsize_sweep
#SBATCH --partition=paula
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00:00

#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/batchsize_sweep_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/batchsize_sweep_%j.err

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

if [[ -z "${RESULTS_CSV:-}" ]]; then
  echo "FEHLER: RESULTS_CSV nicht gesetzt." >&2
  echo "Beispiel:" >&2
  echo "  sbatch --export=RESULTS_CSV=/work2/ra39oxet-DatasetAudioSetSubset/training_runs/tuning_genre_YYYY-MM-DD_HH-MM-SS/results.csv jobs/batchsize_sweep_top_configs.sh" >&2
  exit 1
fi
if [[ ! -f "$RESULTS_CSV" ]]; then
  echo "FEHLER: RESULTS_CSV nicht gefunden: $RESULTS_CSV" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export HP_MAX_EPOCHS="${HP_MAX_EPOCHS:-20}"
export HP_PATIENCE="${HP_PATIENCE:-3}"
export BATCH_SIZES="${BATCH_SIZES:-32 64 128 256}"

echo "Hostname: $(hostname)"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "RESULTS_CSV: $RESULTS_CSV"
echo "DATASET_RUN_NAME: ${DATASET_RUN_NAME:-<neuester Run>}"
echo "BATCH_SIZES: $BATCH_SIZES | HP_MAX_EPOCHS=$HP_MAX_EPOCHS | HP_PATIENCE=$HP_PATIENCE"
echo "Starte run_batchsize_sweep_top_configs.sh …"

bash "$REPO_ROOT/run_batchsize_sweep_top_configs.sh" "$RESULTS_CSV" "${DATASET_RUN_NAME:-}"
exit $?
