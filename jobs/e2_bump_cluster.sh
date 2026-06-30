#!/bin/bash
#
# Slurm-Job: E2-Bump-Analyse (Genre-Head auf frozen Embeddings).
# Links: Genre-Paar-Heatmap (mittlere cross-modale Similarity).
# Rechts: Different-genre Similarities nach Cluster (Within vs. Between).
# Speichert e2_bump_cluster.pdf/.png im Dataset-Run-Ordner.
#
# Nutzung (vom Repo-Root auf dem Cluster):
#   sbatch jobs/e2_bump_cluster.sh
#   sbatch --export=DATASET_RUN_NAME=2026-05-26_15-59-27_audioset,TRAINING_RUN_DIR=/work2/ra39oxet-DatasetAudioSetSubset/training_runs/2026-06-24_17-10 jobs/e2_bump_cluster.sh
#

#SBATCH --job-name=e2_bump_cluster
#SBATCH --partition=paula
#SBATCH --cpus-per-task=8
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-01:00:00

#SBATCH --output=/work2/ra39oxet-DatasetAudioSetSubset/logs/e2_bump_cluster_%j.out
#SBATCH --error=/work2/ra39oxet-DatasetAudioSetSubset/logs/e2_bump_cluster_%j.err

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
echo "GENRE_RUN_DIR: ${GENRE_RUN_DIR:-${TRAINING_RUN_DIR:-<neuester Run mit projection_heads_genre.pt>}}"
echo "BUMP_THRESHOLD: ${BUMP_THRESHOLD:-<-0.5>}"
echo "Starte plot_e2_bump_cluster.py …"

export PYTHONUNBUFFERED=1
python3 "$REPO_ROOT/plot_e2_bump_cluster.py"
exit $?
