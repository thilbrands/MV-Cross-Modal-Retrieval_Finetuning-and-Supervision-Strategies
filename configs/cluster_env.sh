# Cluster paths — edit the two defaults below once, then forget them.
#
# All job/pipeline scripts source this file. Python reads WORK_ROOT from the
# environment (see configs/config.py).
#
# Layout created under WORK_ROOT:
#   AudioSetData/   datasets/   training_runs/   results/   logs/

# ========== EDIT ONCE ==========
WORK_ROOT="${WORK_ROOT:-/work2/ra39oxet-DatasetAudioSetSubset}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/venv/ba/bin/activate}"
# ===============================

export WORK_ROOT
export VENV_ACTIVATE

mkdir -p \
  "$WORK_ROOT/AudioSetData" \
  "$WORK_ROOT/datasets" \
  "$WORK_ROOT/training_runs" \
  "$WORK_ROOT/results" \
  "$WORK_ROOT/logs"
