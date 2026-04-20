#!/bin/bash
#
# Nimmt Top-3 Konfigurationen aus einer results.csv und ergänzt (falls nicht enthalten)
# die beste MLP-Konfiguration. Führt diese Konfigurationen dann mit mehreren Batchgrößen aus.
#
# Nutzung:
#   bash run_batchsize_sweep_top_configs.sh tuning_results/genre/results.csv
#   DATASET_RUN_NAME=2026-03-25_13-12-06_audioset bash run_batchsize_sweep_top_configs.sh tuning_results/pair/results.csv
#   BATCH_SIZES="32 64 128 256" HP_MAX_EPOCHS=20 HP_PATIENCE=3 bash run_batchsize_sweep_top_configs.sh tuning_results/genre/results.csv
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run_batchsize_sweep_top_configs.sh <results.csv> [dataset_run_name]" >&2
  exit 1
fi

RESULTS_CSV="$1"
if [[ ! -f "$RESULTS_CSV" ]]; then
  echo "FEHLER: CSV nicht gefunden: $RESULTS_CSV" >&2
  exit 1
fi

# optionales 2. Argument überschreibt ENV
if [[ $# -ge 2 && -n "${2:-}" ]]; then
  export DATASET_RUN_NAME="$2"
fi

if [[ -z "${DATASET_RUN_NAME:-}" ]]; then
  DATASET_RUN_NAME="$(python3 -c "import sys; sys.path.insert(0,'.'); import config; print(config.get_latest_run_name() or '')")"
  export DATASET_RUN_NAME
fi
if [[ -z "${DATASET_RUN_NAME:-}" ]]; then
  echo "FEHLER: DATASET_RUN_NAME nicht gesetzt und kein Run unter DATASETS_ROOT gefunden." >&2
  exit 1
fi

# Defaults
export HP_MAX_EPOCHS="${HP_MAX_EPOCHS:-20}"
export HP_PATIENCE="${HP_PATIENCE:-3}"
export BATCH_SIZES="${BATCH_SIZES:-32 64 128 256}"

SELECTED_TXT="$(mktemp)"
python3 - "$RESULTS_CSV" > "$SELECTED_TXT" <<'PY'
import csv
import sys

csv_path = sys.argv[1]
rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
if not rows:
    raise SystemExit("Leere CSV.")

# CSV ist bereits sortiert (best -> worst). Falls nicht, hier zur Sicherheit:
rows.sort(key=lambda r: float(r["score_recall_at_10_avg"]), reverse=True)

selected = rows[:3]
has_mlp = any(r.get("head_type") == "mlp" for r in selected)
if not has_mlp:
    best_mlp = next((r for r in rows if r.get("head_type") == "mlp"), None)
    if best_mlp is not None and all(best_mlp.get("trial") != r.get("trial") for r in selected):
        selected.append(best_mlp)

for r in selected:
    print(
        "|".join(
            [
                r["trial"],
                r["training_type"],
                r["lr"],
                r["out_dim"],
                r["temp"],
                r["head_type"],
                r.get("hidden_dim", "256"),
                r["score_recall_at_10_avg"],
            ]
        )
    )
PY

if [[ ! -s "$SELECTED_TXT" ]]; then
  echo "FEHLER: Keine Konfigurationen ausgewählt." >&2
  exit 1
fi

TRAINING_TYPE="$(head -n 1 "$SELECTED_TXT" | cut -d'|' -f2)"
if [[ "$TRAINING_TYPE" != "pair" && "$TRAINING_TYPE" != "genre" ]]; then
  echo "FEHLER: training_type in CSV ungültig: $TRAINING_TYPE" >&2
  exit 1
fi

if [[ "$TRAINING_TYPE" == "pair" ]]; then
  TRAIN_SCRIPT="pipeline/pair_based_training.py"
else
  TRAIN_SCRIPT="pipeline/genre_based_training.py"
fi
EVAL_SCRIPT="pipeline/eval_single_head.py"

SWEEP_ROOT="$(python3 - <<'PY'
import sys
from datetime import datetime
sys.path.insert(0, ".")
import config
name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
d = config.TRAINING_RUNS_ROOT / f"batchsize_sweep_{name}"
d.mkdir(parents=True, exist_ok=True)
print(d)
PY
)"

echo "========== Batchsize Sweep =========="
echo "CSV: $RESULTS_CSV"
echo "Training type: $TRAINING_TYPE"
echo "Dataset-Run: $DATASET_RUN_NAME"
echo "Batch sizes: $BATCH_SIZES"
echo "Output root: $SWEEP_ROOT"
echo ""
echo "Ausgewählte Konfigurationen:"
cat "$SELECTED_TXT"
echo ""

SUMMARY_CSV="$SWEEP_ROOT/summary.csv"
echo "cfg_id,trial,training_type,lr,out_dim,temp,head_type,hidden_dim,batch_size,selection_score_recall_at_10_avg,selection_protocol,run_dir" > "$SUMMARY_CSV"

cfg_id=0
while IFS='|' read -r trial_id ttype lr out_dim temp head_type hidden_dim base_score; do
  cfg_id=$((cfg_id+1))
  for bs in $BATCH_SIZES; do
    run_dir="$SWEEP_ROOT/cfg${cfg_id}_trial${trial_id}_bs${bs}"
    mkdir -p "$run_dir"
    echo "-> cfg=$cfg_id trial=$trial_id bs=$bs (lr=$lr out_dim=$out_dim temp=$temp head=$head_type)"

    export TRAINING_RUN_DIR="$run_dir"
    export HP_LR="$lr"
    export HP_OUT_DIM="$out_dim"
    export HP_TEMP="$temp"
    export HP_HEAD_TYPE="$head_type"
    export HP_HIDDEN_DIM="$hidden_dim"
    export HP_BATCH_SIZE="$bs"
    python3 "$TRAIN_SCRIPT"

    metrics_json="$run_dir/val_metrics.json"
    TRAINING_TYPE="$TRAINING_TYPE" \
    EVAL_SPLIT="val" \
    EVAL_MODEL_PATH="$run_dir" \
    EVAL_METRICS_JSON="$metrics_json" \
    python3 "$EVAL_SCRIPT"

    score="$(python3 - <<PY
import json
with open("$metrics_json", "r", encoding="utf-8") as f:
    m = json.load(f)
print(m["selection_score_recall_at_10_avg"])
PY
)"
    protocol="$(python3 - <<PY
import json
with open("$metrics_json", "r", encoding="utf-8") as f:
    m = json.load(f)
print(m["selection_protocol"])
PY
)"

    echo "${cfg_id},${trial_id},${ttype},${lr},${out_dim},${temp},${head_type},${hidden_dim},${bs},${score},${protocol},${run_dir}" >> "$SUMMARY_CSV"
  done
done < "$SELECTED_TXT"

echo ""
echo "Fertig. Übersicht:"
echo "  $SUMMARY_CSV"
echo "Top-Zeilen:"
python3 - <<PY
import csv
rows = list(csv.DictReader(open("$SUMMARY_CSV", encoding="utf-8")))
rows.sort(key=lambda r: float(r["selection_score_recall_at_10_avg"]), reverse=True)
for r in rows[:8]:
    print(r)
PY

rm -f "$SELECTED_TXT"
