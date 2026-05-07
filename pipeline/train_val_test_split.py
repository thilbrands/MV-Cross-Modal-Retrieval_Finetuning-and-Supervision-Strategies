"""
Train/Val/Test-Split (stratifiziert) auf VLM-gefilterter, neu-balancierter CSV.
Input:  segments_unbalanced_vlm_scored.csv (REMOVE-Zeilen werden ignoriert)
Output: train_val_test_split.csv  (Spalten: video_id, label, vlm_score, split)
"""
import os
import random
import sys
import csv
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
import config
from sklearn.model_selection import train_test_split

random.seed(42)

run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
if not run_name:
    print("FEHLER: DATASET_RUN_NAME nicht gesetzt und kein Run unter DATASETS_ROOT.", flush=True)
    sys.exit(1)
run_dir = config.DATASETS_ROOT / run_name
SCORED_CSV = run_dir / "segments_unbalanced_vlm_scored.csv"
TRAIN_VAL_TEST_SPLIT_CSV = run_dir / "train_val_test_split.csv"

with open(SCORED_CSV, "r", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# REMOVE rausfiltern
samples = []
for row in rows:
    if row.get("vlm_score", "").strip() == "REMOVE":
        continue
    yt_id = row["yt_id"].strip()
    start_s = row["start_seconds"].strip().replace(".", "p")
    video_id = f"{yt_id}_{start_s}"
    samples.append({
        "video_id": video_id,
        "label": row.get("label", ""),
        "vlm_score": row["vlm_score"].strip(),
    })

# Re-balancing: gleich viele Videos pro Genre (Minimum über alle Labels)
by_label = defaultdict(list)
for s in samples:
    by_label[s["label"]].append(s)

print("Videos nach VLM-Filter pro Label (vor Balancing):", flush=True)
for label, items in sorted(by_label.items()):
    print(f"  {label}: {len(items)}", flush=True)

min_count = min(len(v) for v in by_label.values())
print(f"Balancing auf {min_count} pro Label.", flush=True)

balanced = []
for items in by_label.values():
    balanced.extend(random.sample(items, min_count))

labels = [s["label"] for s in balanced]

# Stratifiziert 70/15/15
train_val, test_samples = train_test_split(
    balanced, test_size=0.15, stratify=labels, random_state=42
)
train_val_labels = [s["label"] for s in train_val]
train_samples, val_samples = train_test_split(
    train_val, test_size=(15 / 85), stratify=train_val_labels, random_state=42
)

for s in train_samples:
    s["split"] = "train"
for s in val_samples:
    s["split"] = "val"
for s in test_samples:
    s["split"] = "test"

with open(TRAIN_VAL_TEST_SPLIT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["video_id", "label", "vlm_score", "split"])
    w.writeheader()
    w.writerows(train_samples + val_samples + test_samples)

n_train, n_val, n_test = len(train_samples), len(val_samples), len(test_samples)
print(f"Gesamt: {n_train + n_val + n_test} | Train: {n_train} | Val: {n_val} | Test: {n_test}", flush=True)
print(f"Gespeichert: {TRAIN_VAL_TEST_SPLIT_CSV}", flush=True)
