"""
Train/Val/Test-Split (stratifiziert). Logik 1:1 aus old/05_train_val_test_split.ipynb.
Run: DATASET_RUN_NAME oder neuester Run (get_latest_run_name). Ausgabe: run_dir/train_val_test_split.csv
"""
import os
import sys
import csv
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
import config
from sklearn.model_selection import train_test_split

# Run: aus Umgebung oder neuester
run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
if not run_name:
    print("FEHLER: DATASET_RUN_NAME nicht gesetzt und kein Run unter DATASETS_ROOT.", flush=True)
    sys.exit(1)
run_dir = config.DATASETS_ROOT / run_name
SUBSET_CLEANED_CSV = run_dir / "segments_balanced.csv"
EMBEDDINGS_DIR = run_dir / "embeddings"
TRAIN_VAL_TEST_SPLIT_CSV = run_dir / "train_val_test_split.csv"

with open(SUBSET_CLEANED_CSV, "r", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

samples = []
for row in rows:
    yt_id = row["yt_id"].strip()
    start_s = row["start_seconds"].strip().replace(".", "p")
    video_id = f"{yt_id}_{start_s}"
    if (EMBEDDINGS_DIR / "video" / f"{video_id}.npy").exists() and (EMBEDDINGS_DIR / "audio" / f"{video_id}.npy").exists():
        samples.append({"video_id": video_id, "label": row.get("label", "")})

labels = [s["label"] for s in samples]

# Stratifiziert: 30 % Test, 70 % Train-Pool
train_val, test_samples = train_test_split(
    samples, test_size=0.3, stratify=labels, random_state=42
)
train_val_labels = [s["label"] for s in train_val]
# Vom Train-Pool: 70 % Train, 30 % Val (stratifiziert)
train_samples, val_samples = train_test_split(
    train_val, test_size=0.3, stratify=train_val_labels, random_state=42
)

for s in train_samples:
    s["split"] = "train"
for s in val_samples:
    s["split"] = "val"
for s in test_samples:
    s["split"] = "test"

with open(TRAIN_VAL_TEST_SPLIT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["video_id", "label", "split"])
    w.writeheader()
    w.writerows(train_samples)
    w.writerows(val_samples)
    w.writerows(test_samples)

n_train, n_val, n_test = len(train_samples), len(val_samples), len(test_samples)
print(f"Gesamt: {len(samples)} | Train: {n_train} | Val: {n_val} | Test: {n_test}")
print(f"Gespeichert: {TRAIN_VAL_TEST_SPLIT_CSV}")
