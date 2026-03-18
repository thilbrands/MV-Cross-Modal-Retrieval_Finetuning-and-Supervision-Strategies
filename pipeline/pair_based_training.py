"""
Pair-basiertes Training (Projektions-Heads + InfoNCE). Logik 1:1 aus old/07_training.ipynb.
Run: DATASET_RUN_NAME oder neuester Run. Speichert in training_runs/<Datum_Uhrzeit>/.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
import config
from dataset import PairDataset
from models import ProjectionHead

# Run: aus Umgebung oder neuester
run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
if not run_name:
    print("FEHLER: DATASET_RUN_NAME nicht gesetzt und kein Run unter DATASETS_ROOT.", flush=True)
    sys.exit(1)
dataset_dir = config.DATASETS_ROOT / run_name
EMBEDDINGS_DIR = dataset_dir / "embeddings"
TRAIN_VAL_TEST_SPLIT_CSV = dataset_dir / "train_val_test_split.csv"
DEVICE = config.DEVICE

# Gemeinsamer Ordner (von run_train_and_eval.sh) oder neuer Einzel-Run
if os.environ.get("TRAINING_RUN_DIR"):
    training_run_dir = Path(os.environ["TRAINING_RUN_DIR"])
    training_run_dir.mkdir(parents=True, exist_ok=True)
else:
    training_run_dir = config.get_new_training_run_dir()
CHECKPOINT_PATH = training_run_dir / "projection_heads_pair.pt"


def progress_stderr(epoch_done: int, total: int) -> None:
    """Nur Fortschritt in Prozent nach .err (für tail -f)."""
    pct = int((epoch_done / total) * 100)
    print(f"{pct:3d}%", file=sys.stderr, flush=True)


train_ds = PairDataset("train", TRAIN_VAL_TEST_SPLIT_CSV, EMBEDDINGS_DIR)
val_ds = PairDataset("val", TRAIN_VAL_TEST_SPLIT_CSV, EMBEDDINGS_DIR)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

video_head = ProjectionHead().to(DEVICE)
audio_head = ProjectionHead().to(DEVICE)
opt = torch.optim.Adam(list(video_head.parameters()) + list(audio_head.parameters()), lr=1e-3)

num_epochs = 5
best_val = float("inf")
print(f"Dataset-Run: {run_name} | Train: {len(train_ds)} | Val: {len(val_ds)} | Epochs: {num_epochs} | Device: {DEVICE}", flush=True)
print(f"Training-Run: {training_run_dir}", flush=True)
progress_stderr(0, num_epochs)


def infonce_loss(v_proj, a_proj, temp=0.07):
    """Logits = v_proj @ a_proj.T / temp; Labels = Diagonale (positives Paar)."""
    logits = (v_proj @ a_proj.T) / temp
    labels = torch.arange(v_proj.size(0), device=v_proj.device)
    return nn.functional.cross_entropy(logits, labels)

for epoch in range(num_epochs):
    video_head.train()
    audio_head.train()
    train_loss = 0.0
    for v, a in train_loader:
        v, a = v.to(DEVICE), a.to(DEVICE)
        opt.zero_grad()
        vp, ap = video_head(v), audio_head(a)
        loss = infonce_loss(vp, ap)
        loss.backward()
        opt.step()
        train_loss += loss.item() * v.size(0)
    train_loss /= len(train_ds)

    video_head.eval()
    audio_head.eval()
    val_loss = 0.0
    with torch.no_grad():
        for v, a in val_loader:
            v, a = v.to(DEVICE), a.to(DEVICE)
            vp, ap = video_head(v), audio_head(a)
            val_loss += infonce_loss(vp, ap).item() * v.size(0)
    val_loss /= len(val_ds)

    if val_loss < best_val:
        best_val = val_loss
        torch.save(
            {"video_head": video_head.state_dict(), "audio_head": audio_head.state_dict()},
            CHECKPOINT_PATH,
        )
    print(f"Epoch {epoch+1}/{num_epochs}  train={train_loss:.4f}  val={val_loss:.4f}  best_val={best_val:.4f}", flush=True)
    progress_stderr(epoch + 1, num_epochs)

# Metadaten für Nachvollziehbarkeit (meta_pair.json wenn gemeinsamer Run, sonst meta.json)
meta = {
    "timestamp": datetime.now().isoformat(),
    "dataset_run": run_name,
    "git_commit": config.get_git_commit(),
    "git_dirty": config.get_git_dirty(),
    "training_type": "pair",
    "hyperparams": {"epochs": num_epochs, "lr": 1e-3, "batch_size": 32, "temp": 0.07},
}
meta_file = "meta_pair.json" if os.environ.get("TRAINING_RUN_DIR") else "meta.json"
with open(training_run_dir / meta_file, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"Gespeichert: {CHECKPOINT_PATH}", flush=True)
