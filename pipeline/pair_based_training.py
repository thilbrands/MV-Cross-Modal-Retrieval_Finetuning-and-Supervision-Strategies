"""
Pair-basiertes Training (Projektions-Heads + InfoNCE). Logik 1:1 aus old/07_training.ipynb.
Run: DATASET_RUN_NAME oder neuester Run. Heads werden unter PROJECTION_HEADS_PATH gespeichert.
"""
import os
import sys
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
run_dir = config.DATASETS_ROOT / run_name
EMBEDDINGS_DIR = run_dir / "embeddings"
TRAIN_VAL_TEST_SPLIT_CSV = run_dir / "train_val_test_split.csv"
PROJECTION_HEADS_PATH = config.PROJECTION_HEADS_PATH
DEVICE = config.DEVICE

train_ds = PairDataset("train", TRAIN_VAL_TEST_SPLIT_CSV, EMBEDDINGS_DIR)
val_ds = PairDataset("val", TRAIN_VAL_TEST_SPLIT_CSV, EMBEDDINGS_DIR)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

video_head = ProjectionHead().to(DEVICE)
audio_head = ProjectionHead().to(DEVICE)
opt = torch.optim.Adam(list(video_head.parameters()) + list(audio_head.parameters()), lr=1e-3)


def infonce_loss(v_proj, a_proj, temp=0.07):
    """Logits = v_proj @ a_proj.T / temp; Labels = Diagonale (positives Paar)."""
    logits = (v_proj @ a_proj.T) / temp
    labels = torch.arange(v_proj.size(0), device=v_proj.device)
    return nn.functional.cross_entropy(logits, labels)


num_epochs = 20
best_val = float("inf")

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
            PROJECTION_HEADS_PATH,
        )
    print(f"Epoch {epoch+1}  train={train_loss:.4f}  val={val_loss:.4f}  best_val={best_val:.4f}")

print(f"Gespeichert: {PROJECTION_HEADS_PATH}")
