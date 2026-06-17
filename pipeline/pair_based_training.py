"""
Pair-basiertes Training (Projektions-Heads + InfoNCE). Logik 1:1 aus old/07_training.ipynb.
"""
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
import config
from dataset import PairDataset
from models import ProjectionHead

def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None and v != "" else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None and v != "" else default


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
dataset_dir = config.DATASETS_ROOT / run_name
EMBEDDINGS_DIR = dataset_dir / "embeddings"
TRAIN_VAL_TEST_SPLIT_CSV = dataset_dir / "train_val_test_split.csv"
DEVICE = config.DEVICE
# Output-Directory
if os.environ.get("TRAINING_RUN_DIR"):
    training_run_dir = Path(os.environ["TRAINING_RUN_DIR"])
    training_run_dir.mkdir(parents=True, exist_ok=True)
else:
    training_run_dir = config.get_new_training_run_dir()
CHECKPOINT_PATH = training_run_dir / "projection_heads_pair.pt"

batch_size = _env_int("HP_BATCH_SIZE", 1024)
lr = _env_float("HP_LR", 1e-4)
temp = _env_float("HP_TEMP", 0.1)
out_dim = _env_int("HP_OUT_DIM", 512)
head_type = os.environ.get("HP_HEAD_TYPE", "mlp")
hidden_dim = _env_int("HP_HIDDEN_DIM", 512)
num_epochs = _env_int("HP_MAX_EPOCHS", 20)
patience = _env_int("HP_PATIENCE", 3)
seed = _env_int("HP_SEED", 42)
_set_seed(seed)

_train_genres_env = os.environ.get("TRAIN_GENRES")
train_genres = set(_train_genres_env.split(",")) if _train_genres_env else None

train_ds = PairDataset("train", TRAIN_VAL_TEST_SPLIT_CSV, EMBEDDINGS_DIR, allow_labels=train_genres)
val_ds = PairDataset("val", TRAIN_VAL_TEST_SPLIT_CSV, EMBEDDINGS_DIR, allow_labels=train_genres)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

video_head = ProjectionHead(out_dim=out_dim, head_type=head_type, hidden_dim=hidden_dim).to(DEVICE)
audio_head = ProjectionHead(out_dim=out_dim, head_type=head_type, hidden_dim=hidden_dim).to(DEVICE)
opt = torch.optim.Adam(list(video_head.parameters()) + list(audio_head.parameters()), lr=lr)

epochs_without_improvement = 0
best_val = float("inf")
print(f"Dataset-Run: {run_name} | Train: {len(train_ds)} | Val: {len(val_ds)} | Epochs: {num_epochs} | Device: {DEVICE} | TRAIN_GENRES: {train_genres or 'alle'}", flush=True)
print(f"Training-Run: {training_run_dir}", flush=True)
print(f"Hyperparams: lr={lr} temp={temp} out_dim={out_dim} head_type={head_type} hidden_dim={hidden_dim} batch_size={batch_size} patience={patience} seed={seed}", flush=True)


def infonce_loss(v_proj, a_proj, temp):
    v_proj = F.normalize(v_proj, p=2, dim=-1)
    a_proj = F.normalize(a_proj, p=2, dim=-1)
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
        loss = (infonce_loss(vp, ap, temp=temp) + infonce_loss(ap, vp, temp)) / 2
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
            val_loss += ((infonce_loss(vp, ap, temp=temp) + infonce_loss(ap, vp, temp=temp)) / 2).item() * v.size(0)
    val_loss /= len(val_ds)

    if val_loss < best_val:
        best_val = val_loss
        epochs_without_improvement = 0
        torch.save(
            {"video_head": video_head.state_dict(), "audio_head": audio_head.state_dict()},
            CHECKPOINT_PATH,
        )
    else:
        epochs_without_improvement += 1
    print(f"Epoch {epoch+1}/{num_epochs}  train={train_loss:.4f}  val={val_loss:.4f}  best_val={best_val:.4f}", flush=True)
    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch+1} (patience={patience}).", flush=True)
        break

# Metadaten für Nachvollziehbarkeit (meta_pair.json wenn gemeinsamer Run, sonst meta.json)
meta = {
    "timestamp": datetime.now().isoformat(),
    "dataset_run": run_name,
    "git_commit": config.get_git_commit(),
    "training_type": "pair",
    "train_genres": sorted(train_genres) if train_genres else None,
    "hyperparams": {
        "max_epochs": num_epochs,
        "patience": patience,
        "lr": lr,
        "batch_size": batch_size,
        "temp": temp,
        "out_dim": out_dim,
        "head_type": head_type,
        "hidden_dim": hidden_dim,
        "seed": seed,
    },
}
meta_file = "meta_pair.json" if os.environ.get("TRAINING_RUN_DIR") else "meta.json"
with open(training_run_dir / meta_file, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"Gespeichert: {CHECKPOINT_PATH}", flush=True)
