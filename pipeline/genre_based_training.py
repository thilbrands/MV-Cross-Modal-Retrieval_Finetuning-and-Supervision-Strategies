"""
Genre-basiertes Training (Projektions-Heads + Supervised Contrastive über Labels).
Run: DATASET_RUN_NAME oder neuester Run. Speichert in training_runs/<Datum_Uhrzeit>/.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
import config
from dataset import PairDataset
from models import ProjectionHead

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
CHECKPOINT_PATH = training_run_dir / "projection_heads_genre.pt"

train_ds = PairDataset("train", TRAIN_VAL_TEST_SPLIT_CSV, EMBEDDINGS_DIR, return_label=True)
val_ds = PairDataset("val", TRAIN_VAL_TEST_SPLIT_CSV, EMBEDDINGS_DIR, return_label=True)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

video_head = ProjectionHead().to(DEVICE)
audio_head = ProjectionHead().to(DEVICE)
opt = torch.optim.Adam(list(video_head.parameters()) + list(audio_head.parameters()), lr=1e-3)

num_epochs = 5
best_val = float("inf")
print(f"Dataset-Run: {run_name} | Train: {len(train_ds)} | Val: {len(val_ds)} | Epochs: {num_epochs} | Device: {DEVICE}", flush=True)
print(f"Training-Run: {training_run_dir}", flush=True)

def genre_supcon_loss(v_proj, a_proj, labels, temp: float = 0.07):
    """
    Supervised Contrastive Loss über Genres, symmetrisch für V→A und A→V.
    Positive: alle Samples im Batch mit gleichem Label, Negative: Rest im Batch.
    """
    sim_va = v_proj @ a_proj.T  # [B, B]
    sim_av = sim_va.T
    bsz = v_proj.size(0)
    loss = 0.0
    count = 0

    for i in range(bsz):
        same = [j for j, lab in enumerate(labels) if lab == labels[i]]
        if not same:
            continue
        pos = torch.exp(sim_va[i, same] / temp).sum()
        all_scores = torch.exp(sim_va[i] / temp).sum()
        loss = loss - torch.log(pos / all_scores)
        count += 1

    for i in range(bsz):
        same = [j for j, lab in enumerate(labels) if lab == labels[i]]
        if not same:
            continue
        pos = torch.exp(sim_av[i, same] / temp).sum()
        all_scores = torch.exp(sim_av[i] / temp).sum()
        loss = loss - torch.log(pos / all_scores)
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=v_proj.device)
    return loss / count


for epoch in range(num_epochs):
    video_head.train()
    audio_head.train()
    train_loss = 0.0
    for v, a, labels in train_loader:
        v, a = v.to(DEVICE), a.to(DEVICE)
        opt.zero_grad()
        vp, ap = video_head(v), audio_head(a)
        loss = genre_supcon_loss(vp, ap, labels)
        loss.backward()
        opt.step()
        train_loss += loss.item() * v.size(0)
    train_loss /= len(train_ds)

    video_head.eval()
    audio_head.eval()
    val_loss = 0.0
    with torch.no_grad():
        for v, a, labels in val_loader:
            v, a = v.to(DEVICE), a.to(DEVICE)
            vp, ap = video_head(v), audio_head(a)
            val_loss += genre_supcon_loss(vp, ap, labels).item() * v.size(0)
    val_loss /= len(val_ds)

    if val_loss < best_val:
        best_val = val_loss
        torch.save(
            {"video_head": video_head.state_dict(), "audio_head": audio_head.state_dict()},
            CHECKPOINT_PATH,
        )
    print(f"Epoch {epoch+1}/{num_epochs}  train={train_loss:.4f}  val={val_loss:.4f}  best_val={best_val:.4f}", flush=True)

# Metadaten für Nachvollziehbarkeit
meta = {
    "timestamp": datetime.now().isoformat(),
    "dataset_run": run_name,
    "git_commit": config.get_git_commit(),
    "git_dirty": config.get_git_dirty(),
    "training_type": "genre",
    "hyperparams": {"epochs": num_epochs, "lr": 1e-3, "batch_size": 32, "temp": 0.07},
}
with open(training_run_dir / "meta_genre.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"Gespeichert: {CHECKPOINT_PATH}", flush=True)

