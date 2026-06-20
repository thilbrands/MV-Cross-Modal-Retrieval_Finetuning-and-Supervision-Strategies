"""
Genre-basiertes Training (Projektions-Heads + Supervised Contrastive über Labels).
Run: DATASET_RUN_NAME oder neuester Run. Speichert in training_runs/<Datum_Uhrzeit>/.
"""
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
CHECKPOINT_PATH = training_run_dir / "projection_heads_genre.pt"

batch_size = _env_int("HP_BATCH_SIZE", 64)
lr = _env_float("HP_LR", 1e-3)
temp = _env_float("HP_TEMP", 1.5)
out_dim = _env_int("HP_OUT_DIM", 256)
head_type = os.environ.get("HP_HEAD_TYPE", "mlp")
hidden_dim = _env_int("HP_HIDDEN_DIM", 512)
num_epochs = _env_int("HP_MAX_EPOCHS", 20)
patience = _env_int("HP_PATIENCE", 3)
seed = _env_int("HP_SEED", 42)
_set_seed(seed)

_train_genres_env = os.environ.get("TRAIN_GENRES")
train_genres = set(_train_genres_env.split(",")) if _train_genres_env else None

train_ds = PairDataset("train", TRAIN_VAL_TEST_SPLIT_CSV, EMBEDDINGS_DIR, return_label=True, allow_labels=train_genres)
val_ds = PairDataset("val", TRAIN_VAL_TEST_SPLIT_CSV, EMBEDDINGS_DIR, return_label=True, allow_labels=train_genres)
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

def genre_supcon_loss(v_proj, a_proj, labels, temp: float = 0.07):
    """
     Vektorisierte Implementierung wegen compute limits
    """
    v_proj = F.normalize(v_proj, p=2, dim=-1)
    a_proj = F.normalize(a_proj, p=2, dim=-1)

    labels_arr = np.asarray(labels)
    mask = torch.from_numpy(labels_arr[:, None] == labels_arr[None, :]).to(v_proj.device).float()

    sim_va = (v_proj @ a_proj.T) / temp  # [B, B]
    sim_av = sim_va.T

    def _dir_loss(sim):
        # log_softmax über alle Kandidaten = log(exp(sim)/sum_j exp(sim))
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        return -mean_log_prob_pos.mean()

    return (_dir_loss(sim_va) + _dir_loss(sim_av)) / 2


# def genre_supcon_loss(v_proj, a_proj, labels, temp: float = 0.07):
#     v_proj = F.normalize(v_proj, p=2, dim=-1)
#     a_proj = F.normalize(a_proj, p=2, dim=-1)
#     sim_va = v_proj @ a_proj.T  # [B, B]
#     sim_av = sim_va.T
#     bsz = v_proj.size(0)
#     loss = 0.0
#     count = 0
#
#     for i in range(bsz):
#         same = [j for j, lab in enumerate(labels) if lab == labels[i]]
#         if not same:
#             continue
#         all_scores = torch.exp(sim_va[i] / temp).sum()
#         per_pos = torch.stack(
#             [-torch.log(torch.exp(sim_va[i, p] / temp) / all_scores) for p in same]
#         )
#         loss += per_pos.mean()
#         count += 1
#
#     for i in range(bsz):
#         same = [j for j, lab in enumerate(labels) if lab == labels[i]]
#         if not same:
#             continue
#         all_scores = torch.exp(sim_av[i] / temp).sum()
#         per_pos = torch.stack(
#             [-torch.log(torch.exp(sim_av[i, p] / temp) / all_scores) for p in same]
#         )
#         loss += per_pos.mean()
#         count += 1
#
#     if count == 0:
#         return torch.tensor(0.0, device=v_proj.device)
#     return loss / count


for epoch in range(num_epochs):
    video_head.train()
    audio_head.train()
    train_loss = 0.0
    for v, a, labels in train_loader:
        v, a = v.to(DEVICE), a.to(DEVICE)
        opt.zero_grad()
        vp, ap = video_head(v), audio_head(a)
        loss = genre_supcon_loss(vp, ap, labels, temp=temp)
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
            val_loss += genre_supcon_loss(vp, ap, labels, temp=temp).item() * v.size(0)
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

# Metadaten für Nachvollziehbarkeit
meta = {
    "timestamp": datetime.now().isoformat(),
    "dataset_run": run_name,
    "git_commit": config.get_git_commit(),
    "training_type": "genre",
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
with open(training_run_dir / "meta_genre.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"Gespeichert: {CHECKPOINT_PATH}", flush=True)

