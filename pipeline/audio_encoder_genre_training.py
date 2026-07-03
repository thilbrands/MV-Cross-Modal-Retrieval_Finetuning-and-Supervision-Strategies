"""
Genre-basiertes Training mit trainierbarem Audio-Encoder (Wav2CLIP unfrozen).
Hyperparameter werden per Env-Variablen übergeben (aus dem Tuning).

Unterschiede zu genre_based_training.py:
- Lädt rohe Audiowaveforms statt pre-computed Audio-Embeddings
- Wav2CLIP wird im Forward-Pass ausgeführt und mittrainiert
- Differenzierte Learning Rates: HP_LR für Heads, HP_LR_ENCODER für Wav2CLIP
- Checkpoint enthält zusätzlich Wav2CLIP weights
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
from dataset import RawAudioPairDataset
from models import ProjectionHead, load_wav2clip_finetune
from training_metrics import save_training_metrics_csv


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

if os.environ.get("TRAINING_RUN_DIR"):
    training_run_dir = Path(os.environ["TRAINING_RUN_DIR"])
    training_run_dir.mkdir(parents=True, exist_ok=True)
else:
    training_run_dir = config.get_new_training_run_dir()
CHECKPOINT_PATH = training_run_dir / "audio_encoder_genre.pt"
METRICS_CSV = training_run_dir / "results_audio_encoder_genre.csv"

batch_size = _env_int("HP_BATCH_SIZE", 64)
lr = _env_float("HP_LR", 1e-3)
lr_encoder = _env_float("HP_LR_ENCODER", lr / 10)
temp = _env_float("HP_TEMP", 0.05)
out_dim = _env_int("HP_OUT_DIM", 512)
head_type = os.environ.get("HP_HEAD_TYPE", "mlp")
hidden_dim = _env_int("HP_HIDDEN_DIM", 128)
num_epochs = _env_int("HP_MAX_EPOCHS", 20)
patience = _env_int("HP_PATIENCE", 3)
seed = _env_int("HP_SEED", 42)
_set_seed(seed)

_train_genres_env = os.environ.get("TRAIN_GENRES")
train_genres = set(_train_genres_env.split(",")) if _train_genres_env else None

train_ds = RawAudioPairDataset("train", TRAIN_VAL_TEST_SPLIT_CSV, EMBEDDINGS_DIR, return_label=True, allow_labels=train_genres)
val_ds = RawAudioPairDataset("val", TRAIN_VAL_TEST_SPLIT_CSV, EMBEDDINGS_DIR, return_label=True, allow_labels=train_genres)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

wav2clip_model = load_wav2clip_finetune(DEVICE)
video_head = ProjectionHead(out_dim=out_dim, head_type=head_type, hidden_dim=hidden_dim).to(DEVICE)
audio_head = ProjectionHead(out_dim=out_dim, head_type=head_type, hidden_dim=hidden_dim).to(DEVICE)

opt = torch.optim.Adam([
    {"params": list(video_head.parameters()) + list(audio_head.parameters()), "lr": lr},
    {"params": wav2clip_model.parameters(), "lr": lr_encoder},
])

print(f"Dataset-Run: {run_name} | Train: {len(train_ds)} | Val: {len(val_ds)} | Epochs: {num_epochs} | Device: {DEVICE} | TRAIN_GENRES: {train_genres or 'alle'}", flush=True)
print(f"Training-Run: {training_run_dir}", flush=True)
print(f"Hyperparams: lr={lr} lr_encoder={lr_encoder} temp={temp} out_dim={out_dim} head_type={head_type} hidden_dim={hidden_dim} batch_size={batch_size} patience={patience} seed={seed}", flush=True)


def genre_supcon_loss(v_proj, a_proj, labels, temp: float = 0.05):
    """
    Supervised Contrastive Loss (Khosla et al. 2020, Eq. 2), cross-modal V↔A.
    Pro Anker: Mittel über -log(exp(sim_pos)/sum_j exp(sim_j)) je Positive.
    """
    v_proj = F.normalize(v_proj, p=2, dim=-1)
    a_proj = F.normalize(a_proj, p=2, dim=-1)
    sim_va = v_proj @ a_proj.T
    sim_av = sim_va.T
    bsz = v_proj.size(0)
    loss = 0.0
    count = 0

    for i in range(bsz):
        same = [j for j, lab in enumerate(labels) if lab == labels[i]]
        if not same:
            continue
        all_scores = torch.exp(sim_va[i] / temp).sum()
        per_pos = torch.stack(
            [-torch.log(torch.exp(sim_va[i, p] / temp) / all_scores) for p in same]
        )
        loss += per_pos.mean()
        count += 1

    for i in range(bsz):
        same = [j for j, lab in enumerate(labels) if lab == labels[i]]
        if not same:
            continue
        all_scores = torch.exp(sim_av[i] / temp).sum()
        per_pos = torch.stack(
            [-torch.log(torch.exp(sim_av[i, p] / temp) / all_scores) for p in same]
        )
        loss += per_pos.mean()
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=v_proj.device)
    return loss / count


epochs_without_improvement = 0
best_val = float("inf")
metrics_history = []

for epoch in range(num_epochs):
    video_head.train()
    audio_head.train()
    wav2clip_model.train()
    train_loss = 0.0
    for v, a, labels in train_loader:
        v, a = v.to(DEVICE), a.to(DEVICE)
        opt.zero_grad()
        a_emb = wav2clip_model(a)
        vp, ap = video_head(v), audio_head(a_emb)
        loss = genre_supcon_loss(vp, ap, labels, temp=temp)
        loss.backward()
        opt.step()
        train_loss += loss.item() * v.size(0)
    train_loss /= len(train_ds)

    video_head.eval()
    audio_head.eval()
    wav2clip_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for v, a, labels in val_loader:
            v, a = v.to(DEVICE), a.to(DEVICE)
            a_emb = wav2clip_model(a)
            vp, ap = video_head(v), audio_head(a_emb)
            val_loss += genre_supcon_loss(vp, ap, labels, temp=temp).item() * v.size(0)
    val_loss /= len(val_ds)

    is_best = val_loss < best_val
    if is_best:
        best_val = val_loss
        epochs_without_improvement = 0
        torch.save(
            {
                "video_head": video_head.state_dict(),
                "audio_head": audio_head.state_dict(),
                "wav2clip": wav2clip_model.state_dict(),
                "encoder_finetuned": True,
            },
            CHECKPOINT_PATH,
        )
    else:
        epochs_without_improvement += 1
    metrics_history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val": best_val,
        "is_best": int(is_best),
        "epochs_without_improvement": epochs_without_improvement,
    })
    print(f"Epoch {epoch+1}/{num_epochs}  train={train_loss:.4f}  val={val_loss:.4f}  best_val={best_val:.4f}", flush=True)
    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch+1} (patience={patience}).", flush=True)
        break

meta = {
    "timestamp": datetime.now().isoformat(),
    "dataset_run": run_name,
    "git_commit": config.get_git_commit(),
    "training_type": "audio_encoder_genre",
    "train_genres": sorted(train_genres) if train_genres else None,
    "hyperparams": {
        "max_epochs": num_epochs,
        "patience": patience,
        "lr": lr,
        "lr_encoder": lr_encoder,
        "batch_size": batch_size,
        "temp": temp,
        "out_dim": out_dim,
        "head_type": head_type,
        "hidden_dim": hidden_dim,
        "seed": seed,
    },
}
meta_file = "meta_audio_encoder_genre.json" if os.environ.get("TRAINING_RUN_DIR") else "meta.json"
meta["metrics_csv"] = str(METRICS_CSV)
with open(training_run_dir / meta_file, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

save_training_metrics_csv(metrics_history, METRICS_CSV)
print(f"Gespeichert: {CHECKPOINT_PATH}", flush=True)
print(f"Metriken: {METRICS_CSV}", flush=True)

print("Berechne Test-Embeddings …", flush=True)
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
wav2clip_model.load_state_dict(ckpt["wav2clip"])
wav2clip_model.eval()
test_ds = RawAudioPairDataset("test", TRAIN_VAL_TEST_SPLIT_CSV, EMBEDDINGS_DIR, return_video_id=True)
test_embed_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
ae_emb_dir = training_run_dir / "audio_encoder_genre_test_embeddings"
ae_emb_dir.mkdir(exist_ok=True)
with torch.no_grad():
    for video_ids, _, a_t in test_embed_loader:
        embs = wav2clip_model(a_t.to(DEVICE)).cpu().numpy()
        for video_id, emb in zip(video_ids, embs):
            np.save(ae_emb_dir / f"{video_id}.npy", emb)
print(f"Test-Embeddings: {ae_emb_dir} ({len(test_ds.samples)} Dateien)", flush=True)
