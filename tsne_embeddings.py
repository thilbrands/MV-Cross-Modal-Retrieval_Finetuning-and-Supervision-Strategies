"""
t-SNE-Vergleich: E1, E2, E3a, E3b — Video- und Audio-Embeddings (Test-Split).
Layout und Stil wie tsne_baseline.py (2 Zeilen × 4 Modelle).
Speichert tsne_comparison.pdf im Dataset-Run-Ordner.

Run: python3 tsne_embeddings.py
     TRAINING_RUN_DIR=... python3 tsne_embeddings.py
     AE_PAIR_RUN_DIR=... AE_GENRE_RUN_DIR=... python3 tsne_embeddings.py
"""
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
import config
from models import (
    load_projection_heads_pair,
    load_projection_heads_genre,
    load_audio_encoder_heads_pair,
    load_audio_encoder_heads_genre,
)

run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
if not run_name:
    print("FEHLER: Kein Dataset-Run gefunden.", flush=True)
    sys.exit(1)

run_dir = config.DATASETS_ROOT / run_name
split_csv = run_dir / "train_val_test_split.csv"
embeddings_dir = run_dir / "embeddings"

if os.environ.get("TRAINING_RUN_DIR"):
    training_run = Path(os.environ["TRAINING_RUN_DIR"])
    pair_path = training_run
    genre_path = training_run
else:
    pair_path = config.get_latest_training_run_with("projection_heads_pair.pt")
    genre_path = config.get_latest_training_run_with("projection_heads_genre.pt")

ae_pair_path = Path(os.environ["AE_PAIR_RUN_DIR"]) if os.environ.get("AE_PAIR_RUN_DIR") else config.get_latest_training_run_with("audio_encoder_pair.pt")
ae_genre_path = Path(os.environ["AE_GENRE_RUN_DIR"]) if os.environ.get("AE_GENRE_RUN_DIR") else config.get_latest_training_run_with("audio_encoder_genre.pt")

if not pair_path or not genre_path or not ae_pair_path or not ae_genre_path:
    print("FEHLER: Training-Runs nicht gefunden. TRAINING_RUN_DIR / AE_*_RUN_DIR setzen.", flush=True)
    sys.exit(1)

ae_pair_emb_dir = ae_pair_path / "audio_encoder_pair_test_embeddings"
ae_genre_emb_dir = ae_genre_path / "audio_encoder_genre_test_embeddings"

samples = []
with open(split_csv, "r", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["split"].strip() != "test":
            continue
        video_id = row["video_id"].strip()
        label = row["label"].strip()
        v_path = embeddings_dir / "video" / f"{video_id}.npy"
        a_path = embeddings_dir / "audio" / f"{video_id}.npy"
        ae_pair_path_emb = ae_pair_emb_dir / f"{video_id}.npy"
        ae_genre_path_emb = ae_genre_emb_dir / f"{video_id}.npy"
        if v_path.exists() and a_path.exists() and ae_pair_path_emb.exists() and ae_genre_path_emb.exists():
            samples.append((video_id, label, v_path, a_path, ae_pair_path_emb, ae_genre_path_emb))

if not samples:
    print("FEHLER: Keine Test-Samples mit allen Embeddings gefunden.", flush=True)
    sys.exit(1)

print(f"Dataset-Run: {run_name}", flush=True)
print(f"Pair-Run:      {pair_path}", flush=True)
print(f"Genre-Run:     {genre_path}", flush=True)
print(f"AE-Pair-Run:   {ae_pair_path}", flush=True)
print(f"AE-Genre-Run:  {ae_genre_path}", flush=True)
print(f"Split: test | Samples: {len(samples)}", flush=True)

labels = [s[1] for s in samples]
unique_labels = sorted(set(labels))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
label_to_color = {label: color for label, color in zip(unique_labels, colors)}

V = torch.tensor(np.stack([np.load(s[2]) for s in samples]), dtype=torch.float32, device=config.DEVICE)
A = torch.tensor(np.stack([np.load(s[3]) for s in samples]), dtype=torch.float32, device=config.DEVICE)
A_ae_pair = torch.tensor(np.stack([np.load(s[4]) for s in samples]), dtype=torch.float32, device=config.DEVICE)
A_ae_genre = torch.tensor(np.stack([np.load(s[5]) for s in samples]), dtype=torch.float32, device=config.DEVICE)

video_head_pair, audio_head_pair = load_projection_heads_pair(pair_path)
video_head_genre, audio_head_genre = load_projection_heads_genre(genre_path)
video_head_ae_pair, audio_head_ae_pair = load_audio_encoder_heads_pair(ae_pair_path)
video_head_ae_genre, audio_head_ae_genre = load_audio_encoder_heads_genre(ae_genre_path)


def _project(video_head, audio_head, video, audio):
    with torch.no_grad():
        v = F.normalize(video_head(video), p=2, dim=-1).cpu().numpy()
        a = F.normalize(audio_head(audio), p=2, dim=-1).cpu().numpy()
    return v, a


def _project_video(video_head, video):
    with torch.no_grad():
        return F.normalize(video_head(video), p=2, dim=-1).cpu().numpy()


def _project_audio(audio_head, audio):
    with torch.no_grad():
        return F.normalize(audio_head(audio), p=2, dim=-1).cpu().numpy()


v_e1, a_e1 = _project(video_head_pair, audio_head_pair, V, A)
v_e2, a_e2 = _project(video_head_genre, audio_head_genre, V, A)
v_e3a = _project_video(video_head_ae_pair, V)
a_e3a = _project_audio(audio_head_ae_pair, A_ae_pair)
v_e3b = _project_video(video_head_ae_genre, V)
a_e3b = _project_audio(audio_head_ae_genre, A_ae_genre)

model_embeddings = [
    ("E1", v_e1, a_e1),
    ("E2", v_e2, a_e2),
    ("E3a", v_e3a, a_e3a),
    ("E3b", v_e3b, a_e3b),
]


def compute_tsne(embs):
    return TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(embs)


def _plot_panel(ax, coords, title):
    for label in unique_labels:
        idx = [i for i, sample_label in enumerate(labels) if sample_label == label]
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            c=[label_to_color[label]],
            label=label,
            s=20,
            alpha=0.7,
        )
    ax.set_title(title, fontsize=22)
    ax.axis("off")


print("Berechne t-SNE …", flush=True)
fig, axes = plt.subplots(2, 4, figsize=(40, 16))
for col, (name, video_embs, audio_embs) in enumerate(model_embeddings):
    _plot_panel(axes[0, col], compute_tsne(video_embs), f"{name} (video)")
    _plot_panel(axes[1, col], compute_tsne(audio_embs), f"{name} (audio)")

for x in (0.25, 0.5, 0.75):
    fig.add_artist(plt.Line2D([x, x], [0.05, 0.95], transform=fig.transFigure, color="grey", linewidth=0.8))
fig.add_artist(plt.Line2D([0.05, 0.95], [0.5, 0.5], transform=fig.transFigure, color="grey", linewidth=0.8))

handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=label_to_color[label], markersize=16, label=label)
    for label in unique_labels
]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=18, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
out_path = run_dir / "tsne_comparison.pdf"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Gespeichert: {out_path}", flush=True)
