"""
Gemeinsames t-SNE pro Modell: Video- UND Audio-Embeddings in EINEM t-SNE-Raum.
Farbe = Genre, Marker = Modalität (Kreis = Video, Dreieck = Audio).
So ist „liegen Video und Audio am selben Ort?" tatsächlich interpretierbar.

Eine Zeile, vier Modelle (E1, E2, E3a, E3b). Test-Split.
Speichert tsne_joint.pdf im Dataset-Run-Ordner.

Run: python3 tsne_joint.py
     TRAINING_RUN_DIR=... AE_PAIR_RUN_DIR=... AE_GENRE_RUN_DIR=... python3 tsne_joint.py
"""
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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

for label, p in [
    ("Dataset embeddings/video", embeddings_dir / "video"),
    ("Dataset embeddings/audio", embeddings_dir / "audio"),
    ("AE-Pair Test-Embeddings", ae_pair_emb_dir),
    ("AE-Genre Test-Embeddings", ae_genre_emb_dir),
    ("Split-CSV", split_csv),
]:
    if not p.exists():
        print(f"FEHLER: {label} nicht gefunden: {p}", flush=True)
        sys.exit(1)

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


def _project_video(video_head, video):
    with torch.no_grad():
        return F.normalize(video_head(video), p=2, dim=-1).cpu().numpy()


def _project_audio(audio_head, audio):
    with torch.no_grad():
        return F.normalize(audio_head(audio), p=2, dim=-1).cpu().numpy()


model_embeddings = [
    ("E1", _project_video(video_head_pair, V), _project_audio(audio_head_pair, A)),
    ("E2", _project_video(video_head_genre, V), _project_audio(audio_head_genre, A)),
    ("E3a", _project_video(video_head_ae_pair, V), _project_audio(audio_head_ae_pair, A_ae_pair)),
    ("E3b", _project_video(video_head_ae_genre, V), _project_audio(audio_head_ae_genre, A_ae_genre)),
]


def compute_tsne(embs):
    return TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(embs)


print("Berechne gemeinsame t-SNE …", flush=True)
n = len(samples)
fig, axes = plt.subplots(1, 4, figsize=(40, 11))
for ax, (name, video_embs, audio_embs) in zip(axes, model_embeddings):
    joint = np.concatenate([video_embs, audio_embs], axis=0)
    coords = compute_tsne(joint)
    v_coords, a_coords = coords[:n], coords[n:]
    for label in unique_labels:
        idx = [i for i, sample_label in enumerate(labels) if sample_label == label]
        ax.scatter(v_coords[idx, 0], v_coords[idx, 1], c=[label_to_color[label]], marker="o", s=22, alpha=0.7, linewidths=0)
        ax.scatter(a_coords[idx, 0], a_coords[idx, 1], c=[label_to_color[label]], marker="^", s=22, alpha=0.7, linewidths=0)
    ax.set_title(name, fontsize=22)
    ax.axis("off")

for x in (0.25, 0.5, 0.75):
    fig.add_artist(plt.Line2D([x, x], [0.12, 0.92], transform=fig.transFigure, color="grey", linewidth=0.8))

genre_handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=label_to_color[label], markersize=16, label=label)
    for label in unique_labels
]
modality_handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#444444", markersize=16, label="Video"),
    plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#444444", markersize=16, label="Audio"),
]

legend_genre = fig.legend(handles=genre_handles, loc="lower center", ncol=5, fontsize=16, bbox_to_anchor=(0.5, -0.02), title="Genre")
fig.add_artist(legend_genre)
fig.legend(handles=modality_handles, loc="lower center", ncol=2, fontsize=16, bbox_to_anchor=(0.5, 0.08), title="Modalität")

plt.tight_layout(rect=[0, 0.12, 1, 1])
out_path = run_dir / "tsne_joint.pdf"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Gespeichert: {out_path}", flush=True)
