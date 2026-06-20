"""
Genre Similarity Heatmap: paarweise Cosinus-Ähnlichkeit zwischen Genre-Centroids.
Audio: frozen Wav2CLIP | Video: frozen CLIP (ViT-B/32). Val-Split.
Speichert genre_similarity.png im Dataset-Run-Ordner.

Run: python3 genre_similarity.py
     DATASET_RUN_NAME=... python3 genre_similarity.py
"""
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
import config

run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
if not run_name:
    print("FEHLER: Kein Dataset-Run gefunden.", flush=True)
    sys.exit(1)

run_dir = config.DATASETS_ROOT / run_name
split_csv = run_dir / "train_val_test_split.csv"
embeddings_dir = run_dir / "embeddings"


def _load_genre_embeddings(modality: str) -> dict[str, list[np.ndarray]]:
    genre_embs: dict[str, list[np.ndarray]] = defaultdict(list)
    subdir = "audio" if modality == "audio" else "video"
    with open(split_csv, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"].strip() != "val":
                continue
            video_id = row["video_id"].strip()
            label = row["label"].strip()
            path = embeddings_dir / subdir / f"{video_id}.npy"
            if path.exists():
                genre_embs[label].append(np.load(path))
    return genre_embs


def _similarity_matrix(genre_embs: dict[str, list[np.ndarray]], genres: list[str]) -> np.ndarray:
    centroids = np.stack([np.mean(genre_embs[g], axis=0) for g in genres])
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    return centroids @ centroids.T


genre_embs_audio = _load_genre_embeddings("audio")
genre_embs_video = _load_genre_embeddings("video")

genres = sorted(set(genre_embs_audio) & set(genre_embs_video))
if not genres:
    print("FEHLER: Keine gemeinsamen Genre-Embeddings auf Val-Split.", flush=True)
    sys.exit(1)

print(f"Genres ({len(genres)}): {genres}", flush=True)

sim_audio = _similarity_matrix(genre_embs_audio, genres)
sim_video = _similarity_matrix(genre_embs_video, genres)

fig, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)
for ax, sim, title in zip(
    axes,
    [sim_audio, sim_video],
    ["Audio — Wav2CLIP (frozen)", "Video — CLIP (frozen)"],
):
    sns.heatmap(
        sim,
        xticklabels=genres,
        yticklabels=genres,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        cbar=False,
        ax=ax,
    )
    ax.set_title(f"Genre Cosine Similarity — {title}\n(Val-Split)")
    ax.tick_params(axis="x", rotation=45)

fig.colorbar(axes[1].collections[0], ax=axes, location="right", pad=0.02, shrink=0.85)
out_path = run_dir / "genre_similarity.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Gespeichert: {out_path}", flush=True)
