"""
Genre Similarity Heatmap: paarweise Cosinus-Ähnlichkeit zwischen Genre-Centroids.
Basiert auf frozen Wav2CLIP Audio-Embeddings (Val-Split).
Speichert genre_similarity.png im Dataset-Run-Ordner.

Run: python3 genre_similarity.py
     DATASET_RUN_NAME=... python3 genre_similarity.py
"""
import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
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

# Val-Embeddings nach Genre gruppieren
genre_embs = defaultdict(list)
with open(split_csv, "r", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["split"].strip() != "val":
            continue
        video_id = row["video_id"].strip()
        label = row["label"].strip()
        a_path = embeddings_dir / "audio" / f"{video_id}.npy"
        if a_path.exists():
            genre_embs[label].append(np.load(a_path))

genres = sorted(genre_embs.keys())
print(f"Genres: {genres}", flush=True)

# Centroids berechnen und L2-normalisieren
centroids = np.stack([
    np.mean(genre_embs[g], axis=0) for g in genres
])
centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

# Paarweise Cosinus-Ähnlichkeit
sim_matrix = centroids @ centroids.T

# Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    sim_matrix,
    xticklabels=genres,
    yticklabels=genres,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=0,
    vmax=1,
    ax=ax,
)
ax.set_title("Genre Cosine Similarity (Wav2CLIP frozen, Val-Split)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

out_path = run_dir / "genre_similarity.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Gespeichert: {out_path}", flush=True)
