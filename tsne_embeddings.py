"""
t-SNE Visualisierung der pre-computed Embeddings (Val-Split).
Speichert tsne_audio.png und tsne_video.png im Run-Ordner.

Run: python3 pipeline/tsne_embeddings.py
     DATASET_RUN_NAME=... python3 pipeline/tsne_embeddings.py
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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

# Val-Samples laden
samples = []
with open(split_csv, "r", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["split"].strip() != "val":
            continue
        video_id = row["video_id"].strip()
        label = row["label"].strip()
        a_path = embeddings_dir / "audio" / f"{video_id}.npy"
        v_path = embeddings_dir / "video" / f"{video_id}.npy"
        if a_path.exists() and v_path.exists():
            samples.append((video_id, label, v_path, a_path))

print(f"Val-Samples: {len(samples)}", flush=True)

labels = [s[1] for s in samples]
unique_labels = sorted(set(labels))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
label_to_color = {l: c for l, c in zip(unique_labels, colors)}

audio_embs = np.stack([np.load(s[3]) for s in samples])
video_embs = np.stack([np.load(s[2]) for s in samples])


def plot_tsne(embs, labels, title, out_path):
    print(f"Berechne t-SNE für {title} …", flush=True)
    coords = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(embs)
    fig, ax = plt.subplots(figsize=(10, 8))
    for label in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == label]
        ax.scatter(coords[idx, 0], coords[idx, 1], c=[label_to_color[label]], label=label, s=20, alpha=0.7)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gespeichert: {out_path}", flush=True)


plot_tsne(audio_embs, labels, "t-SNE Audio Embeddings (Val)", run_dir / "tsne_audio.png")
plot_tsne(video_embs, labels, "t-SNE Video Embeddings (Val)", run_dir / "tsne_video.png")
