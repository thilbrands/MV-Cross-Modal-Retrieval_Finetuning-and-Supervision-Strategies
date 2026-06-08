"""
t-SNE Visualisierung der Baseline: frozen CLIP (Video) und frozen Wav2CLIP (Audio).
Beide Panels auf einem Bild, eingefärbt nach Genre. Nutzt den TEST-Split.
Speichert tsne_baseline.png im Dataset-Run-Ordner.

Run: python3 tsne_baseline.py
     DATASET_RUN_NAME=... python3 tsne_baseline.py
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

# Test-Samples laden (frozen CLIP-Video + frozen Wav2CLIP-Audio)
samples = []
with open(split_csv, "r", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["split"].strip() != "test":
            continue
        video_id = row["video_id"].strip()
        label = row["label"].strip()
        a_path = embeddings_dir / "audio" / f"{video_id}.npy"
        v_path = embeddings_dir / "video" / f"{video_id}.npy"
        if a_path.exists() and v_path.exists():
            samples.append((video_id, label, v_path, a_path))

print(f"Dataset-Run: {run_name}", flush=True)
print(f"Split: test | Samples: {len(samples)}", flush=True)

labels = [s[1] for s in samples]
unique_labels = sorted(set(labels))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
label_to_color = {l: c for l, c in zip(unique_labels, colors)}

video_embs = np.stack([np.load(s[2]) for s in samples])
audio_embs = np.stack([np.load(s[3]) for s in samples])


def compute_tsne(embs):
    return TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(embs)


print("Berechne t-SNE …", flush=True)
coords = [compute_tsne(video_embs), compute_tsne(audio_embs)]
titles = ["CLIP (frozen)", "Wav2CLIP (frozen)"]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, c, title in zip(axes, coords, titles):
    for label in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == label]
        ax.scatter(c[idx, 0], c[idx, 1], c=[label_to_color[label]], label=label, s=15, alpha=0.7)
    ax.set_title(title, fontsize=14)
    ax.axis("off")

handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=label_to_color[l], markersize=8, label=l) for l in unique_labels]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
out_path = run_dir / "tsne_baseline.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Gespeichert: {out_path}", flush=True)
