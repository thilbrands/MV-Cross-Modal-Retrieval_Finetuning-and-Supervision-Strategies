"""
t-SNE Visualisierung: Audio frozen / E1 finetuned / E3 finetuned / Video (Test-Split).
Speichert tsne_comparison.png im Dataset-Run-Ordner.

Run: python3 tsne_embeddings.py
     AE_PAIR_RUN_DIR=... AE_GENRE_RUN_DIR=... python3 tsne_embeddings.py
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

ae_pair_run = Path(os.environ["AE_PAIR_RUN_DIR"]) if os.environ.get("AE_PAIR_RUN_DIR") else config.get_latest_training_run_with("audio_encoder_pair.pt")
ae_genre_run = Path(os.environ["AE_GENRE_RUN_DIR"]) if os.environ.get("AE_GENRE_RUN_DIR") else config.get_latest_training_run_with("audio_encoder_genre.pt")

if not ae_pair_run or not ae_genre_run:
    print("FEHLER: Audio-Encoder-Runs nicht gefunden. AE_PAIR_RUN_DIR / AE_GENRE_RUN_DIR setzen.", flush=True)
    sys.exit(1)

ae_pair_emb_dir = ae_pair_run / "audio_encoder_test_embeddings"
ae_genre_emb_dir = ae_genre_run / "audio_encoder_test_embeddings"

# Test-Samples laden
samples = []
with open(split_csv, "r", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["split"].strip() != "test":
            continue
        video_id = row["video_id"].strip()
        label = row["label"].strip()
        a_path = embeddings_dir / "audio" / f"{video_id}.npy"
        v_path = embeddings_dir / "video" / f"{video_id}.npy"
        ae_pair_path = ae_pair_emb_dir / f"{video_id}.npy"
        ae_genre_path = ae_genre_emb_dir / f"{video_id}.npy"
        if a_path.exists() and v_path.exists() and ae_pair_path.exists() and ae_genre_path.exists():
            samples.append((video_id, label, v_path, a_path, ae_pair_path, ae_genre_path))

print(f"Test-Samples: {len(samples)}", flush=True)

labels = [s[1] for s in samples]
unique_labels = sorted(set(labels))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
label_to_color = {l: c for l, c in zip(unique_labels, colors)}

audio_frozen   = np.stack([np.load(s[3]) for s in samples])
audio_e1       = np.stack([np.load(s[4]) for s in samples])
audio_e3       = np.stack([np.load(s[5]) for s in samples])
video_embs     = np.stack([np.load(s[2]) for s in samples])

def compute_tsne(embs):
    return TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(embs)

print("Berechne t-SNE …", flush=True)
coords = [compute_tsne(e) for e in [audio_frozen, audio_e1, audio_e3, video_embs]]
titles = ["Wav2CLIP (frozen)", "Wav2CLIP finetuned (InfoNCE)", "Wav2CLIP finetuned (SupCon)", "CLIP (frozen)"]

fig, axes = plt.subplots(1, 4, figsize=(30, 7))
for ax, c, title in zip(axes, coords, titles):
    for label in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == label]
        ax.scatter(c[idx, 0], c[idx, 1], c=[label_to_color[label]], label=label, s=15, alpha=0.7)
    ax.set_title(title, fontsize=12)
    ax.axis("off")

handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=label_to_color[l], markersize=8, label=l) for l in unique_labels]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
out_path = run_dir / "tsne_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Gespeichert: {out_path}", flush=True)
