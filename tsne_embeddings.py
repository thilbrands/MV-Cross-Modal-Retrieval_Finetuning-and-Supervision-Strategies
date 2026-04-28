"""
t-SNE Visualisierung: Audio frozen vs. Audio finetuned (E3) vs. Video (Test-Split).
Speichert tsne_comparison.png im Dataset-Run-Ordner.
Run: python3 tsne_embeddings.py
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

ae_run_dir = Path(os.environ["AE_PAIR_RUN_DIR"]) if os.environ.get("AE_PAIR_RUN_DIR") else config.get_latest_training_run_with("audio_encoder_pair.pt")
if not ae_run_dir:
    print("FEHLER: Kein Audio-Encoder-Run gefunden. AE_PAIR_RUN_DIR setzen.", flush=True)
    sys.exit(1)

ae_emb_dir = ae_run_dir / "audio_encoder_test_embeddings"

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
        ae_path = ae_emb_dir / f"{video_id}.npy"
        if a_path.exists() and v_path.exists() and ae_path.exists():
            samples.append((video_id, label, v_path, a_path, ae_path))

print(f"Test-Samples: {len(samples)}", flush=True)

labels = [s[1] for s in samples]
unique_labels = sorted(set(labels))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
label_to_color = {l: c for l, c in zip(unique_labels, colors)}

audio_frozen = np.stack([np.load(s[3]) for s in samples])
audio_finetuned = np.stack([np.load(s[4]) for s in samples])
video_embs = np.stack([np.load(s[2]) for s in samples])


def compute_tsne(embs):
    return TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(embs)


print("Berechne t-SNE …", flush=True)
coords_audio_frozen = compute_tsne(audio_frozen)
coords_audio_finetuned = compute_tsne(audio_finetuned)
coords_video = compute_tsne(video_embs)

fig, axes = plt.subplots(1, 3, figsize=(24, 7))
titles = ["Audio frozen (Wav2CLIP)", "Audio finetuned (E3)", "Video (CLIP)"]
coords_list = [coords_audio_frozen, coords_audio_finetuned, coords_video]

for ax, coords, title in zip(axes, coords_list, titles):
    for label in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == label]
        ax.scatter(coords[idx, 0], coords[idx, 1], c=[label_to_color[label]], label=label, s=15, alpha=0.7)
    ax.set_title(title, fontsize=13)
    ax.axis("off")

handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=label_to_color[l], markersize=8, label=l) for l in unique_labels]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
out_path = run_dir / "tsne_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Gespeichert: {out_path}", flush=True)
