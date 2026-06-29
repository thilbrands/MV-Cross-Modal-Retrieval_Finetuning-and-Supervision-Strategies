"""
E3b-Similarity-Verteilung, aufgeteilt nach Genre-Cluster-Zugehoerigkeit.

Im E3b-Raum (Audio-Encoder Genre) wird die volle Video x Audio Cosine-Similarity auf
dem Test-Split berechnet und in drei Gruppen geteilt:

  Same genre        - Video- und Audio-Genre identisch (positive Paare, Protocol B)
  Within cluster    - different genre, ABER beide Genres aus CLUSTER_GENRES
                      {Electronic, Funk, Hip hop, Pop, Reggae}
  Between clusters   - alle uebrigen different-genre Paare

Drei ueberlagerte Density-Histogramme + Median-Linien.
Speichert e3b_cluster_split.pdf/.png im Dataset-Run-Ordner.

Env-Vars:
  DATASET_RUN_NAME   - Dataset-Run (default: neuester)
  TRAINING_RUN_DIR   - Run mit audio_encoder_genre.pt (auch als AE-Genre-Default)
  AE_GENRE_RUN_DIR   - ueberschreibt TRAINING_RUN_DIR fuer den Genre-Encoder

Run: python3 plot_e3b_cluster_split.py
     TRAINING_RUN_DIR=... python3 plot_e3b_cluster_split.py
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

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
import config
from models import load_audio_encoder_heads_genre

CLUSTER_GENRES = {"Electronic music", "Funk", "Hip hop music", "Pop music", "Reggae"}

COLOR_SAME = "#2ca02c"
COLOR_WITHIN = "#ff7f0e"
COLOR_BETWEEN = "#1f77b4"

run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
if not run_name:
    print("FEHLER: Kein Dataset-Run gefunden.", flush=True)
    sys.exit(1)

run_dir = config.DATASETS_ROOT / run_name
split_csv = run_dir / "train_val_test_split.csv"
embeddings_dir = run_dir / "embeddings"

ae_genre_path = (
    Path(os.environ["AE_GENRE_RUN_DIR"]) if os.environ.get("AE_GENRE_RUN_DIR")
    else Path(os.environ["TRAINING_RUN_DIR"]) if os.environ.get("TRAINING_RUN_DIR")
    else config.get_latest_training_run_with("audio_encoder_genre.pt")
)
if not ae_genre_path:
    print("FEHLER: AE-Genre-Run nicht gefunden. AE_GENRE_RUN_DIR / TRAINING_RUN_DIR setzen.", flush=True)
    sys.exit(1)

ae_genre_emb_dir = ae_genre_path / "audio_encoder_genre_test_embeddings"
for label, p in [
    ("Dataset embeddings/video", embeddings_dir / "video"),
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
        ae_genre_emb = ae_genre_emb_dir / f"{video_id}.npy"
        if v_path.exists() and ae_genre_emb.exists():
            samples.append((video_id, label, v_path, ae_genre_emb))

if not samples:
    print("FEHLER: Keine Test-Samples mit allen Embeddings gefunden.", flush=True)
    sys.exit(1)

print(f"Dataset-Run:  {run_name}", flush=True)
print(f"AE-Genre-Run: {ae_genre_path}", flush=True)
print(f"Test-Samples: {len(samples)}", flush=True)

labels = np.array([s[1] for s in samples])
in_cluster = np.array([l in CLUSTER_GENRES for l in labels])

V = torch.tensor(np.stack([np.load(s[2]) for s in samples]), dtype=torch.float32, device=config.DEVICE)
A_ae_genre = torch.tensor(np.stack([np.load(s[3]) for s in samples]), dtype=torch.float32, device=config.DEVICE)

video_head, audio_head = load_audio_encoder_heads_genre(ae_genre_path)

with torch.no_grad():
    v = F.normalize(video_head(V), p=2, dim=-1)
    a = F.normalize(audio_head(A_ae_genre), p=2, dim=-1)
    sim = (v @ a.T).cpu().numpy()

same_mask = labels[:, None] == labels[None, :]
diff_mask = ~same_mask
both_in_cluster = in_cluster[:, None] & in_cluster[None, :]

within_mask = diff_mask & both_in_cluster
between_mask = diff_mask & ~both_in_cluster

groups = [
    ("Same genre", sim[same_mask], COLOR_SAME),
    ("Within cluster", sim[within_mask], COLOR_WITHIN),
    ("Between clusters", sim[between_mask], COLOR_BETWEEN),
]

print(f"\nCluster (within) = {sorted(CLUSTER_GENRES)}", flush=True)
for name, vals, _ in groups:
    print(f"{name:<16} n={vals.size:>9}  median={np.median(vals):.3f}  mean={vals.mean():.3f}", flush=True)

all_vals = np.concatenate([vals for _, vals, _ in groups])
xmin, xmax = float(all_vals.min()), float(all_vals.max())
pad = 0.02 * (xmax - xmin)
xlim = (xmin - pad, xmax + pad)
bins = np.linspace(xlim[0], xlim[1], 80)

fig, ax = plt.subplots(figsize=(7, 3.5))
for name, vals, color in groups:
    ax.hist(vals, bins=bins, density=True, color=color, alpha=0.45, label=name)
    ax.axvline(np.median(vals), color=color, linestyle="--", linewidth=1.2)

ax.set_xlabel("Cosine similarity", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_xlim(xlim)
ax.tick_params(axis="both", labelsize=9)
ax.grid(which="major", axis="y", linestyle="-", linewidth=0.4, alpha=0.5)
for spine in ax.spines.values():
    spine.set_linewidth(0.6)
    spine.set_color("#666666")
ax.legend(fontsize=7.5, frameon=True, framealpha=0.9, edgecolor="#cccccc")

fig.tight_layout()
out_base = run_dir / "e3b_cluster_split"
fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nGespeichert: {out_base}.pdf", flush=True)
print(f"Gespeichert: {out_base}.png", flush=True)
