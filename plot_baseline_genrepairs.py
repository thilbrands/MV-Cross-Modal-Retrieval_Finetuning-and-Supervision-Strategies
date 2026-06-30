"""
Genre-Paar-Heatmap im FROZEN Embedding-Raum (Baseline, ohne Projektionskoepfe),
in beide Richtungen.

Berechnet die volle cross-modale Cosine-Similarity zwischen frozen Video- (CLIP)
und Audio-Embeddings (Wav2CLIP) auf dem Test-Split und mittelt blockweise je Genre-Paar.

Zwei Panels (gemeinsame Farbskala):
  (a) V->A: Zeile = Query-Genre (Video), Spalte = Audio-Genre
  (b) A->V: Zeile = Query-Genre (Audio), Spalte = Video-Genre
Diagonale (same genre) ausmaskiert.

Env-Vars:
  DATASET_RUN_NAME   - Dataset-Run (default: neuester)

Run: python3 plot_baseline_genrepairs.py
     DATASET_RUN_NAME=... python3 plot_baseline_genrepairs.py
"""
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
import config

GENRE_SHORT = {
    "Blues": "Blues",
    "Classical music": "Classical",
    "Country": "Country",
    "Electronic music": "Electronic",
    "Funk": "Funk",
    "Hip hop music": "Hip hop",
    "Jazz": "Jazz",
    "Pop music": "Pop",
    "Reggae": "Reggae",
    "Rock music": "Rock",
}

run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
if not run_name:
    print("FEHLER: Kein Dataset-Run gefunden.", flush=True)
    sys.exit(1)

run_dir = config.DATASETS_ROOT / run_name
split_csv = run_dir / "train_val_test_split.csv"
embeddings_dir = run_dir / "embeddings"

for label, p in [
    ("Dataset embeddings/video", embeddings_dir / "video"),
    ("Dataset embeddings/audio", embeddings_dir / "audio"),
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
        if v_path.exists() and a_path.exists():
            samples.append((video_id, label, v_path, a_path))

if not samples:
    print("FEHLER: Keine Test-Samples mit allen Embeddings gefunden.", flush=True)
    sys.exit(1)

print(f"Dataset-Run:  {run_name}", flush=True)
print(f"Test-Samples: {len(samples)}", flush=True)

labels = np.array([s[1] for s in samples])
genres = sorted(set(labels))
genre_to_idx = {g: i for i, g in enumerate(genres)}
genre_labels_short = [GENRE_SHORT.get(g, g) for g in genres]
label_idx = np.array([genre_to_idx[l] for l in labels])
n_genres = len(genres)

V = torch.tensor(np.stack([np.load(s[2]) for s in samples]), dtype=torch.float32, device=config.DEVICE)
A = torch.tensor(np.stack([np.load(s[3]) for s in samples]), dtype=torch.float32, device=config.DEVICE)

with torch.no_grad():
    vn = F.normalize(V, p=2, dim=-1)
    an = F.normalize(A, p=2, dim=-1)
    sim_va = (vn @ an.T).cpu().numpy()  # [n_video, n_audio]

idx_by_genre = {g: np.where(label_idx == g)[0] for g in range(n_genres)}

# Blockweises Mittel: mean_va[gi, gj] = mean cos(Video in gi, Audio in gj)
mean_va = np.full((n_genres, n_genres), np.nan, dtype=np.float64)
for gi in range(n_genres):
    for gj in range(n_genres):
        mean_va[gi, gj] = float(sim_va[np.ix_(idx_by_genre[gi], idx_by_genre[gj])].mean())

# A->V ist die transponierte Sicht: Zeile = Audio-Genre, Spalte = Video-Genre
mean_av = mean_va.T
mask_diag = np.eye(n_genres, dtype=bool)

fig, (ax_va, ax_av, cax) = plt.subplots(
    1, 3, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1, 1, 0.04]}
)
hm_kwargs = dict(
    mask=mask_diag, cmap="coolwarm", vmin=-1.0, vmax=1.0, center=0.0,
    annot=True, fmt=".2f", annot_kws={"size": 6}, linewidths=0.4, linecolor="white",
    xticklabels=genre_labels_short, yticklabels=genre_labels_short,
)

sns.heatmap(mean_va, ax=ax_va, cbar=False, **hm_kwargs)
ax_va.set_title("(a) V\u2192A (frozen)", fontsize=10)
ax_va.set_xlabel("Audio genre", fontsize=10)
ax_va.set_ylabel("Query genre (video)", fontsize=10)

sns.heatmap(mean_av, ax=ax_av, cbar=True, cbar_ax=cax,
            cbar_kws={"label": "Mean cosine similarity"}, **hm_kwargs)
ax_av.set_title("(b) A\u2192V (frozen)", fontsize=10)
ax_av.set_xlabel("Video genre", fontsize=10)
ax_av.set_ylabel("Query genre (audio)", fontsize=10)

for ax in (ax_va, ax_av):
    ax.tick_params(axis="both", labelsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
cax.tick_params(labelsize=8)

fig.tight_layout()
out_base = run_dir / "baseline_genrepairs"
fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Gespeichert: {out_base}.pdf", flush=True)
print(f"Gespeichert: {out_base}.png", flush=True)
