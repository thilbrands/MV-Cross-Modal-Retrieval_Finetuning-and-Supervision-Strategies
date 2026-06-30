"""
E2-Bump-Analyse (Genre-Head auf frozen Embeddings, SupCon).

Zwei Panels:
  (a) 10x10-Heatmap der mittleren cross-modalen Cosine-Similarity je Genre-Paar
      (Zeile = Video-Genre der Query, Spalte = Audio-Genre; Diagonale = same genre,
      ausmaskiert). Stark negative Off-Diagonal-Zellen = antipodale Genre-Paare.
  (b) Different-genre Similarities, aufgeteilt nach Cluster-Zugehoerigkeit:
      Within cluster  - beide Genres aus CLUSTER_GENRES {Electronic, Funk, Hip hop, Pop, Reggae}
      Between clusters - alle uebrigen different-genre Paare
      (Same genre absichtlich weggelassen - steckt schon im Similarity-Histogramm.)

E2 nutzt dieselben frozen Embeddings wie die Baseline, nur mit dem trainierten
Genre-Projektionskopf (kein Audio-Encoder).

Env-Vars:
  DATASET_RUN_NAME   - Dataset-Run (default: neuester)
  TRAINING_RUN_DIR   - Run mit projection_heads_genre.pt (auch als Genre-Default)
  GENRE_RUN_DIR      - ueberschreibt TRAINING_RUN_DIR fuer den Genre-Head
  BUMP_THRESHOLD     - gestrichelte Bump-Linie im Histogramm (default: -0.5)

Run: python3 plot_e2_bump_cluster.py
     TRAINING_RUN_DIR=... python3 plot_e2_bump_cluster.py
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
from models import load_projection_heads_genre

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
CLUSTER_GENRES = {"Electronic music", "Funk", "Hip hop music", "Pop music", "Reggae"}

COLOR_WITHIN = "#ff7f0e"
COLOR_BETWEEN = "#1f77b4"
BUMP_THRESHOLD = float(os.environ.get("BUMP_THRESHOLD", "-0.5"))

run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
if not run_name:
    print("FEHLER: Kein Dataset-Run gefunden.", flush=True)
    sys.exit(1)

run_dir = config.DATASETS_ROOT / run_name
split_csv = run_dir / "train_val_test_split.csv"
embeddings_dir = run_dir / "embeddings"

genre_path = (
    Path(os.environ["GENRE_RUN_DIR"]) if os.environ.get("GENRE_RUN_DIR")
    else Path(os.environ["TRAINING_RUN_DIR"]) if os.environ.get("TRAINING_RUN_DIR")
    else config.get_latest_training_run_with("projection_heads_genre.pt")
)
if not genre_path:
    print("FEHLER: Genre-Head-Run nicht gefunden. GENRE_RUN_DIR / TRAINING_RUN_DIR setzen.", flush=True)
    sys.exit(1)

for label, p in [
    ("Dataset embeddings/video", embeddings_dir / "video"),
    ("Dataset embeddings/audio", embeddings_dir / "audio"),
    ("Genre-Head", genre_path / "projection_heads_genre.pt"),
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
print(f"Genre-Run:    {genre_path}", flush=True)
print(f"Test-Samples: {len(samples)}", flush=True)
print(f"Bump-Linie:   sim = {BUMP_THRESHOLD}", flush=True)

labels = np.array([s[1] for s in samples])
genres = sorted(set(labels))
genre_to_idx = {g: i for i, g in enumerate(genres)}
genre_labels_short = [GENRE_SHORT.get(g, g) for g in genres]
label_idx = np.array([genre_to_idx[l] for l in labels])
n_genres = len(genres)
in_cluster = np.array([l in CLUSTER_GENRES for l in labels])

V = torch.tensor(np.stack([np.load(s[2]) for s in samples]), dtype=torch.float32, device=config.DEVICE)
A = torch.tensor(np.stack([np.load(s[3]) for s in samples]), dtype=torch.float32, device=config.DEVICE)

video_head, audio_head = load_projection_heads_genre(genre_path)

with torch.no_grad():
    v = F.normalize(video_head(V), p=2, dim=-1)
    a = F.normalize(audio_head(A), p=2, dim=-1)
    sim = (v @ a.T).cpu().numpy()

# --- Panel (a): Genre-Paar-Mittelwert-Matrix ---
idx_by_genre = {g: np.where(label_idx == g)[0] for g in range(n_genres)}
mean_mat = np.full((n_genres, n_genres), np.nan, dtype=np.float64)
for gi in range(n_genres):
    for gj in range(n_genres):
        mean_mat[gi, gj] = float(sim[np.ix_(idx_by_genre[gi], idx_by_genre[gj])].mean())
mask_diag = np.eye(n_genres, dtype=bool)

# --- Panel (b): Within vs. Between Cluster (nur different-genre) ---
same_mask = labels[:, None] == labels[None, :]
diff_mask = ~same_mask
both_in_cluster = in_cluster[:, None] & in_cluster[None, :]
within_vals = sim[diff_mask & both_in_cluster]
between_vals = sim[diff_mask & ~both_in_cluster]

print(f"\nCluster (within) = {sorted(CLUSTER_GENRES)}", flush=True)
print(f"Within cluster   n={within_vals.size:>9}  median={np.median(within_vals):.3f}  mean={within_vals.mean():.3f}", flush=True)
print(f"Between clusters n={between_vals.size:>9}  median={np.median(between_vals):.3f}  mean={between_vals.mean():.3f}", flush=True)

groups = [("Within cluster", within_vals, COLOR_WITHIN), ("Between clusters", between_vals, COLOR_BETWEEN)]
all_vals = np.concatenate([within_vals, between_vals])
xmin, xmax = float(all_vals.min()), float(all_vals.max())
pad = 0.02 * (xmax - xmin)
xlim = (xmin - pad, xmax + pad)
bins = np.linspace(xlim[0], xlim[1], 80)

fig, (ax_hm, ax_hist) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.0, 1.15]})

sns.heatmap(
    mean_mat,
    ax=ax_hm,
    mask=mask_diag,
    cmap="coolwarm",
    vmin=-1.0,
    vmax=1.0,
    center=0.0,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 6},
    linewidths=0.4,
    linecolor="white",
    xticklabels=genre_labels_short,
    yticklabels=genre_labels_short,
    cbar_kws={"label": "Mean cosine similarity", "shrink": 0.8},
)
ax_hm.set_title("(a) Mean cross-modal similarity per genre pair (E2)", fontsize=10)
ax_hm.set_xlabel("Audio genre", fontsize=10)
ax_hm.set_ylabel("Query genre (video)", fontsize=10)
ax_hm.tick_params(axis="both", labelsize=8)
ax_hm.set_xticklabels(ax_hm.get_xticklabels(), rotation=45, ha="right")
ax_hm.set_yticklabels(ax_hm.get_yticklabels(), rotation=0)

for name, vals, color in groups:
    ax_hist.hist(vals, bins=bins, density=True, color=color, alpha=0.5, label=name)
    ax_hist.axvline(np.median(vals), color=color, linestyle="--", linewidth=1.2)
ax_hist.axvline(BUMP_THRESHOLD, color="#333333", linestyle=":", linewidth=1.0)
ax_hist.set_title("(b) Different-genre similarities by cluster (E2)", fontsize=10)
ax_hist.set_xlabel("Cosine similarity", fontsize=10)
ax_hist.set_ylabel("Density", fontsize=10)
ax_hist.set_xlim(xlim)
ax_hist.tick_params(axis="both", labelsize=9)
ax_hist.grid(which="major", axis="y", linestyle="-", linewidth=0.4, alpha=0.5)
for spine in ax_hist.spines.values():
    spine.set_linewidth(0.6)
    spine.set_color("#666666")
ax_hist.legend(fontsize=7.5, frameon=True, framealpha=0.9, edgecolor="#cccccc")

fig.tight_layout()
out_base = run_dir / "e2_bump_cluster"
fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nGespeichert: {out_base}.pdf", flush=True)
print(f"Gespeichert: {out_base}.png", flush=True)
