"""
E3b-Bump-Analyse: Welche Genre-Paare verursachen den negativen Bump (~-0.9) in der
Different-Genre-Verteilung der Cosine-Similarity?

Berechnet die volle Video×Audio-Cosine-Similarity-Matrix im E3b-Raum (Audio-Encoder
Genre) auf dem Test-Split und zerlegt die Different-Genre-Werte nach Genre-Paaren
(Zeile = Video-Genre der Query, Spalte = Audio-Genre).

Zwei Panels:
  (a) 10x10-Heatmap der mittleren cross-modalen Cosine-Similarity je Genre-Paar
      (Diagonale = same genre, ausmaskiert). Stark negative Off-Diagonal-Zellen sind
      die antipodalen Paare, die den Bump erzeugen.
  (b) Gestapeltes Histogramm der Different-Genre-Similarities: die Top-K Genre-Paare
      mit dem groessten Beitrag im Bump-Bereich (sim < BUMP_THRESHOLD) sind farbig,
      der Rest grau. Damit ist sichtbar, welche Paare den linken Bump bilden.

Env-Vars:
  DATASET_RUN_NAME   - Dataset-Run (default: neuester)
  TRAINING_RUN_DIR   - Run mit audio_encoder_genre.pt (auch als AE-Genre-Default)
  AE_GENRE_RUN_DIR   - ueberschreibt TRAINING_RUN_DIR fuer den Genre-Encoder
  BUMP_THRESHOLD     - Schwelle fuer den Bump-Bereich (default: -0.5)
  BUMP_TOPK          - Anzahl hervorgehobener Genre-Paare (default: 6)

Run: python3 plot_e3b_bump_genrepairs.py
     TRAINING_RUN_DIR=... python3 plot_e3b_bump_genrepairs.py
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
from models import load_audio_encoder_heads_genre

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

BUMP_THRESHOLD = float(os.environ.get("BUMP_THRESHOLD", "-0.5"))
TOPK = int(os.environ.get("BUMP_TOPK", "6"))

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
print(f"Bump-Schwelle: sim < {BUMP_THRESHOLD} | Top-K: {TOPK}", flush=True)

labels = [s[1] for s in samples]
genres = sorted(set(labels))
genre_to_idx = {g: i for i, g in enumerate(genres)}
genre_labels_short = [GENRE_SHORT.get(g, g) for g in genres]
label_idx = np.array([genre_to_idx[l] for l in labels])
n_genres = len(genres)

V = torch.tensor(np.stack([np.load(s[2]) for s in samples]), dtype=torch.float32, device=config.DEVICE)
A_ae_genre = torch.tensor(np.stack([np.load(s[3]) for s in samples]), dtype=torch.float32, device=config.DEVICE)

video_head, audio_head = load_audio_encoder_heads_genre(ae_genre_path)

with torch.no_grad():
    v = F.normalize(video_head(V), p=2, dim=-1)
    a = F.normalize(audio_head(A_ae_genre), p=2, dim=-1)
    sim = (v @ a.T).cpu().numpy()

idx_by_genre = {g: np.where(label_idx == g)[0] for g in range(n_genres)}

mean_mat = np.full((n_genres, n_genres), np.nan, dtype=np.float64)
pair_vals = {}
pair_bump_count = {}
for gi in range(n_genres):
    for gj in range(n_genres):
        block = sim[np.ix_(idx_by_genre[gi], idx_by_genre[gj])].ravel()
        mean_mat[gi, gj] = float(block.mean())
        if gi != gj:
            pair_vals[(gi, gj)] = block
            pair_bump_count[(gi, gj)] = int((block < BUMP_THRESHOLD).sum())

mask_diag = np.eye(n_genres, dtype=bool)

ranked = sorted(pair_bump_count.items(), key=lambda kv: kv[1], reverse=True)
total_bump = sum(pair_bump_count.values()) or 1
print("\n=== Top Genre-Paare im Bump-Bereich (sim < {:.2f}) ===".format(BUMP_THRESHOLD), flush=True)
print(f"{'Video genre':<12} {'Audio genre':<12} {'#bump':>8} {'%bump':>7} {'mean sim':>9}", flush=True)
for (gi, gj), cnt in ranked[:max(TOPK, 10)]:
    print(
        f"{genre_labels_short[gi]:<12} {genre_labels_short[gj]:<12} "
        f"{cnt:>8} {100.0 * cnt / total_bump:>6.1f}% {mean_mat[gi, gj]:>9.3f}",
        flush=True,
    )

top_pairs = [p for p, _ in ranked[:TOPK]]
top_set = set(top_pairs)
top_arrays = [pair_vals[p] for p in top_pairs]
top_labels = [f"{genre_labels_short[gi]}\u2192{genre_labels_short[gj]}" for gi, gj in top_pairs]
other_array = np.concatenate([pair_vals[p] for p in pair_vals if p not in top_set])

all_neg = np.concatenate(list(pair_vals.values()))
xmin, xmax = float(all_neg.min()), float(all_neg.max())
pad = 0.02 * (xmax - xmin)
bins = np.linspace(xmin - pad, xmax + pad, 80)

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
ax_hm.set_title("(a) Mean cross-modal similarity per genre pair (E3b)", fontsize=10)
ax_hm.set_xlabel("Audio genre", fontsize=10)
ax_hm.set_ylabel("Query genre (video)", fontsize=10)
ax_hm.tick_params(axis="both", labelsize=8)
ax_hm.set_xticklabels(ax_hm.get_xticklabels(), rotation=45, ha="right")
ax_hm.set_yticklabels(ax_hm.get_yticklabels(), rotation=0)

colors = list(plt.cm.tab10(np.linspace(0, 1, max(TOPK, 1))))
ax_hist.hist(
    [*top_arrays, other_array],
    bins=bins,
    stacked=True,
    color=[*colors[:TOPK], "#cccccc"],
    label=[*top_labels, "Other pairs"],
    edgecolor="none",
)
ax_hist.axvline(BUMP_THRESHOLD, color="#333333", linestyle="--", linewidth=1.0)
ax_hist.set_title("(b) Different-genre similarities, split by top genre pairs", fontsize=10)
ax_hist.set_xlabel("Cosine similarity", fontsize=10)
ax_hist.set_ylabel("Pair count", fontsize=10)
ax_hist.tick_params(axis="both", labelsize=9)
ax_hist.grid(which="major", axis="y", linestyle="-", linewidth=0.4, alpha=0.5)
for spine in ax_hist.spines.values():
    spine.set_linewidth(0.6)
    spine.set_color("#666666")
ax_hist.legend(fontsize=7, frameon=True, framealpha=0.9, edgecolor="#cccccc", ncol=1, loc="upper center")

fig.tight_layout()
out_base = run_dir / "e3b_bump_genrepairs"
fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nGespeichert: {out_base}.pdf", flush=True)
print(f"Gespeichert: {out_base}.png", flush=True)
