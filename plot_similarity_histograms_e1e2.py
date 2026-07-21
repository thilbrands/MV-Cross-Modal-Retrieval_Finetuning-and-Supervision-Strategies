"""
Cosine-Similarity-Histogramme: Same-Genre (positiv) vs. Different-Genre (negativ).
Drei Panels: (a) Baseline (frozen), (b) E1 (Pair-Head), (c) E2 (Genre-Head).

E1/E2 nutzen dieselben frozen Embeddings wie die Baseline, nur mit den trainierten
Projektionskoepfen (kein Audio-Encoder). Berechnet die volle Video x Audio
Cosine-Similarity-Matrix auf dem Test-Split und teilt nach Protocol B
(gleiches Genre = positiv) auf.

Env-Vars:
  DATASET_RUN_NAME   - Dataset-Run (default: neuester)
  TRAINING_RUN_DIR   - Run mit projection_heads_pair.pt / projection_heads_genre.pt
  PAIR_RUN_DIR / GENRE_RUN_DIR - ueberschreiben TRAINING_RUN_DIR pro Head

Run: python3 plot_similarity_histograms_e1e2.py
     TRAINING_RUN_DIR=... python3 plot_similarity_histograms_e1e2.py
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
from models import load_projection_heads_pair, load_projection_heads_genre
from similarity_centering import (
    assert_ranking_unchanged,
    center_rows,
    median_gap,
    query_matrix,
    same_diff_values,
    zscore_rows,
)

COLOR_SAME = "#2ca02c"
COLOR_DIFF = "#d62728"

run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
if not run_name:
    print("FEHLER: Kein Dataset-Run gefunden.", flush=True)
    sys.exit(1)

run_dir = config.DATASETS_ROOT / run_name
split_csv = run_dir / "train_val_test_split.csv"
embeddings_dir = run_dir / "embeddings"

pair_path = (
    Path(os.environ["PAIR_RUN_DIR"]) if os.environ.get("PAIR_RUN_DIR")
    else Path(os.environ["TRAINING_RUN_DIR"]) if os.environ.get("TRAINING_RUN_DIR")
    else config.get_latest_training_run_with("projection_heads_pair.pt")
)
genre_path = (
    Path(os.environ["GENRE_RUN_DIR"]) if os.environ.get("GENRE_RUN_DIR")
    else Path(os.environ["TRAINING_RUN_DIR"]) if os.environ.get("TRAINING_RUN_DIR")
    else config.get_latest_training_run_with("projection_heads_genre.pt")
)
if not pair_path or not genre_path:
    print("FEHLER: Head-Run nicht gefunden. PAIR_RUN_DIR / GENRE_RUN_DIR / TRAINING_RUN_DIR setzen.", flush=True)
    sys.exit(1)

for label, p in [
    ("Dataset embeddings/video", embeddings_dir / "video"),
    ("Dataset embeddings/audio", embeddings_dir / "audio"),
    ("Pair-Head", pair_path / "projection_heads_pair.pt"),
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
print(f"Pair-Run:     {pair_path}", flush=True)
print(f"Genre-Run:    {genre_path}", flush=True)
print(f"Test-Samples: {len(samples)}", flush=True)

labels = np.array([s[1] for s in samples])
same_genre_mask = labels[:, None] == labels[None, :]

V = torch.tensor(np.stack([np.load(s[2]) for s in samples]), dtype=torch.float32, device=config.DEVICE)
A = torch.tensor(np.stack([np.load(s[3]) for s in samples]), dtype=torch.float32, device=config.DEVICE)

video_head_pair, audio_head_pair = load_projection_heads_pair(pair_path)
video_head_genre, audio_head_genre = load_projection_heads_genre(genre_path)


def _sim_matrix(video, audio):
    with torch.no_grad():
        vn = F.normalize(video, p=2, dim=-1)
        an = F.normalize(audio, p=2, dim=-1)
        return (vn @ an.T).cpu().numpy()


def _projected_sim_matrix(video_head, audio_head, video, audio):
    with torch.no_grad():
        v = F.normalize(video_head(video), p=2, dim=-1)
        a = F.normalize(audio_head(audio), p=2, dim=-1)
        return (v @ a.T).cpu().numpy()


sim_baseline = _sim_matrix(V, A)
sim_e1 = _projected_sim_matrix(video_head_pair, audio_head_pair, V, A)
sim_e2 = _projected_sim_matrix(video_head_genre, audio_head_genre, V, A)

models = [
    ("(a) Baseline", sim_baseline),
    ("(b) E1", sim_e1),
    ("(c) E2", sim_e2),
]

pos_neg = []
for title, sim in models:
    pos = sim[same_genre_mask]
    neg = sim[~same_genre_mask]
    pos_neg.append((title, pos, neg))
    print(f"{title}: same median={np.median(pos):.3f}  diff median={np.median(neg):.3f}", flush=True)

all_vals = np.concatenate([np.concatenate([p, n]) for _, p, n in pos_neg])
xmin, xmax = float(all_vals.min()), float(all_vals.max())
pad = 0.02 * (xmax - xmin)
xlim = (xmin - pad, xmax + pad)
bins = np.linspace(xlim[0], xlim[1], 80)

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharex=True, sharey=True)
for ax, (title, pos, neg) in zip(axes, pos_neg):
    ax.hist(pos, bins=bins, density=True, color=COLOR_SAME, alpha=0.6, label="Same genre")
    ax.hist(neg, bins=bins, density=True, color=COLOR_DIFF, alpha=0.6, label="Different genre")
    ax.axvline(np.median(pos), color=COLOR_SAME, linestyle="--", linewidth=1.2)
    ax.axvline(np.median(neg), color=COLOR_DIFF, linestyle="--", linewidth=1.2)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Cosine similarity", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(which="major", axis="y", linestyle="-", linewidth=0.4, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#666666")

axes[0].set_ylabel("Density", fontsize=10)
axes[0].set_xlim(xlim)
axes[-1].legend(fontsize=7.5, frameon=True, framealpha=0.9, edgecolor="#cccccc")

fig.tight_layout()
out_base = run_dir / "similarity_histograms_e1e2"
fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Gespeichert: {out_base}.pdf", flush=True)
print(f"Gespeichert: {out_base}.png", flush=True)

# --- Per-Query-zentrierte Histogramme (V2A: Zeilen=Video-Queries, axis=1=Kandidaten) ---
gap_rows = []


def _append_gap_rows(model_key: str, model_title: str, sim_va: np.ndarray, same_mask: np.ndarray):
    for direction in ("V2A", "A2V"):
        sim_q = query_matrix(sim_va, direction)
        rel_dir = direction
        for variant, sim_use in (
            ("raw", sim_q),
            ("centered", center_rows(sim_q)),
            ("zscore", zscore_rows(sim_q)),
        ):
            if variant != "raw":
                assert_ranking_unchanged(sim_q, sim_use, labels, f"{model_key}_{direction}_{variant}")
            pos, neg = same_diff_values(sim_use, same_mask)
            gap_rows.append({
                "model_key": model_key,
                "model": model_title,
                "direction": rel_dir,
                "variant": variant,
                "median_same_genre": float(np.median(pos)),
                "median_diff_genre": float(np.median(neg)),
                "median_gap": median_gap(pos, neg),
            })


def _plot_centered_panels(models_centered, out_name: str, xlim, bins):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharex=True, sharey=True)
    for ax, (title, pos, neg) in zip(axes, models_centered):
        ax.hist(pos, bins=bins, density=True, color=COLOR_SAME, alpha=0.6, label="Same genre")
        ax.hist(neg, bins=bins, density=True, color=COLOR_DIFF, alpha=0.6, label="Different genre")
        ax.axvline(np.median(pos), color=COLOR_SAME, linestyle="--", linewidth=1.2)
        ax.axvline(np.median(neg), color=COLOR_DIFF, linestyle="--", linewidth=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Query-centered cosine similarity", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(which="major", axis="y", linestyle="-", linewidth=0.4, alpha=0.5)
    axes[0].set_ylabel("Density", fontsize=10)
    axes[-1].legend(fontsize=7.5, frameon=True, framealpha=0.9, edgecolor="#cccccc")
    fig.tight_layout()
    out_base = run_dir / out_name
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Gespeichert: {out_base}.pdf", flush=True)


model_defs = [
    ("baseline", "(a) Baseline", sim_baseline),
    ("pair", "(b) E1", sim_e1),
    ("genre", "(c) E2", sim_e2),
]

for model_key, model_title, sim_va in model_defs:
    _append_gap_rows(model_key, model_title.replace("(", "").replace(")", "").strip(), sim_va, same_genre_mask)

# Histogramme: V2A-query-zentriert (axis=1 über Audio-Kandidaten pro Video-Query)
centered_panels = []
zscore_panels = []
for title, sim_va in models:
    sim_q = query_matrix(sim_va, "V2A")
    sim_c = center_rows(sim_q)
    sim_z = zscore_rows(sim_q)
    assert_ranking_unchanged(sim_q, sim_c, labels, title + "_centered")
    assert_ranking_unchanged(sim_q, sim_z, labels, title + "_zscore")
    pos_c, neg_c = same_diff_values(sim_c, same_genre_mask)
    pos_z, neg_z = same_diff_values(sim_z, same_genre_mask)
    centered_panels.append((title + " (centered)", pos_c, neg_c))
    zscore_panels.append((title + " (z-score)", pos_z, neg_z))
    print(
        f"{title} centered: gap={median_gap(pos_c, neg_c):.4f}  "
        f"(raw {median_gap(sim_va[same_genre_mask], sim_va[~same_genre_mask]):.4f})",
        flush=True,
    )

all_c = np.concatenate([np.concatenate([p, n]) for _, p, n in centered_panels])
xmin, xmax = float(all_c.min()), float(all_c.max())
pad = 0.02 * (xmax - xmin) if xmax > xmin else 0.01
xlim_c = (xmin - pad, xmax + pad)
bins_c = np.linspace(xlim_c[0], xlim_c[1], 80)
_plot_centered_panels(centered_panels, "similarity_histograms_e1e2_centered", xlim_c, bins_c)

all_z = np.concatenate([np.concatenate([p, n]) for _, p, n in zscore_panels])
xmin, xmax = float(all_z.min()), float(all_z.max())
pad = 0.02 * (xmax - xmin) if xmax > xmin else 0.01
xlim_z = (xmin - pad, xmax + pad)
bins_z = np.linspace(xlim_z[0], xlim_z[1], 80)
_plot_centered_panels(zscore_panels, "similarity_histograms_e1e2_zscore", xlim_z, bins_z)

gap_csv = run_dir / "similarity_median_gap_e1e2.csv"
with open(gap_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model_key", "model", "direction", "variant",
            "median_same_genre", "median_diff_genre", "median_gap",
        ],
    )
    writer.writeheader()
    writer.writerows(gap_rows)
print(f"Gespeichert: {gap_csv}", flush=True)
