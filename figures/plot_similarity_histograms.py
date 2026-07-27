"""
Cosine-Similarity-Histogramme: Same-Genre (positiv) vs. Different-Genre (negativ).
Zwei Panels: (a) Baseline (frozen), (b) E3a (Audio-Encoder Pair, bestes Modell).

Berechnet die volle Video×Audio-Cosine-Similarity-Matrix auf dem Test-Split und
teilt die Werte nach Protocol B (gleiches Genre = positiv) auf.

Run: python3 plot_similarity_histograms.py
     TRAINING_RUN_DIR=... AE_PAIR_RUN_DIR=... python3 plot_similarity_histograms.py
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

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "configs"))
import config
from models import load_audio_encoder_heads_pair, load_audio_encoder_heads_genre

COLOR_SAME = "#2ca02c"
COLOR_DIFF = "#d62728"

run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
if not run_name:
    print("FEHLER: Kein Dataset-Run gefunden.", flush=True)
    sys.exit(1)

run_dir = config.DATASETS_ROOT / run_name
split_csv = run_dir / "train_val_test_split.csv"
embeddings_dir = run_dir / "embeddings"

ae_pair_path = (
    Path(os.environ["AE_PAIR_RUN_DIR"]) if os.environ.get("AE_PAIR_RUN_DIR")
    else Path(os.environ["TRAINING_RUN_DIR"]) if os.environ.get("TRAINING_RUN_DIR")
    else config.get_latest_training_run_with("audio_encoder_pair.pt")
)
ae_genre_path = (
    Path(os.environ["AE_GENRE_RUN_DIR"]) if os.environ.get("AE_GENRE_RUN_DIR")
    else Path(os.environ["TRAINING_RUN_DIR"]) if os.environ.get("TRAINING_RUN_DIR")
    else config.get_latest_training_run_with("audio_encoder_genre.pt")
)
if not ae_pair_path or not ae_genre_path:
    print("FEHLER: AE-Run nicht gefunden. AE_PAIR_RUN_DIR / AE_GENRE_RUN_DIR / TRAINING_RUN_DIR setzen.", flush=True)
    sys.exit(1)

ae_pair_emb_dir = ae_pair_path / "audio_encoder_pair_test_embeddings"
ae_genre_emb_dir = ae_genre_path / "audio_encoder_genre_test_embeddings"
for label, p in [
    ("Dataset embeddings/video", embeddings_dir / "video"),
    ("Dataset embeddings/audio", embeddings_dir / "audio"),
    ("AE-Pair Test-Embeddings", ae_pair_emb_dir),
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
        a_path = embeddings_dir / "audio" / f"{video_id}.npy"
        ae_pair_emb = ae_pair_emb_dir / f"{video_id}.npy"
        ae_genre_emb = ae_genre_emb_dir / f"{video_id}.npy"
        if v_path.exists() and a_path.exists() and ae_pair_emb.exists() and ae_genre_emb.exists():
            samples.append((video_id, label, v_path, a_path, ae_pair_emb, ae_genre_emb))

if not samples:
    print("FEHLER: Keine Test-Samples mit allen Embeddings gefunden.", flush=True)
    sys.exit(1)

print(f"Dataset-Run:  {run_name}", flush=True)
print(f"AE-Pair-Run:  {ae_pair_path}", flush=True)
print(f"AE-Genre-Run: {ae_genre_path}", flush=True)
print(f"Test-Samples: {len(samples)}", flush=True)

labels = np.array([s[1] for s in samples])
same_genre_mask = labels[:, None] == labels[None, :]

V = torch.tensor(np.stack([np.load(s[2]) for s in samples]), dtype=torch.float32, device=config.DEVICE)
A = torch.tensor(np.stack([np.load(s[3]) for s in samples]), dtype=torch.float32, device=config.DEVICE)
A_ae_pair = torch.tensor(np.stack([np.load(s[4]) for s in samples]), dtype=torch.float32, device=config.DEVICE)
A_ae_genre = torch.tensor(np.stack([np.load(s[5]) for s in samples]), dtype=torch.float32, device=config.DEVICE)

video_head_ae_pair, audio_head_ae_pair = load_audio_encoder_heads_pair(ae_pair_path)
video_head_ae_genre, audio_head_ae_genre = load_audio_encoder_heads_genre(ae_genre_path)


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
sim_e3a = _projected_sim_matrix(video_head_ae_pair, audio_head_ae_pair, V, A_ae_pair)
sim_e3b = _projected_sim_matrix(video_head_ae_genre, audio_head_ae_genre, V, A_ae_genre)

models = [
    ("(a) Baseline", sim_baseline),
    ("(b) E3a", sim_e3a),
    ("(c) E3b", sim_e3b),
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
out_base = config.resolve_plot_output_dir(run_dir) / "similarity_histograms"
fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Gespeichert: {out_base}.pdf", flush=True)
print(f"Gespeichert: {out_base}.png", flush=True)
