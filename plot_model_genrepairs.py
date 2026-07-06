"""
Genre-Paar-Heatmaps im projizierten Raum — Gegenstück zu plot_baseline_genrepairs.py.

Pro Konfiguration (E1, E2, E3a, E3b) ein eigenes PDF/PNG mit zwei Panels:
  (a) V→A: Query-Genre (Video), Spalte = Audio-Genre
  (b) A→V: Query-Genre (Audio), Spalte = Video-Genre
Diagonale (same genre) ausmaskiert.

Alle Checkpoints kommen standardmäßig aus einem gemeinsamen TRAINING_RUN_DIR
(z. B. 2026-07-04_19-06).

Env-Vars:
  DATASET_RUN_NAME   — Dataset-Run (default: neuester)
  TRAINING_RUN_DIR   — Run mit projection_heads_*.pt und audio_encoder_*.pt
  PLOT_OUTPUT_DIR    — Ausgabeordner (default: TRAINING_RUN_DIR)

Run:
  TRAINING_RUN_DIR=/work2/.../training_runs/2026-07-04_19-06 python3 plot_model_genrepairs.py
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
from models import (
    load_audio_encoder_heads_genre,
    load_audio_encoder_heads_pair,
    load_projection_heads_genre,
    load_projection_heads_pair,
)

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


def _training_run_dir() -> Path:
    if os.environ.get("TRAINING_RUN_DIR"):
        return Path(os.environ["TRAINING_RUN_DIR"])
    for filename in (
        "projection_heads_pair.pt",
        "projection_heads_genre.pt",
        "audio_encoder_pair.pt",
        "audio_encoder_genre.pt",
    ):
        run = config.get_latest_training_run_with(filename)
        if run:
            return run
    print("FEHLER: TRAINING_RUN_DIR nicht gesetzt und kein passender Run gefunden.", flush=True)
    sys.exit(1)


def _projected_sim(video, audio, video_head, audio_head):
    with torch.no_grad():
        v = F.normalize(video_head(video), p=2, dim=-1)
        a = F.normalize(audio_head(audio), p=2, dim=-1)
        return (v @ a.T).cpu().numpy()


def _mean_genre_matrices(sim: np.ndarray, label_idx: np.ndarray, n_genres: int):
    idx_by_genre = {g: np.where(label_idx == g)[0] for g in range(n_genres)}
    mean_va = np.full((n_genres, n_genres), np.nan, dtype=np.float64)
    for gi in range(n_genres):
        for gj in range(n_genres):
            mean_va[gi, gj] = float(sim[np.ix_(idx_by_genre[gi], idx_by_genre[gj])].mean())
    return mean_va, mean_va.T


def _save_genrepairs_figure(
    mean_va: np.ndarray,
    mean_av: np.ndarray,
    model_label: str,
    genre_labels_short: list[str],
    mask_diag: np.ndarray,
    out_base: Path,
) -> None:
    fig, (ax_va, ax_av, cax) = plt.subplots(
        1, 3, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1, 1, 0.04]}
    )
    hm_kwargs = dict(
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
    )

    sns.heatmap(mean_va, ax=ax_va, cbar=False, **hm_kwargs)
    ax_va.set_title(f"(a) V\u2192A ({model_label})", fontsize=10)
    ax_va.set_xlabel("Audio genre", fontsize=10)
    ax_va.set_ylabel("Query genre (video)", fontsize=10)

    sns.heatmap(mean_av, ax=ax_av, cbar=True, cbar_ax=cax, cbar_kws={"label": "Mean cosine similarity"}, **hm_kwargs)
    ax_av.set_title(f"(b) A\u2192V ({model_label})", fontsize=10)
    ax_av.set_xlabel("Video genre", fontsize=10)
    ax_av.set_ylabel("Query genre (audio)", fontsize=10)

    for ax in (ax_va, ax_av):
        ax.tick_params(axis="both", labelsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    cax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Gespeichert: {out_base}.pdf", flush=True)
    print(f"Gespeichert: {out_base}.png", flush=True)


run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
if not run_name:
    print("FEHLER: Kein Dataset-Run gefunden.", flush=True)
    sys.exit(1)

training_run_dir = _training_run_dir()
output_dir = Path(os.environ["PLOT_OUTPUT_DIR"]) if os.environ.get("PLOT_OUTPUT_DIR") else training_run_dir
output_dir.mkdir(parents=True, exist_ok=True)

run_dir = config.DATASETS_ROOT / run_name
split_csv = run_dir / "train_val_test_split.csv"
embeddings_dir = run_dir / "embeddings"
ae_pair_emb_dir = training_run_dir / "audio_encoder_pair_test_embeddings"
ae_genre_emb_dir = training_run_dir / "audio_encoder_genre_test_embeddings"

required = [
    ("projection_heads_pair.pt", training_run_dir / "projection_heads_pair.pt"),
    ("projection_heads_genre.pt", training_run_dir / "projection_heads_genre.pt"),
    ("audio_encoder_pair.pt", training_run_dir / "audio_encoder_pair.pt"),
    ("audio_encoder_genre.pt", training_run_dir / "audio_encoder_genre.pt"),
    ("E3a Test-Embeddings", ae_pair_emb_dir),
    ("E3b Test-Embeddings", ae_genre_emb_dir),
    ("Dataset embeddings/video", embeddings_dir / "video"),
    ("Dataset embeddings/audio", embeddings_dir / "audio"),
    ("Split-CSV", split_csv),
]
for label, p in required:
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
        if all(p.exists() for p in (v_path, a_path, ae_pair_emb, ae_genre_emb)):
            samples.append((label, v_path, a_path, ae_pair_emb, ae_genre_emb))

if not samples:
    print("FEHLER: Keine Test-Samples mit allen Embeddings gefunden.", flush=True)
    sys.exit(1)

print(f"Dataset-Run:    {run_name}", flush=True)
print(f"Training-Run:   {training_run_dir}", flush=True)
print(f"Ausgabeordner:  {output_dir}", flush=True)
print(f"Test-Samples:   {len(samples)}", flush=True)

labels = np.array([s[0] for s in samples])
genres = sorted(set(labels))
genre_labels_short = [GENRE_SHORT.get(g, g) for g in genres]
genre_to_idx = {g: i for i, g in enumerate(genres)}
label_idx = np.array([genre_to_idx[l] for l in labels])
n_genres = len(genres)
mask_diag = np.eye(n_genres, dtype=bool)

V = torch.tensor(np.stack([np.load(s[1]) for s in samples]), dtype=torch.float32, device=config.DEVICE)
A = torch.tensor(np.stack([np.load(s[2]) for s in samples]), dtype=torch.float32, device=config.DEVICE)
A_ae_pair = torch.tensor(np.stack([np.load(s[3]) for s in samples]), dtype=torch.float32, device=config.DEVICE)
A_ae_genre = torch.tensor(np.stack([np.load(s[4]) for s in samples]), dtype=torch.float32, device=config.DEVICE)

video_head_pair, audio_head_pair = load_projection_heads_pair(training_run_dir)
video_head_genre, audio_head_genre = load_projection_heads_genre(training_run_dir)
video_head_ae_pair, audio_head_ae_pair = load_audio_encoder_heads_pair(training_run_dir)
video_head_ae_genre, audio_head_ae_genre = load_audio_encoder_heads_genre(training_run_dir)

configs = [
    ("e1", "E1", V, A, video_head_pair, audio_head_pair),
    ("e2", "E2", V, A, video_head_genre, audio_head_genre),
    ("e3a", "E3a", V, A_ae_pair, video_head_ae_pair, audio_head_ae_pair),
    ("e3b", "E3b", V, A_ae_genre, video_head_ae_genre, audio_head_ae_genre),
]

for file_stem, model_label, video, audio, video_head, audio_head in configs:
    sim = _projected_sim(video, audio, video_head, audio_head)
    mean_va, mean_av = _mean_genre_matrices(sim, label_idx, n_genres)
    _save_genrepairs_figure(
        mean_va,
        mean_av,
        model_label,
        genre_labels_short,
        mask_diag,
        output_dir / f"{file_stem}_genrepairs",
    )
