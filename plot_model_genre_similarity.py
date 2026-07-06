"""
Genre-Cosine-Similarity zwischen Genre-Centroids — pro Modalität (Audio / Video).
Projiziertes Gegenstück zu genre_similarity.py (frozen) für E1, E2, E3a, E3b.

Pro Konfiguration ein eigenes PDF/PNG mit zwei Panels:
  links  — Audio: Centroid-Similarity im projizierten Audio-Raum
  rechts — Video: Centroid-Similarity im projizierten Video-Raum
Val-Split (wie genre_similarity.py).

Env-Vars:
  DATASET_RUN_NAME   — Dataset-Run (default: neuester)
  TRAINING_RUN_DIR   — Run mit allen Checkpoints (E1–E3b)
  PLOT_OUTPUT_DIR    — Ausgabeordner (default: TRAINING_RUN_DIR)

Run:
  TRAINING_RUN_DIR=/work2/.../training_runs/2026-07-04_19-06 python3 plot_model_genre_similarity.py
"""
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
import config
from dataset import RawAudioPairDataset
from models import (
    load_audio_encoder_heads_genre,
    load_audio_encoder_heads_pair,
    load_projection_heads_genre,
    load_projection_heads_pair,
    load_wav2clip_finetune,
)


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


def _similarity_matrix(genre_embs: dict[str, list[np.ndarray]], genres: list[str]) -> np.ndarray:
    centroids = np.stack([np.mean(genre_embs[g], axis=0) for g in genres])
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    return centroids @ centroids.T


def _group_projected(
    labels: list[str],
    video: torch.Tensor,
    audio: torch.Tensor,
    video_head,
    audio_head,
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    with torch.no_grad():
        v_proj = F.normalize(video_head(video), p=2, dim=-1).cpu().numpy()
        a_proj = F.normalize(audio_head(audio), p=2, dim=-1).cpu().numpy()
    genre_video: dict[str, list[np.ndarray]] = defaultdict(list)
    genre_audio: dict[str, list[np.ndarray]] = defaultdict(list)
    for label, v_vec, a_vec in zip(labels, v_proj, a_proj):
        genre_video[label].append(v_vec)
        genre_audio[label].append(a_vec)
    return genre_audio, genre_video


@torch.no_grad()
def _encode_val_audio(wav2clip_model, split_csv: Path, embeddings_dir: Path) -> dict[str, torch.Tensor]:
    ds = RawAudioPairDataset("val", split_csv, embeddings_dir, return_video_id=True)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    by_id: dict[str, torch.Tensor] = {}
    for video_ids, a_t, _ in loader:
        embs = wav2clip_model(a_t.to(config.DEVICE)).cpu()
        for video_id, emb in zip(video_ids, embs):
            by_id[video_id] = emb
    return by_id


def _save_figure(
    sim_audio: np.ndarray,
    sim_video: np.ndarray,
    model_label: str,
    genres: list[str],
    out_base: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)
    for ax, sim, modality in zip(
        axes,
        [sim_audio, sim_video],
        ["Audio", "Video"],
    ):
        sns.heatmap(
            sim,
            xticklabels=genres,
            yticklabels=genres,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            cbar=False,
            ax=ax,
        )
        ax.set_title(f"Genre Cosine Similarity — {modality} (projected, {model_label})\n(Val-Split)")
        ax.tick_params(axis="x", rotation=45)

    fig.colorbar(axes[1].collections[0], ax=axes, location="right", pad=0.02, shrink=0.85)
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", dpi=150, bbox_inches="tight")
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

for label, p in [
    ("projection_heads_pair.pt", training_run_dir / "projection_heads_pair.pt"),
    ("projection_heads_genre.pt", training_run_dir / "projection_heads_genre.pt"),
    ("audio_encoder_pair.pt", training_run_dir / "audio_encoder_pair.pt"),
    ("audio_encoder_genre.pt", training_run_dir / "audio_encoder_genre.pt"),
    ("Split-CSV", split_csv),
    ("embeddings/video", embeddings_dir / "video"),
    ("embeddings/audio", embeddings_dir / "audio"),
    ("embeddings/audio_raw", embeddings_dir / "audio_raw"),
]:
    if not p.exists():
        print(f"FEHLER: {label} nicht gefunden: {p}", flush=True)
        sys.exit(1)

val_samples = []
with open(split_csv, "r", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["split"].strip() != "val":
            continue
        video_id = row["video_id"].strip()
        label = row["label"].strip()
        v_path = embeddings_dir / "video" / f"{video_id}.npy"
        a_path = embeddings_dir / "audio" / f"{video_id}.npy"
        a_raw = embeddings_dir / "audio_raw" / f"{video_id}.npy"
        if v_path.exists() and a_path.exists() and a_raw.exists():
            val_samples.append((video_id, label, v_path, a_path))

if not val_samples:
    print("FEHLER: Keine Val-Samples mit frozen Embeddings gefunden.", flush=True)
    sys.exit(1)

labels = [s[1] for s in val_samples]
video_ids = [s[0] for s in val_samples]
genres = sorted(set(labels))
if not genres:
    print("FEHLER: Keine Genres auf Val-Split.", flush=True)
    sys.exit(1)

print(f"Dataset-Run:    {run_name}", flush=True)
print(f"Training-Run:   {training_run_dir}", flush=True)
print(f"Ausgabeordner:  {output_dir}", flush=True)
print(f"Val-Samples:    {len(val_samples)}", flush=True)
print(f"Genres ({len(genres)}): {genres}", flush=True)

V = torch.tensor(np.stack([np.load(s[2]) for s in val_samples]), dtype=torch.float32, device=config.DEVICE)
A = torch.tensor(np.stack([np.load(s[3]) for s in val_samples]), dtype=torch.float32, device=config.DEVICE)

video_head_pair, audio_head_pair = load_projection_heads_pair(training_run_dir)
video_head_genre, audio_head_genre = load_projection_heads_genre(training_run_dir)
video_head_ae_pair, audio_head_ae_pair = load_audio_encoder_heads_pair(training_run_dir)
video_head_ae_genre, audio_head_ae_genre = load_audio_encoder_heads_genre(training_run_dir)

ckpt_ae_pair = torch.load(training_run_dir / "audio_encoder_pair.pt", map_location=config.DEVICE)
wav2clip_pair = load_wav2clip_finetune(config.DEVICE, unfreeze="full")
wav2clip_pair.load_state_dict(ckpt_ae_pair["wav2clip"])
wav2clip_pair.eval()

ckpt_ae_genre = torch.load(training_run_dir / "audio_encoder_genre.pt", map_location=config.DEVICE)
wav2clip_genre = load_wav2clip_finetune(config.DEVICE, unfreeze="full")
wav2clip_genre.load_state_dict(ckpt_ae_genre["wav2clip"])
wav2clip_genre.eval()

ae_pair_by_id = _encode_val_audio(wav2clip_pair, split_csv, embeddings_dir)
ae_genre_by_id = _encode_val_audio(wav2clip_genre, split_csv, embeddings_dir)
missing_pair = [vid for vid in video_ids if vid not in ae_pair_by_id]
missing_genre = [vid for vid in video_ids if vid not in ae_genre_by_id]
if missing_pair or missing_genre:
    print(f"FEHLER: AE-Val-Embeddings fehlen (pair={len(missing_pair)}, genre={len(missing_genre)}).", flush=True)
    sys.exit(1)

A_ae_pair = torch.stack([ae_pair_by_id[vid] for vid in video_ids]).to(config.DEVICE)
A_ae_genre = torch.stack([ae_genre_by_id[vid] for vid in video_ids]).to(config.DEVICE)

configs = [
    ("e1", "E1", V, A, video_head_pair, audio_head_pair),
    ("e2", "E2", V, A, video_head_genre, audio_head_genre),
    ("e3a", "E3a", V, A_ae_pair, video_head_ae_pair, audio_head_ae_pair),
    ("e3b", "E3b", V, A_ae_genre, video_head_ae_genre, audio_head_ae_genre),
]

for file_stem, model_label, video, audio, video_head, audio_head in configs:
    genre_audio, genre_video = _group_projected(labels, video, audio, video_head, audio_head)
    sim_audio = _similarity_matrix(genre_audio, genres)
    sim_video = _similarity_matrix(genre_video, genres)
    _save_figure(sim_audio, sim_video, model_label, genres, output_dir / f"{file_stem}_genre_similarity")
