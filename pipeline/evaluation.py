"""
Evaluation: MRR, Recall@1/5/10, Mean Rank. V→A und A→V.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
import config
from dataset import PairDataset
from models import ProjectionHead, load_projection_heads_pair, load_projection_heads_genre, load_audio_encoder_heads_pair, load_audio_encoder_heads_genre
from metrics import (
    MRR,
    recall_at_k,
    mean_rank,
    labels_from_split_csv,
    label_relevance_matrix,
    pair_relevance_matrix,
)

# Run: aus Umgebung oder neuester
run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
if not run_name:
    print("FEHLER: DATASET_RUN_NAME nicht gesetzt und kein Run unter DATASETS_ROOT.", flush=True)
    sys.exit(1)
run_dir = config.DATASETS_ROOT / run_name
EMBEDDINGS_DIR = run_dir / "embeddings"
TRAIN_VAL_TEST_SPLIT_CSV = run_dir / "train_val_test_split.csv"
DEVICE = config.DEVICE

# Ein Ordner (Pipeline) oder neueste Einzel-Runs für Pair/Genre
if os.environ.get("TRAINING_RUN_DIR"):
    shared_run_dir = Path(os.environ["TRAINING_RUN_DIR"])
    pair_path = shared_run_dir if (shared_run_dir / "projection_heads_pair.pt").exists() else None
    genre_path = shared_run_dir if (shared_run_dir / "projection_heads_genre.pt").exists() else None
    pair_run_dir = shared_run_dir if pair_path else None
    genre_run_dir = shared_run_dir if genre_path else None
else:
    pair_run_dir = config.get_latest_training_run_with("projection_heads_pair.pt")
    genre_run_dir = config.get_latest_training_run_with("projection_heads_genre.pt")
    pair_path = pair_run_dir if pair_run_dir else None
    genre_path = genre_run_dir if genre_run_dir else None

# Audio-Encoder-Checkpoints (optional)
ae_pair_path = Path(os.environ["AE_PAIR_RUN_DIR"]) if os.environ.get("AE_PAIR_RUN_DIR") else config.get_latest_training_run_with("audio_encoder_pair.pt")
ae_genre_path = Path(os.environ["AE_GENRE_RUN_DIR"]) if os.environ.get("AE_GENRE_RUN_DIR") else config.get_latest_training_run_with("audio_encoder_genre.pt")


def _meta_commit(run_dir: Optional[Path], meta_name: str = "meta.json") -> str:
    if run_dir is None:
        return ""
    for name in (meta_name, "meta.json"):
        p = run_dir / name
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    m = json.load(f)
                return m.get("git_commit", "") or ""
            except Exception:
                pass
    return ""


# Ausgabe sammeln, damit wir sie bei Pipeline in den Run-Ordner schreiben können
_out_lines: List[str] = []


def _out(s: str) -> None:
    print(s, flush=True)
    _out_lines.append(s)


_out(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
_out(f"Git-Commit (Eval): {config.get_git_commit()}")
_out(f"Dataset-Run: {run_name}")
_out(f"Pair-based Head: {pair_path or config.PROJECTION_HEADS_PATH} (train commit: {_meta_commit(pair_run_dir, 'meta_pair.json') or '-'})")
_out(f"Genre-based Head: {genre_path or config.PROJECTION_HEADS_GENRE_PATH} (train commit: {_meta_commit(genre_run_dir, 'meta_genre.json') or '-'})")
_out(f"Audio-Encoder Pair: {ae_pair_path or '-'}")
_out(f"Audio-Encoder Genre: {ae_genre_path or '-'}")

test_ds = PairDataset("test", TRAIN_VAL_TEST_SPLIT_CSV, EMBEDDINGS_DIR)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
video_head_pair, audio_head_pair = load_projection_heads_pair(pair_path)
video_head_genre, audio_head_genre = load_projection_heads_genre(genre_path)
video_head_ae_pair, audio_head_ae_pair = load_audio_encoder_heads_pair(ae_pair_path)
video_head_ae_genre, audio_head_ae_genre = load_audio_encoder_heads_genre(ae_genre_path)

# Relevanz = Genre-Tag (Spalte "label"), gleiche Reihenfolge wie test_ds
labels = labels_from_split_csv(
    TRAIN_VAL_TEST_SPLIT_CSV, "test", relevance_column="label", embeddings_dir=EMBEDDINGS_DIR
)
_out("Labels shape: " + str(labels.shape))

# Pre-computed Embeddings (für head-only Modelle)
V_list, A_list = [], []
for v, a in test_loader:
    V_list.append(v)
    A_list.append(a)
V = torch.cat(V_list, dim=0).to(DEVICE)
A = torch.cat(A_list, dim=0).to(DEVICE)
Vn = F.normalize(V, p=2, dim=-1)
An = F.normalize(A, p=2, dim=-1)

# Pre-computed Audio-Encoder-Embeddings laden (in gleicher Reihenfolge wie test_ds)
def _load_ae_embeddings(run_path, subdir, samples):
    emb_dir = run_path / subdir
    embs = [torch.tensor(np.load(emb_dir / f"{video_id}.npy"), dtype=torch.float32) for video_id, *_ in samples]
    return torch.stack(embs).to(DEVICE)

A_ae_pair = _load_ae_embeddings(ae_pair_path, "audio_encoder_pair_test_embeddings", test_ds.samples) if ae_pair_path else None
A_ae_genre = _load_ae_embeddings(ae_genre_path, "audio_encoder_genre_test_embeddings", test_ds.samples) if ae_genre_path else None

# Ähnlichkeitsmatrizen
sim_baseline = (Vn @ An.T).cpu()
with torch.no_grad():
    video_head_rand = ProjectionHead().to(DEVICE).eval()
    audio_head_rand = ProjectionHead().to(DEVICE).eval()
    v_rand = F.normalize(video_head_rand(V), p=2, dim=-1)
    a_rand = F.normalize(audio_head_rand(A), p=2, dim=-1)

    v_pair = F.normalize(video_head_pair(V), p=2, dim=-1)
    a_pair = F.normalize(audio_head_pair(A), p=2, dim=-1)
    v_genre = F.normalize(video_head_genre(V), p=2, dim=-1)
    a_genre = F.normalize(audio_head_genre(A), p=2, dim=-1)

    v_ae_pair = F.normalize(video_head_ae_pair(V), p=2, dim=-1)
    a_ae_pair = F.normalize(audio_head_ae_pair(A_ae_pair), p=2, dim=-1)

    v_ae_genre = F.normalize(video_head_ae_genre(V), p=2, dim=-1)
    a_ae_genre = F.normalize(audio_head_ae_genre(A_ae_genre), p=2, dim=-1)

sim_rand = (v_rand @ a_rand.T).cpu()
sim_pair = (v_pair @ a_pair.T).cpu()
sim_genre = (v_genre @ a_genre.T).cpu()
sim_ae_pair = (v_ae_pair @ a_ae_pair.T).cpu()
sim_ae_genre = (v_ae_genre @ a_ae_genre.T).cpu()

rel_pair = pair_relevance_matrix(sim_baseline.size(0))
rel_label = label_relevance_matrix(labels)


def print_metrics(name, sim, relevance):
    _out(f"  {name}")
    _out(
        "    MRR: " + str(MRR(sim, relevance=relevance))
        + " | Recall@1: " + str(recall_at_k(sim, 1, relevance=relevance))
        + " | Recall@5: " + str(recall_at_k(sim, 5, relevance=relevance))
        + " | Recall@10: " + str(recall_at_k(sim, 10, relevance=relevance))
        + " | Mean Rank: " + str(mean_rank(sim, relevance=relevance))
    )


_out("=== Protokoll A: Pair-basierte Relevanz (exaktes Video-Audio-Paar) ===")
_out("=== V→A (Video als Query, Audio retrieval) ===")
print_metrics("Baseline", sim_baseline, rel_pair)
print_metrics("Untrained heads", sim_rand, rel_pair)
print_metrics("Pair-based", sim_pair, rel_pair)
print_metrics("Genre-based", sim_genre, rel_pair)
print_metrics("Audio-Encoder Pair", sim_ae_pair, rel_pair)
print_metrics("Audio-Encoder Genre", sim_ae_genre, rel_pair)
_out("")
_out("=== A→V (Audio als Query, Video retrieval) ===")
print_metrics("Baseline", sim_baseline.T, rel_pair.T)
print_metrics("Untrained heads", sim_rand.T, rel_pair.T)
print_metrics("Pair-based", sim_pair.T, rel_pair.T)
print_metrics("Genre-based", sim_genre.T, rel_pair.T)
print_metrics("Audio-Encoder Pair", sim_ae_pair.T, rel_pair.T)
print_metrics("Audio-Encoder Genre", sim_ae_genre.T, rel_pair.T)
_out("")
_out("=== Protokoll B: Label-basierte Relevanz (gleiches Genre) ===")
_out("=== V→A (Video als Query, Audio retrieval) ===")
print_metrics("Baseline", sim_baseline, rel_label)
print_metrics("Untrained heads", sim_rand, rel_label)
print_metrics("Pair-based", sim_pair, rel_label)
print_metrics("Genre-based", sim_genre, rel_label)
print_metrics("Audio-Encoder Pair", sim_ae_pair, rel_label)
print_metrics("Audio-Encoder Genre", sim_ae_genre, rel_label)
_out("")
_out("=== A→V (Audio als Query, Video retrieval) ===")
print_metrics("Baseline", sim_baseline.T, rel_label.T)
print_metrics("Untrained heads", sim_rand.T, rel_label.T)
print_metrics("Pair-based", sim_pair.T, rel_label.T)
print_metrics("Genre-based", sim_genre.T, rel_label.T)
print_metrics("Audio-Encoder Pair", sim_ae_pair.T, rel_label.T)
print_metrics("Audio-Encoder Genre", sim_ae_genre.T, rel_label.T)

# Bei Pipeline: Evaluation-Ausgabe in denselben Run-Ordner schreiben
if os.environ.get("TRAINING_RUN_DIR"):
    out_path = Path(os.environ["TRAINING_RUN_DIR"]) / "evaluation_output.txt"
    _out_lines.append("")
    _out_lines.append("Gespeichert: " + str(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(_out_lines))
    print("", flush=True)
    print("Gespeichert: " + str(out_path), flush=True)
