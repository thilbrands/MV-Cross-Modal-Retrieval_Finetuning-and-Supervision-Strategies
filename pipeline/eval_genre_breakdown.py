"""
Genre-Breakdown-Evaluation für E4-Exploration.
Berechnet Retrieval-Metriken pro Genre und aggregiert nach Seen/Unseen.
Beide Protokolle: A (exaktes Paar) und B (gleiches Genre).

Env-Vars:
  TRAINING_RUN_DIR  — Ordner mit den Heads (z.B. e4_exploration/)
  DATASET_RUN_NAME  — Dataset-Run (default: neuester)
  TRAIN_GENRES      — kommagetrennte Trainings-Genres; wenn gesetzt, wird
                      Seen/Unseen-Trennung durchgeführt
  AE_PAIR_RUN_DIR   — Ordner mit audio_encoder_pair.pt (default: TRAINING_RUN_DIR)
  AE_GENRE_RUN_DIR  — Ordner mit audio_encoder_genre.pt (default: TRAINING_RUN_DIR)
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
import config
from dataset import PairDataset
from models import (
    ProjectionHead,
    load_projection_heads_pair,
    load_projection_heads_genre,
    load_audio_encoder_heads_pair,
    load_audio_encoder_heads_genre,
)
from metrics import labels_from_split_csv, label_relevance_matrix, pair_relevance_matrix, MRR, recall_at_k, mean_rank
from training_metrics import save_genre_breakdown_csv

# --- Pfade ---
run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
if not run_name:
    print("FEHLER: DATASET_RUN_NAME nicht gesetzt.", flush=True)
    sys.exit(1)
run_dir = config.DATASETS_ROOT / run_name
EMBEDDINGS_DIR = run_dir / "embeddings"
TRAIN_VAL_TEST_SPLIT_CSV = run_dir / "train_val_test_split.csv"
DEVICE = config.DEVICE

if os.environ.get("TRAINING_RUN_DIR"):
    shared_run_dir = Path(os.environ["TRAINING_RUN_DIR"])
    pair_path = shared_run_dir if (shared_run_dir / "projection_heads_pair.pt").exists() else None
    genre_path = shared_run_dir if (shared_run_dir / "projection_heads_genre.pt").exists() else None
else:
    pair_path = config.get_latest_training_run_with("projection_heads_pair.pt")
    genre_path = config.get_latest_training_run_with("projection_heads_genre.pt")
    shared_run_dir = genre_path or pair_path

ae_pair_path = Path(os.environ["AE_PAIR_RUN_DIR"]) if os.environ.get("AE_PAIR_RUN_DIR") else config.get_latest_training_run_with("audio_encoder_pair.pt")
ae_genre_path = Path(os.environ["AE_GENRE_RUN_DIR"]) if os.environ.get("AE_GENRE_RUN_DIR") else config.get_latest_training_run_with("audio_encoder_genre.pt")

# Seen/Unseen aus TRAIN_GENRES
_train_genres_env = os.environ.get("TRAIN_GENRES")
seen_genres = set(_train_genres_env.split(",")) if _train_genres_env else None

# --- Daten laden ---
test_ds = PairDataset("test", TRAIN_VAL_TEST_SPLIT_CSV, EMBEDDINGS_DIR)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

labels = labels_from_split_csv(TRAIN_VAL_TEST_SPLIT_CSV, "test", relevance_column="label", embeddings_dir=EMBEDDINGS_DIR)
all_genres = sorted(set(labels.tolist()))

V_list, A_list = [], []
for v, a in test_loader:
    V_list.append(v)
    A_list.append(a)
V = torch.cat(V_list, dim=0).to(DEVICE)
A = torch.cat(A_list, dim=0).to(DEVICE)
Vn = F.normalize(V, p=2, dim=-1)
An = F.normalize(A, p=2, dim=-1)

def _load_ae_embeddings(run_path, subdir):
    emb_dir = run_path / subdir
    embs = [torch.tensor(np.load(emb_dir / f"{vid}.npy"), dtype=torch.float32) for vid, *_ in test_ds.samples]
    return torch.stack(embs).to(DEVICE)

A_ae_pair = _load_ae_embeddings(ae_pair_path, "audio_encoder_pair_test_embeddings") if ae_pair_path else None
A_ae_genre = _load_ae_embeddings(ae_genre_path, "audio_encoder_genre_test_embeddings") if ae_genre_path else None

# --- Heads laden ---
video_head_pair, audio_head_pair = load_projection_heads_pair(pair_path)
video_head_genre, audio_head_genre = load_projection_heads_genre(genre_path)
video_head_ae_pair, audio_head_ae_pair = load_audio_encoder_heads_pair(ae_pair_path)
video_head_ae_genre, audio_head_ae_genre = load_audio_encoder_heads_genre(ae_genre_path)

# --- Ähnlichkeitsmatrizen berechnen ---
sim_baseline = (Vn @ An.T).cpu()
with torch.no_grad():
    v_rand = F.normalize(ProjectionHead().to(DEVICE).eval()(V), p=2, dim=-1)
    a_rand = F.normalize(ProjectionHead().to(DEVICE).eval()(A), p=2, dim=-1)
    v_pair = F.normalize(video_head_pair(V), p=2, dim=-1)
    a_pair = F.normalize(audio_head_pair(A), p=2, dim=-1)
    v_genre = F.normalize(video_head_genre(V), p=2, dim=-1)
    a_genre = F.normalize(audio_head_genre(A), p=2, dim=-1)
    v_ae_pair = F.normalize(video_head_ae_pair(V), p=2, dim=-1)
    a_ae_pair = F.normalize(audio_head_ae_pair(A_ae_pair), p=2, dim=-1) if A_ae_pair is not None else None
    v_ae_genre = F.normalize(video_head_ae_genre(V), p=2, dim=-1)
    a_ae_genre = F.normalize(audio_head_ae_genre(A_ae_genre), p=2, dim=-1) if A_ae_genre is not None else None

sim_rand = (v_rand @ a_rand.T).cpu()
sim_pair = (v_pair @ a_pair.T).cpu()
sim_genre = (v_genre @ a_genre.T).cpu()
sim_ae_pair = (v_ae_pair @ a_ae_pair.T).cpu() if a_ae_pair is not None else None
sim_ae_genre = (v_ae_genre @ a_ae_genre.T).cpu() if a_ae_genre is not None else None

rel_pair = pair_relevance_matrix(len(test_ds))
rel_label = label_relevance_matrix(labels)

# --- Genre-Breakdown ---
_out_lines: List[str] = []
breakdown_rows: List[dict] = []

def _out(s: str) -> None:
    print(s, flush=True)
    _out_lines.append(s)

def genre_metrics(sim: torch.Tensor, genre: str, rel: torch.Tensor) -> dict:
    idx = [i for i, l in enumerate(labels.tolist()) if l == genre]
    if not idx:
        return {}
    sim_sub = sim[idx, :]
    rel_sub = rel[idx, :]
    return {
        "n": len(idx),
        "mrr": MRR(sim_sub, relevance=rel_sub),
        "r1": recall_at_k(sim_sub, 1, relevance=rel_sub),
        "r5": recall_at_k(sim_sub, 5, relevance=rel_sub),
        "r10": recall_at_k(sim_sub, 10, relevance=rel_sub),
        "mr": mean_rank(sim_sub, relevance=rel_sub),
    }

def _metric_row(
    model_key: str,
    model: str,
    protocol: str,
    protocol_name: str,
    direction: str,
    row_type: str,
    genre: str,
    is_seen: str,
    m: dict,
) -> dict:
    return {
        "model_key": model_key,
        "model": model,
        "protocol": protocol,
        "protocol_name": protocol_name,
        "direction": direction,
        "row_type": row_type,
        "genre": genre,
        "is_seen": is_seen,
        "n": m.get("n", ""),
        "mrr": m["mrr"],
        "recall_at_1": m["r1"],
        "recall_at_5": m["r5"],
        "recall_at_10": m["r10"],
        "mean_rank": m["mr"],
    }


def _avg_metrics(buckets: dict) -> dict:
    def _avg(vals):
        return sum(vals) / len(vals) if vals else 0.0
    return {
        "mrr": _avg(buckets["mrr"]),
        "r1": _avg(buckets["r1"]),
        "r5": _avg(buckets["r5"]),
        "r10": _avg(buckets["r10"]),
        "mr": _avg(buckets["mr"]),
    }


def _print_protocol(
    model_key: str,
    model: str,
    sim_va: torch.Tensor,
    sim_av: torch.Tensor,
    rel: torch.Tensor,
    prot_label: str,
    protocol: str,
    protocol_name: str,
) -> Dict:
    """Gibt eine Protokoll-Tabelle aus und gibt Seen/Unseen/Total-MRR je Richtung zurück."""
    _out(f"\n  [ {prot_label} ]")
    header = f"  {'Genre':<20} {'Seen':>5}  {'N':>5}  {'MRR':>6}  {'R@1':>6}  {'R@5':>6}  {'R@10':>6}  {'MRank':>7}"
    sep = "  " + "-" * 68

    result: Dict = {}
    for direction, sim, direction_key in [
        ("V→A", sim_va, "V2A"),
        ("A→V", sim_av, "A2V"),
    ]:
        _out(f"\n  --- {direction} ---")
        _out(header)
        _out(sep)

        seen_vals = {k: [] for k in ["mrr", "r1", "r5", "r10", "mr"]}
        unseen_vals = {k: [] for k in ["mrr", "r1", "r5", "r10", "mr"]}

        for genre in all_genres:
            m = genre_metrics(sim, genre, rel)
            if not m:
                continue
            if seen_genres is not None:
                is_seen = genre in seen_genres
                tag = "✓" if is_seen else "✗"
                bucket = seen_vals if is_seen else unseen_vals
                for k in ["mrr", "r1", "r5", "r10", "mr"]:
                    bucket[k].append(m[k])
                seen_flag = "1" if is_seen else "0"
            else:
                tag = "—"
                seen_flag = ""
            breakdown_rows.append(
                _metric_row(
                    model_key, model, protocol, protocol_name, direction_key,
                    "genre", genre, seen_flag, m,
                )
            )
            _out(f"  {genre:<20} {tag:>5}  {m['n']:>5}  {m['mrr']:>6.3f}  {m['r1']:>6.3f}  {m['r5']:>6.3f}  {m['r10']:>6.3f}  {m['mr']:>7.1f}")

        _out(sep)
        if seen_genres is not None:
            sv = _avg_metrics(seen_vals)
            uv = _avg_metrics(unseen_vals)
            av = _avg_metrics({k: seen_vals[k] + unseen_vals[k] for k in seen_vals})
            _out(f"  {'Seen Ø (7)':<20} {'':>5}  {'':>5}  {sv['mrr']:>6.3f}  {sv['r1']:>6.3f}  {sv['r5']:>6.3f}  {sv['r10']:>6.3f}  {sv['mr']:>7.1f}")
            _out(f"  {'Unseen Ø (3)':<20} {'':>5}  {'':>5}  {uv['mrr']:>6.3f}  {uv['r1']:>6.3f}  {uv['r5']:>6.3f}  {uv['r10']:>6.3f}  {uv['mr']:>7.1f}")
            _out(f"  {'Total Ø':<20} {'':>5}  {'':>5}  {av['mrr']:>6.3f}  {av['r1']:>6.3f}  {av['r5']:>6.3f}  {av['r10']:>6.3f}  {av['mr']:>7.1f}")
            breakdown_rows.append(
                _metric_row(model_key, model, protocol, protocol_name, direction_key, "seen_mean", "", "", sv)
            )
            breakdown_rows.append(
                _metric_row(model_key, model, protocol, protocol_name, direction_key, "unseen_mean", "", "", uv)
            )
            breakdown_rows.append(
                _metric_row(model_key, model, protocol, protocol_name, direction_key, "overall_mean", "", "", av)
            )
            result[direction_key] = {"seen": sv["mrr"], "unseen": uv["mrr"], "total": av["mrr"]}

    return result

def print_breakdown(model_key: str, title: str, sim_va: torch.Tensor, sim_av: torch.Tensor) -> None:
    _out(f"\n{'='*70}")
    _out(f"  {title}")
    _out(f"{'='*70}")
    _print_protocol(
        model_key, title, sim_va, sim_av, rel_pair,
        "Protokoll A: Pair-basiert (exaktes Paar)", "A", "pair",
    )
    _print_protocol(
        model_key, title, sim_va, sim_av, rel_label,
        "Protokoll B: Label-basiert (gleiches Genre)", "B", "label",
    )

_out(f"Dataset-Run:      {run_name}")
_out(f"Training-Run-Dir: {shared_run_dir or '-'}")
_out(f"TRAIN_GENRES:     {_train_genres_env or 'alle (keine Seen/Unseen-Trennung)'}")
_out(f"Test-Samples:     {len(test_ds)} | Genres: {all_genres}")

print_breakdown("baseline", "Baseline (kein Head)", sim_baseline, sim_baseline.T)
print_breakdown("untrained", "Untrained Heads", sim_rand, sim_rand.T)
print_breakdown("pair", "Pair-based Head", sim_pair, sim_pair.T)
print_breakdown("genre", "Genre-based Head", sim_genre, sim_genre.T)
if sim_ae_pair is not None:
    print_breakdown("audio_encoder_pair", "Audio-Encoder Pair", sim_ae_pair, sim_ae_pair.T)
if sim_ae_genre is not None:
    print_breakdown("audio_encoder_genre", "Audio-Encoder Genre", sim_ae_genre, sim_ae_genre.T)

# --- Ausgabe speichern ---
if os.environ.get("TRAINING_RUN_DIR"):
    out_dir = Path(os.environ["TRAINING_RUN_DIR"])
    out_path = out_dir / "genre_breakdown.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(_out_lines))
    breakdown_csv = out_dir / "results_genre_breakdown.csv"
    save_genre_breakdown_csv(breakdown_rows, breakdown_csv)
    print(f"\nGespeichert: {out_path}", flush=True)
    print(f"Gespeichert: {breakdown_csv}", flush=True)

    if seen_genres is not None:
        try:
            from plot_genre_breakdown import plot_genre_breakdown
            plot_genre_breakdown(out_dir)
        except Exception as e:
            print(f"Plot fehlgeschlagen: {e}", flush=True)
