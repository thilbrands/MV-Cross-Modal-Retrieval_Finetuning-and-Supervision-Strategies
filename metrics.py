"""
Retrieval-Metriken (MRR, Recall@k, Mean Rank). Relevanz = gleicher Label.
Logik wie in old/09_evaluation.ipynb.
"""
import csv
from pathlib import Path

import numpy as np
import torch


def labels_from_split_csv(
    split_csv: Path,
    split_name: str,
    relevance_column: str = "label",
    embeddings_dir: Path | None = None,
) -> np.ndarray:
    """Labels für Split in derselben Reihenfolge wie PairDataset(split_name, ...).
    Wenn embeddings_dir gesetzt, nur Zeilen mit existierenden video/audio .npy (wie PairDataset)."""
    rows = []
    emb_dir = Path(embeddings_dir) if embeddings_dir else None
    with open(split_csv, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split", "").strip() != split_name:
                continue
            video_id = row.get("video_id", "").strip()
            if not video_id:
                continue
            if emb_dir:
                v = emb_dir / "video" / f"{video_id}.npy"
                a = emb_dir / "audio" / f"{video_id}.npy"
                if not v.exists() or not a.exists():
                    continue
            rows.append(row.get(relevance_column, ""))
    return np.array(rows)


def _first_relevant_rank(sim_row: torch.Tensor, query_label: str, labels: np.ndarray) -> float:
    """Rank (1-based) des ersten relevanten Kandidaten; 0 wenn keiner relevant."""
    order = torch.argsort(sim_row, descending=True)
    for k, j in enumerate(order, start=1):
        if labels[int(j)] == query_label:
            return float(k)
    return 0.0


def MRR(sim: torch.Tensor, labels: np.ndarray) -> float:
    """Mean Reciprocal Rank: sim[i,j] = Query i vs Kandidat j; Relevanz = gleicher Label."""
    n = sim.size(0)
    rr = 0.0
    for i in range(n):
        rank = _first_relevant_rank(sim[i], labels[i], labels)
        rr += (1.0 / rank) if rank > 0 else 0.0
    return rr / n if n else 0.0


def recall_at_k(sim: torch.Tensor, k: int, labels: np.ndarray) -> float:
    """Anteil Queries, bei denen mindestens ein relevanter Kandidat in den Top-k ist."""
    n = sim.size(0)
    hit = 0
    for i in range(n):
        top_k = torch.argsort(sim[i], descending=True)[:k]
        if any(labels[int(j)] == labels[i] for j in top_k.tolist()):
            hit += 1
    return hit / n if n else 0.0


def mean_rank(sim: torch.Tensor, labels: np.ndarray) -> float:
    """Durchschnittlicher Rank (1-based) des ersten relevanten Kandidaten pro Query."""
    n = sim.size(0)
    ranks = []
    for i in range(n):
        r = _first_relevant_rank(sim[i], labels[i], labels)
        if r > 0:
            ranks.append(r)
    return sum(ranks) / len(ranks) if ranks else 0.0
