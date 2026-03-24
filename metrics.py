"""
Retrieval-Metriken (MRR, Recall@k, Mean Rank).
Unterstützt:
- Label-basierte Relevanz (gleicher Label)
- Pair-basierte Relevanz (exaktes Gegenstück, Diagonale)
"""
import csv
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch


def labels_from_split_csv(
    split_csv: Path,
    split_name: str,
    relevance_column: str = "label",
    embeddings_dir: Optional[Path] = None,
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


def label_relevance_matrix(labels: Union[np.ndarray, list]) -> torch.Tensor:
    """Relevanzmatrix für Label-basiertes Retrieval (N x N, bool)."""
    arr = np.array(labels)
    if arr.ndim != 1:
        raise ValueError("labels muss 1D sein.")
    if len(arr) == 0:
        return torch.zeros((0, 0), dtype=torch.bool)
    return torch.from_numpy(arr[:, None] == arr[None, :]).bool()


def pair_relevance_matrix(n: int) -> torch.Tensor:
    """Relevanzmatrix für exaktes Pair-Matching (N x N, Diagonale)."""
    if n < 0:
        raise ValueError("n muss >= 0 sein.")
    return torch.eye(n, dtype=torch.bool)


def _first_relevant_rank_mask(sim_row: torch.Tensor, rel_row: torch.Tensor) -> float:
    """Rank (1-based) des ersten relevanten Kandidaten; 0 wenn keiner relevant."""
    order = torch.argsort(sim_row, descending=True)
    for k, j in enumerate(order, start=1):
        if bool(rel_row[int(j)]):
            return float(k)
    return 0.0


def _as_relevance_matrix(
    relevance: Union[np.ndarray, torch.Tensor, list],
    n_queries: int,
    n_candidates: int,
) -> torch.Tensor:
    """Normalisiert relevance auf bool-Tensor (N x M)."""
    if isinstance(relevance, torch.Tensor):
        rel = relevance
    else:
        rel = torch.as_tensor(relevance)
    rel = rel.bool()
    if rel.ndim != 2:
        raise ValueError("relevance muss 2D sein (n_queries x n_candidates).")
    if rel.shape[0] != n_queries or rel.shape[1] != n_candidates:
        raise ValueError(
            f"relevance hat Form {tuple(rel.shape)}, erwartet ({n_queries}, {n_candidates})."
        )
    return rel


def MRR(
    sim: torch.Tensor,
    labels: Optional[np.ndarray] = None,
    relevance: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
) -> float:
    """Mean Reciprocal Rank.

    Entweder `labels` (1D, label-basiert) ODER `relevance` (2D bool-Matrix) angeben.
    """
    n = sim.size(0)
    m = sim.size(1)
    if relevance is None:
        if labels is None:
            raise ValueError("Bitte labels oder relevance angeben.")
        relevance = label_relevance_matrix(labels)
    rel = _as_relevance_matrix(relevance, n, m)

    n = sim.size(0)
    rr = 0.0
    for i in range(n):
        rank = _first_relevant_rank_mask(sim[i], rel[i])
        rr += (1.0 / rank) if rank > 0 else 0.0
    return rr / n if n else 0.0


def recall_at_k(
    sim: torch.Tensor,
    k: int,
    labels: Optional[np.ndarray] = None,
    relevance: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
) -> float:
    """Recall@k: Anteil Queries mit >=1 relevantem Kandidaten in den Top-k."""
    n = sim.size(0)
    m = sim.size(1)
    if relevance is None:
        if labels is None:
            raise ValueError("Bitte labels oder relevance angeben.")
        relevance = label_relevance_matrix(labels)
    rel = _as_relevance_matrix(relevance, n, m)

    n = sim.size(0)
    hit = 0
    for i in range(n):
        top_k = torch.argsort(sim[i], descending=True)[:k]
        if any(bool(rel[i, int(j)]) for j in top_k.tolist()):
            hit += 1
    return hit / n if n else 0.0


def mean_rank(
    sim: torch.Tensor,
    labels: Optional[np.ndarray] = None,
    relevance: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
) -> float:
    """Durchschnittlicher Rank (1-based) des ersten relevanten Kandidaten pro Query."""
    n = sim.size(0)
    m = sim.size(1)
    if relevance is None:
        if labels is None:
            raise ValueError("Bitte labels oder relevance angeben.")
        relevance = label_relevance_matrix(labels)
    rel = _as_relevance_matrix(relevance, n, m)

    n = sim.size(0)
    ranks = []
    for i in range(n):
        r = _first_relevant_rank_mask(sim[i], rel[i])
        if r > 0:
            ranks.append(r)
    return sum(ranks) / len(ranks) if ranks else 0.0
