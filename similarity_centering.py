"""Hilfsfunktionen für per-Query-zentrierte Similarity-Analysen."""
import numpy as np
import torch

from metrics import MRR, label_relevance_matrix


def query_matrix(sim_va: np.ndarray, direction: str) -> np.ndarray:
    """
    Orientierung für Query-Zentrierung.

    sim_va[i, j] = Cosine(video_i, audio_j)  (V×A, shape n×n)

    V2A: Zeilen = Video-Queries, Spalten = Audio-Kandidaten → axis=1 über Kandidaten.
    A2V: sim_va.T → Zeilen = Audio-Queries, Spalten = Video-Kandidaten → axis=1.
    """
    if direction == "V2A":
        return sim_va
    if direction == "A2V":
        return sim_va.T
    raise ValueError(f"Unbekannte direction: {direction!r}")


def center_rows(sim: np.ndarray) -> np.ndarray:
    return sim - sim.mean(axis=1, keepdims=True)


def zscore_rows(sim: np.ndarray) -> np.ndarray:
    mu = sim.mean(axis=1, keepdims=True)
    sd = sim.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (sim - mu) / sd


def assert_ranking_unchanged(sim: np.ndarray, sim_transformed: np.ndarray, labels: np.ndarray, name: str) -> None:
    """Zeilenweise monotone Transformation → MRR und Mean-Rank-Reihenfolge unverändert."""
    rel = label_relevance_matrix(labels)
    sim_t = torch.from_numpy(sim).float()
    sim_c_t = torch.from_numpy(sim_transformed).float()
    mrr_a = MRR(sim_t, relevance=rel)
    mrr_b = MRR(sim_c_t, relevance=rel)
    if abs(mrr_a - mrr_b) > 1e-5:
        raise RuntimeError(f"MRR geändert nach Transformation ({name}): {mrr_a} vs {mrr_b}")
    # Spot-check: gleiche Top-1 pro Query
    for i in range(min(sim.shape[0], 50)):
        if int(np.argmax(sim[i])) != int(np.argmax(sim_transformed[i])):
            raise RuntimeError(f"Top-1 geändert bei Query {i} ({name})")


def same_diff_values(sim: np.ndarray, same_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = sim[same_mask]
    neg = sim[~same_mask]
    return pos, neg


def median_gap(pos: np.ndarray, neg: np.ndarray) -> float:
    return float(np.median(pos) - np.median(neg))
