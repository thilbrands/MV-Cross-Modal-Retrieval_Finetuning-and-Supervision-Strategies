"""
Rank-Verteilung relevanter Kandidaten (Protocol B).

Pro Query: Rang (1-basiert) des **ersten** same-genre-Kandidaten (Basis für MRR / Mean Rank).
Modelle: Baseline, E1, E2, E3a, E3b — Richtungen V→A und A→V.

Ausgabe (PLOT_OUTPUT_DIR oder TRAINING_RUN_DIR):
  rank_distribution_summary.csv   — Median, P99, % Rang>100, Mean Rank, MRR (+ Sanity-Check)
  rank_first_relevant.csv         — pro Query
  rank_ecdf_e1_e2.pdf / .png      — ECDF E1 vs E2 überlagert
  rank_position_histograms.pdf    — Histogramme aller Modelle (log-x)
  rank_positions.csv              — (legacy) alle same-genre-Ränge pro Query
  per_genre_r10.csv / .pdf

Env: DATASET_RUN_NAME, TRAINING_RUN_DIR, PAIR_RUN_DIR, GENRE_RUN_DIR,
     AE_PAIR_RUN_DIR, AE_GENRE_RUN_DIR, PLOT_OUTPUT_DIR
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
from metrics import MRR, label_relevance_matrix, mean_rank
from models import (
    load_audio_encoder_heads_genre,
    load_audio_encoder_heads_pair,
    load_projection_heads_genre,
    load_projection_heads_pair,
)

REF_RANK = 100
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
COLOR_E1 = "#1f77b4"
COLOR_E2 = "#ff7f0e"


def _head_run(env_name: str, training_run_dir, filename: str):
    if os.environ.get(env_name):
        return Path(os.environ[env_name])
    if training_run_dir and (Path(training_run_dir) / filename).exists():
        return Path(training_run_dir)
    return config.get_latest_training_run_with(filename)


def _sim_baseline(V, A):
    with torch.no_grad():
        vn = F.normalize(V, p=2, dim=-1)
        an = F.normalize(A, p=2, dim=-1)
        return (vn @ an.T).cpu().numpy()


def _sim_projected(video_head, audio_head, V, A):
    with torch.no_grad():
        v = F.normalize(video_head(V), p=2, dim=-1)
        a = F.normalize(audio_head(A), p=2, dim=-1)
        return (v @ a.T).cpu().numpy()


def _direction_sim(sim_va: np.ndarray, direction: str) -> np.ndarray:
    """V2A: Zeilen=Video-Queries, Spalten=Audio-Kandidaten. A2V: transponiert."""
    return sim_va if direction == "V2A" else sim_va.T


def _first_relevant_ranks(sim: np.ndarray, rel: torch.Tensor) -> np.ndarray:
    """Rang des ersten relevanten Kandidaten pro Query (1-basiert); 0 wenn keiner."""
    n = sim.shape[0]
    ranks = np.zeros(n, dtype=np.float64)
    rel_np = rel.cpu().numpy()
    for i in range(n):
        order = np.argsort(-sim[i])
        for k, j in enumerate(order, start=1):
            if rel_np[i, j]:
                ranks[i] = float(k)
                break
    return ranks


def _rank_stats(ranks: np.ndarray) -> dict:
    valid = ranks[ranks > 0]
    if len(valid) == 0:
        return {
            "median_rank": float("nan"),
            "p99_rank": float("nan"),
            "pct_rank_gt_100": float("nan"),
            "mean_rank": float("nan"),
            "mrr": 0.0,
            "n_queries": int(len(ranks)),
        }
    return {
        "median_rank": float(np.median(valid)),
        "p99_rank": float(np.percentile(valid, 99)),
        "pct_rank_gt_100": float(np.mean(valid > REF_RANK) * 100.0),
        "mean_rank": float(np.mean(valid)),
        "mrr": float(np.mean(1.0 / valid)),
        "n_queries": int(len(ranks)),
    }


def _verify_metrics(sim: np.ndarray, rel: torch.Tensor, stats: dict, model_key: str, direction: str) -> None:
    sim_t = torch.from_numpy(sim).float()
    mrr_ref = float(MRR(sim_t, relevance=rel))
    mr_ref = float(mean_rank(sim_t, relevance=rel))
    ok_mrr = abs(mrr_ref - stats["mrr"]) < 1e-5
    ok_mr = abs(mr_ref - stats["mean_rank"]) < 1e-4
    status = "OK" if ok_mrr and ok_mr else "MISMATCH"
    print(
        f"  Sanity [{status}] {model_key} {direction}: "
        f"MRR={stats['mrr']:.6f} (ref {mrr_ref:.6f}) | "
        f"MeanRank={stats['mean_rank']:.4f} (ref {mr_ref:.4f})",
        flush=True,
    )
    if not (ok_mrr and ok_mr):
        raise RuntimeError(f"Sanity-Check fehlgeschlagen für {model_key} {direction}")


def _collect_all_same_genre_ranks(sim, rel, direction, model_key, labels):
    """Legacy: alle same-genre-Ränge (nicht nur erster Treffer)."""
    n = sim.shape[0]
    rows = []
    rel_np = rel.cpu().numpy()
    for query_idx in range(n):
        order = np.argsort(-sim[query_idx])
        rank_of = np.empty(n, dtype=np.int64)
        rank_of[order] = np.arange(1, n + 1)
        for cand_idx in np.where(rel_np[query_idx])[0]:
            rows.append({
                "model_key": model_key,
                "direction": direction,
                "query_idx": query_idx,
                "query_genre": labels[query_idx],
                "rank_position": int(rank_of[cand_idx]),
            })
    return rows


def _recall_at_10_per_query(sim: np.ndarray, rel: torch.Tensor) -> np.ndarray:
    n = sim.shape[0]
    hits = np.zeros(n, dtype=np.float64)
    rel_np = rel.cpu().numpy()
    for i in range(n):
        top10 = np.argsort(-sim[i])[:10]
        if any(rel_np[i, int(j)] for j in top10):
            hits[i] = 1.0
    return hits


def _per_genre_r10(sim, rel, labels, genres):
    per_query = _recall_at_10_per_query(sim, rel)
    by_genre = {}
    for g in genres:
        idx = np.where(labels == g)[0]
        by_genre[g] = float(per_query[idx].mean()) if len(idx) else 0.0
    return by_genre, float(per_query.mean())


def _plot_ecdf_e1_e2(first_ranks: dict, output_dir: Path, n: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, direction, title in zip(axes, ("V2A", "A2V"), ("V→A", "A→V")):
        for model_key, label, color in (("pair", "E1", COLOR_E1), ("genre", "E2", COLOR_E2)):
            ranks = first_ranks[(model_key, direction)]
            valid = np.sort(ranks[ranks > 0])
            if len(valid) == 0:
                continue
            y = np.arange(1, len(valid) + 1) / len(valid)
            ax.plot(valid, y, label=label, color=color, linewidth=1.8)
        ax.set_xscale("log")
        ax.set_xlim(1, n)
        ax.axvline(REF_RANK, color="#888888", linestyle="--", linewidth=1.0, label=f"rank={REF_RANK}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Rank of first relevant candidate", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("ECDF", fontsize=10)
    fig.tight_layout()
    out = output_dir / "rank_ecdf_e1_e2"
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Gespeichert: {out}.pdf", flush=True)


# --- main ---

run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
if not run_name:
    print("FEHLER: Kein Dataset-Run gefunden.", flush=True)
    sys.exit(1)

training_run_dir = Path(os.environ["TRAINING_RUN_DIR"]) if os.environ.get("TRAINING_RUN_DIR") else None
run_dir = config.DATASETS_ROOT / run_name
split_csv = run_dir / "train_val_test_split.csv"
embeddings_dir = run_dir / "embeddings"
output_dir = Path(os.environ["PLOT_OUTPUT_DIR"]) if os.environ.get("PLOT_OUTPUT_DIR") else (
    training_run_dir if training_run_dir else run_dir
)
output_dir.mkdir(parents=True, exist_ok=True)

pair_path = _head_run("PAIR_RUN_DIR", training_run_dir, "projection_heads_pair.pt")
genre_path = _head_run("GENRE_RUN_DIR", training_run_dir, "projection_heads_genre.pt")
ae_pair_path = _head_run("AE_PAIR_RUN_DIR", training_run_dir, "audio_encoder_pair.pt")
ae_genre_path = _head_run("AE_GENRE_RUN_DIR", training_run_dir, "audio_encoder_genre.pt")
if not all([pair_path, genre_path, ae_pair_path, ae_genre_path]):
    print("FEHLER: Mindestens ein Checkpoint-Pfad fehlt.", flush=True)
    sys.exit(1)

ae_pair_emb_dir = ae_pair_path / "audio_encoder_pair_test_embeddings"
ae_genre_emb_dir = ae_genre_path / "audio_encoder_genre_test_embeddings"

samples = []
with open(split_csv, "r", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["split"].strip() != "test":
            continue
        video_id = row["video_id"].strip()
        label = row["label"].strip()
        v_path = embeddings_dir / "video" / f"{video_id}.npy"
        a_path = embeddings_dir / "audio" / f"{video_id}.npy"
        ae_p = ae_pair_emb_dir / f"{video_id}.npy"
        ae_g = ae_genre_emb_dir / f"{video_id}.npy"
        if all(p.exists() for p in (v_path, a_path, ae_p, ae_g)):
            samples.append((label, v_path, a_path, ae_p, ae_g))

if not samples:
    print("FEHLER: Keine Test-Samples gefunden.", flush=True)
    sys.exit(1)

labels = np.array([s[0] for s in samples])
genres = sorted(set(labels))
n = len(labels)
rel_va = label_relevance_matrix(labels)

print(f"Dataset-Run:  {run_name}", flush=True)
print(f"Test-Samples: {n}", flush=True)
print(f"Ausgabe:      {output_dir}", flush=True)

DEVICE = config.DEVICE
V = torch.tensor(np.stack([np.load(s[1]) for s in samples]), dtype=torch.float32, device=DEVICE)
A = torch.tensor(np.stack([np.load(s[2]) for s in samples]), dtype=torch.float32, device=DEVICE)
A_ae_pair = torch.tensor(np.stack([np.load(s[3]) for s in samples]), dtype=torch.float32, device=DEVICE)
A_ae_genre = torch.tensor(np.stack([np.load(s[4]) for s in samples]), dtype=torch.float32, device=DEVICE)

video_head_pair, audio_head_pair = load_projection_heads_pair(pair_path)
video_head_genre, audio_head_genre = load_projection_heads_genre(genre_path)
video_head_ae_pair, audio_head_ae_pair = load_audio_encoder_heads_pair(ae_pair_path)
video_head_ae_genre, audio_head_ae_genre = load_audio_encoder_heads_genre(ae_genre_path)

sim_configs = [
    ("baseline", "Baseline", _sim_baseline(V, A)),
    ("pair", "E1", _sim_projected(video_head_pair, audio_head_pair, V, A)),
    ("genre", "E2", _sim_projected(video_head_genre, audio_head_genre, V, A)),
    ("audio_encoder_pair", "E3a", _sim_projected(video_head_ae_pair, audio_head_ae_pair, V, A_ae_pair)),
    ("audio_encoder_genre", "E3b", _sim_projected(video_head_ae_genre, audio_head_ae_genre, V, A_ae_genre)),
]

summary_rows = []
first_relevant_rows = []
legacy_rows = []
first_ranks_map = {}
hist_first = {}
r10_by_model_dir_genre = {}
r10_overall = {}

print("\n=== Rank-Statistiken (Protocol B, erster relevanter Treffer) ===", flush=True)
for model_key, model_label, sim_va in sim_configs:
    for direction in ("V2A", "A2V"):
        sim = _direction_sim(sim_va, direction)
        rel = rel_va if direction == "V2A" else rel_va.T
        ranks = _first_relevant_ranks(sim, rel)
        stats = _rank_stats(ranks)
        _verify_metrics(sim, rel, stats, model_key, direction)

        summary_rows.append({
            "model_key": model_key,
            "model": model_label,
            "direction": direction,
            **stats,
        })
        print(
            f"{model_label:8} {direction}: median={stats['median_rank']:.1f}  "
            f"p99={stats['p99_rank']:.1f}  >{REF_RANK}={stats['pct_rank_gt_100']:.1f}%  "
            f"MR={stats['mean_rank']:.2f}  MRR={stats['mrr']:.4f}",
            flush=True,
        )

        first_ranks_map[(model_key, direction)] = ranks
        hist_first[(model_key, direction)] = ranks[ranks > 0]
        for qi, r in enumerate(ranks):
            if r > 0:
                first_relevant_rows.append({
                    "model_key": model_key,
                    "model": model_label,
                    "direction": direction,
                    "query_idx": qi,
                    "query_genre": labels[qi],
                    "first_relevant_rank": int(r),
                })
        legacy_rows.extend(_collect_all_same_genre_ranks(sim, rel, direction, model_key, labels))
        by_genre, overall = _per_genre_r10(sim, rel, labels, genres)
        r10_by_model_dir_genre[(model_key, direction)] = by_genre
        r10_overall[(model_key, direction)] = overall

summary_csv = output_dir / "rank_distribution_summary.csv"
with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model_key", "model", "direction", "median_rank", "p99_rank",
            "pct_rank_gt_100", "mean_rank", "mrr", "n_queries",
        ],
    )
    writer.writeheader()
    writer.writerows(summary_rows)
print(f"\nGespeichert: {summary_csv}", flush=True)

first_csv = output_dir / "rank_first_relevant.csv"
with open(first_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["model_key", "model", "direction", "query_idx", "query_genre", "first_relevant_rank"],
    )
    writer.writeheader()
    writer.writerows(first_relevant_rows)
print(f"Gespeichert: {first_csv}", flush=True)

legacy_csv = output_dir / "rank_positions.csv"
with open(legacy_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["model_key", "direction", "query_idx", "query_genre", "rank_position"]
    )
    writer.writeheader()
    writer.writerows(legacy_rows)
print(f"Gespeichert: {legacy_csv}", flush=True)

_plot_ecdf_e1_e2(first_ranks_map, output_dir, n)

# Histogramme: alle Modelle, erster relevanter Rang
model_order = [
    ("baseline", "Baseline"),
    ("pair", "E1"),
    ("genre", "E2"),
    ("audio_encoder_pair", "E3a"),
    ("audio_encoder_genre", "E3b"),
]
fig, axes = plt.subplots(len(model_order), 2, figsize=(10, 2.2 * len(model_order)), sharex=True, sharey=True)
bins = np.unique(np.geomspace(1, n, num=50).astype(int))
if bins[-1] != n:
    bins = np.append(bins, n)
for row, (mk, mlbl) in enumerate(model_order):
    for col, direction in enumerate(("V2A", "A2V")):
        ax = axes[row, col]
        ranks = hist_first[(mk, direction)]
        ax.hist(ranks, bins=bins, density=True, color="#4c72b0", alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(REF_RANK, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_xlim(1, n)
        if col == 0:
            ax.set_ylabel(mlbl, fontsize=9)
        if row == 0:
            ax.set_title("V→A" if direction == "V2A" else "A→V", fontsize=10)
        if row == len(model_order) - 1:
            ax.set_xlabel("First relevant rank", fontsize=9)
fig.tight_layout()
hist_out = output_dir / "rank_position_histograms"
fig.savefig(f"{hist_out}.pdf", bbox_inches="tight")
fig.savefig(f"{hist_out}.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Gespeichert: {hist_out}.pdf", flush=True)

# Per-genre R@10 (E1/E2)
r10_csv_rows = []
for (model_key, direction), by_genre in r10_by_model_dir_genre.items():
    if model_key not in ("pair", "genre"):
        continue
    for g in genres:
        r10_csv_rows.append({
            "model_key": model_key, "direction": direction, "genre": g, "recall_at_10": by_genre[g],
        })
    r10_csv_rows.append({
        "model_key": model_key, "direction": direction, "genre": "ALL",
        "recall_at_10": r10_overall[(model_key, direction)],
    })
r10_csv_path = output_dir / "per_genre_r10.csv"
with open(r10_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["model_key", "direction", "genre", "recall_at_10"])
    writer.writeheader()
    writer.writerows(r10_csv_rows)
print(f"Gespeichert: {r10_csv_path}", flush=True)

e1_va = r10_by_model_dir_genre[("pair", "V2A")]
e2_va = r10_by_model_dir_genre[("genre", "V2A")]
sorted_genres = sorted(genres, key=lambda g: e1_va[g] - e2_va[g], reverse=True)
labels_short = [GENRE_SHORT.get(g, g) for g in sorted_genres]
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
width = 0.36
x = np.arange(len(sorted_genres))
for ax, direction, title in zip(axes, ("V2A", "A2V"), ("V→A", "A→V")):
    e1 = r10_by_model_dir_genre[("pair", direction)]
    e2 = r10_by_model_dir_genre[("genre", direction)]
    ax.bar(x - width / 2, [e1[g] * 100 for g in sorted_genres], width, label="E1", color=COLOR_E1)
    ax.bar(x + width / 2, [e2[g] * 100 for g in sorted_genres], width, label="E2", color=COLOR_E2)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("R@10 (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
fig.tight_layout()
r10_out = output_dir / "per_genre_r10"
fig.savefig(f"{r10_out}.pdf", bbox_inches="tight")
fig.savefig(f"{r10_out}.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Gespeichert: {r10_out}.pdf", flush=True)
