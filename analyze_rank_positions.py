"""
Rank-Verteilung relevanter Kandidaten (Protocol B, E1 vs. E2).

Für jede Query: alle Kandidaten der Gegenmodalität nach Cosine-Similarity ranken;
Rang (1-basiert) jedes gleich-genre-Kandidaten speichern.

Ausgabe:
  rank_positions.csv
  rank_position_histograms.pdf / .png
  per_genre_r10.csv
  per_genre_r10.pdf / .png

Env-Vars:
  DATASET_RUN_NAME, TRAINING_RUN_DIR, PAIR_RUN_DIR, GENRE_RUN_DIR, PLOT_OUTPUT_DIR
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
from metrics import label_relevance_matrix
from models import load_projection_heads_genre, load_projection_heads_pair

REF_RANK = 367
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


def _projected_sim(video_head, audio_head, video, audio):
    with torch.no_grad():
        v = F.normalize(video_head(video), p=2, dim=-1)
        a = F.normalize(audio_head(audio), p=2, dim=-1)
        return (v @ a.T).cpu().numpy()


def _collect_ranks(sim: np.ndarray, rel: torch.Tensor, direction: str, model_key: str, labels: np.ndarray):
    n = sim.shape[0]
    rows = []
    all_ranks = []
    for query_idx in range(n):
        order = np.argsort(-sim[query_idx])
        rank_of = np.empty(n, dtype=np.int64)
        rank_of[order] = np.arange(1, n + 1)
        rel_idx = torch.where(rel[query_idx])[0].cpu().numpy()
        query_genre = labels[query_idx]
        for cand_idx in rel_idx:
            r = int(rank_of[cand_idx])
            rows.append({
                "model_key": model_key,
                "direction": direction,
                "query_idx": query_idx,
                "query_genre": query_genre,
                "rank_position": r,
            })
            all_ranks.append(r)
    return rows, np.array(all_ranks, dtype=np.int64)


def _summary(model_key: str, direction: str, ranks: np.ndarray) -> None:
    med = float(np.median(ranks))
    frac_top_ref = float(np.mean(ranks <= REF_RANK))
    frac_top10 = float(np.mean(ranks <= 10))
    print(
        f"{model_key} {direction}: median_rank={med:.1f}  "
        f"frac_top_{REF_RANK}={frac_top_ref:.3f}  frac_top_10={frac_top10:.3f}  n={len(ranks)}",
        flush=True,
    )


def _recall_at_10_per_query(sim: np.ndarray, rel: torch.Tensor) -> np.ndarray:
    """Pro Query: 1 wenn >=1 relevanter Kandidat in Top-10, sonst 0."""
    n = sim.shape[0]
    hits = np.zeros(n, dtype=np.float64)
    for i in range(n):
        top10 = np.argsort(-sim[i])[:10]
        if any(bool(rel[i, int(j)]) for j in top10):
            hits[i] = 1.0
    return hits


def _per_genre_r10(sim: np.ndarray, rel: torch.Tensor, labels: np.ndarray, genres: list[str]):
    per_query = _recall_at_10_per_query(sim, rel)
    overall = float(per_query.mean())
    by_genre = {}
    for g in genres:
        idx = np.where(labels == g)[0]
        by_genre[g] = float(per_query[idx].mean()) if len(idx) else 0.0
    return by_genre, overall


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
if not pair_path or not genre_path:
    print("FEHLER: Head-Run nicht gefunden.", flush=True)
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
            samples.append((label, v_path, a_path))

if not samples:
    print("FEHLER: Keine Test-Samples gefunden.", flush=True)
    sys.exit(1)

labels = np.array([s[0] for s in samples])
genres = sorted(set(labels))
n = len(labels)
rel = label_relevance_matrix(labels)

print(f"Dataset-Run:  {run_name}", flush=True)
print(f"Pair-Run:     {pair_path}", flush=True)
print(f"Genre-Run:    {genre_path}", flush=True)
print(f"Test-Samples: {n}", flush=True)
print(f"Ausgabe:      {output_dir}", flush=True)
print(f"Referenz-Rang: {REF_RANK}", flush=True)

V = torch.tensor(np.stack([np.load(s[1]) for s in samples]), dtype=torch.float32, device=config.DEVICE)
A = torch.tensor(np.stack([np.load(s[2]) for s in samples]), dtype=torch.float32, device=config.DEVICE)
video_head_pair, audio_head_pair = load_projection_heads_pair(pair_path)
video_head_genre, audio_head_genre = load_projection_heads_genre(genre_path)

models = [
    ("pair", "E1", video_head_pair, audio_head_pair),
    ("genre", "E2", video_head_genre, audio_head_genre),
]

all_rows = []
hist_data = {}
r10_by_model_dir_genre = {}
r10_overall = {}

for model_key, _label, v_head, a_head in models:
    sim_va = _projected_sim(v_head, a_head, V, A)
    sim_av = sim_va.T
    for direction, sim in (("V2A", sim_va), ("A2V", sim_av)):
        rows, ranks = _collect_ranks(sim, rel, direction, model_key, labels)
        all_rows.extend(rows)
        hist_data[(model_key, direction)] = ranks
        _summary(model_key, direction, ranks)
        by_genre, overall = _per_genre_r10(sim, rel, labels, genres)
        r10_by_model_dir_genre[(model_key, direction)] = by_genre
        r10_overall[(model_key, direction)] = overall

csv_path = output_dir / "rank_positions.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["model_key", "direction", "query_idx", "query_genre", "rank_position"]
    )
    writer.writeheader()
    writer.writerows(all_rows)
print(f"Gespeichert: {csv_path}", flush=True)

bins = np.unique(np.geomspace(1, n, num=50).astype(int))
if bins[-1] != n:
    bins = np.append(bins, n)

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
panel_order = [
    (0, 0, "pair", "V2A", "E1 (V→A)"),
    (0, 1, "genre", "V2A", "E2 (V→A)"),
    (1, 0, "pair", "A2V", "E1 (A→V)"),
    (1, 1, "genre", "A2V", "E2 (A→V)"),
]
for row, col, model_key, direction, title in panel_order:
    ax = axes[row, col]
    ranks = hist_data[(model_key, direction)]
    ax.hist(ranks, bins=bins, density=True, color="#4c72b0", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(REF_RANK, color="#333333", linestyle="--", linewidth=1.2, label=f"rank={REF_RANK}")
    ax.set_xscale("log")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(1, n)
    if col == 0:
        ax.set_ylabel("Density", fontsize=10)
    if row == 1:
        ax.set_xlabel("Rank position", fontsize=10)
    ax.tick_params(labelsize=8)

fig.tight_layout()
out_base = output_dir / "rank_position_histograms"
fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Gespeichert: {out_base}.pdf", flush=True)
print(f"Gespeichert: {out_base}.png", flush=True)

r10_csv_rows = []
for (model_key, direction), by_genre in r10_by_model_dir_genre.items():
    for g in genres:
        r10_csv_rows.append({
            "model_key": model_key,
            "direction": direction,
            "genre": g,
            "recall_at_10": by_genre[g],
        })
    r10_csv_rows.append({
        "model_key": model_key,
        "direction": direction,
        "genre": "ALL",
        "recall_at_10": r10_overall[(model_key, direction)],
    })

r10_csv_path = output_dir / "per_genre_r10.csv"
with open(r10_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["model_key", "direction", "genre", "recall_at_10"])
    writer.writeheader()
    writer.writerows(r10_csv_rows)
print(f"Gespeichert: {r10_csv_path}", flush=True)

print("\n=== Per-genre R@10 (Protocol B) ===", flush=True)
for direction in ("V2A", "A2V"):
    print(f"\n{direction}:", flush=True)
    e1 = r10_by_model_dir_genre[("pair", direction)]
    e2 = r10_by_model_dir_genre[("genre", direction)]
    for g in genres:
        print(f"  {g}: E1={e1[g]*100:.1f}%  E2={e2[g]*100:.1f}%  Δ={(e1[g]-e2[g])*100:+.1f}pp", flush=True)
    print(
        f"  ALL: E1={r10_overall[('pair', direction)]*100:.1f}%  "
        f"E2={r10_overall[('genre', direction)]*100:.1f}%",
        flush=True,
    )

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
width = 0.36

for ax, direction, title in zip(
    axes,
    ("V2A", "A2V"),
    ("V→A", "A→V"),
):
    e1 = r10_by_model_dir_genre[("pair", direction)]
    e2 = r10_by_model_dir_genre[("genre", direction)]
    sorted_genres = sorted(genres, key=lambda g: e1[g] - e2[g], reverse=True)
    x = np.arange(len(sorted_genres))
    e1_vals = [e1[g] * 100 for g in sorted_genres]
    e2_vals = [e2[g] * 100 for g in sorted_genres]
    labels_short = [GENRE_SHORT.get(g, g) for g in sorted_genres]

    ax.bar(x - width / 2, e1_vals, width, label="E1", color=COLOR_E1)
    ax.bar(x + width / 2, e2_vals, width, label="E2", color=COLOR_E2)
    ax.axhline(r10_overall[("pair", direction)] * 100, color=COLOR_E1, linestyle="--", linewidth=1.2, alpha=0.85)
    ax.axhline(r10_overall[("genre", direction)] * 100, color=COLOR_E2, linestyle="--", linewidth=1.2, alpha=0.85)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("R@10 (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=9)

fig.tight_layout()
r10_out = output_dir / "per_genre_r10"
fig.savefig(f"{r10_out}.pdf", bbox_inches="tight")
fig.savefig(f"{r10_out}.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Gespeichert: {r10_out}.pdf", flush=True)
print(f"Gespeichert: {r10_out}.png", flush=True)
