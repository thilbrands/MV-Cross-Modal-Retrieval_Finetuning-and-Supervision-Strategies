"""
Rank-Verteilung relevanter Kandidaten (Protocol B, E1 vs. E2).

Für jede Query: alle Kandidaten der Gegenmodalität nach Cosine-Similarity ranken;
Rang (1-basiert) jedes gleich-genre-Kandidaten speichern.

Ausgabe:
  rank_positions.csv
  rank_position_histograms.pdf / .png

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

for model_key, _label, v_head, a_head in models:
    sim_va = _projected_sim(v_head, a_head, V, A)
    sim_av = sim_va.T
    for direction, sim in (("V2A", sim_va), ("A2V", sim_av)):
        rows, ranks = _collect_ranks(sim, rel, direction, model_key, labels)
        all_rows.extend(rows)
        hist_data[(model_key, direction)] = ranks
        _summary(model_key, direction, ranks)

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
