"""
Cross-modaler Genre-Centroid-Margin Δ_i und Korrelation mit Protocol-B-Rang.

  Δ_i = s(e_i, μ_{g(i)}) − max_{g ≠ g(i)} s(e_i, μ_g)

  s  = Cosine Similarity
  μ_g = Genre-Zentroid in der *anderen* Modalität (Retrieval ist cross-modal)

Aggregation: pro Query → Statistik pro Genre × Richtung; Overall = Macro-Mittel über Genres.

Ausgabe (unter PLOT_OUTPUT_DIR / TRAINING_RUN_DIR / Dataset-Run):
  centroid_margin.csv              — pro Query
  centroid_margin_summary.csv      — pro Modell × Richtung × Genre (+ ALL macro)
  centroid_margin_boxplot.pdf/.png — Δ_i-Verteilung über Queries (E1–E3b, V→A / A→V)
  centroid_margin_per_genre.pdf/.png — Median-Δ pro Genre (Balken, E1–E3b)
  centroid_margin_rank_corr.pdf/.png — Scatter Δ vs. Rank

Env-Vars:
  DATASET_RUN_NAME, TRAINING_RUN_DIR, PAIR_RUN_DIR, GENRE_RUN_DIR,
  AE_PAIR_RUN_DIR, AE_GENRE_RUN_DIR, PLOT_OUTPUT_DIR

Run:
  TRAINING_RUN_DIR=... DATASET_RUN_NAME=... python3 analyze_soft_assignment.py
  sbatch --export=DATASET_RUN_NAME=...,TRAINING_RUN_DIR=... jobs/analyze_soft_assignment.sh
"""
from __future__ import annotations

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
from scipy.stats import pearsonr, spearmanr

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
import config
from metrics import label_relevance_matrix
from models import (
    load_audio_encoder_heads_genre,
    load_audio_encoder_heads_pair,
    load_projection_heads_genre,
    load_projection_heads_pair,
)

MODEL_ORDER = [
    ("pair", "E1"),
    ("genre", "E2"),
    ("audio_encoder_pair", "E3a"),
    ("audio_encoder_genre", "E3b"),
]
COLORS = {
    "E1": "#1f77b4",
    "E2": "#ff7f0e",
    "E3a": "#2ca02c",
    "E3b": "#d62728",
}
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


def _head_run(env_name: str, training_run_dir: Path | None, filename: str) -> Path | None:
    if os.environ.get(env_name):
        return Path(os.environ[env_name])
    if training_run_dir and (training_run_dir / filename).exists():
        return training_run_dir
    return config.get_latest_training_run_with(filename)


def _project(video_head, audio_head, video: torch.Tensor, audio: torch.Tensor):
    with torch.no_grad():
        v = F.normalize(video_head(video), p=2, dim=-1).cpu().numpy()
        a = F.normalize(audio_head(audio), p=2, dim=-1).cpu().numpy()
    return v, a


def _genre_centroids(emb: np.ndarray, labels: np.ndarray, genres: list[str]) -> np.ndarray:
    """(G, D) L2-normalisierte Genre-Zentroide."""
    cents = []
    for g in genres:
        mu = emb[labels == g].mean(axis=0)
        nrm = np.linalg.norm(mu)
        cents.append(mu / nrm if nrm > 0 else mu)
    return np.stack(cents).astype(np.float64)


def _centroid_margin(
    query_emb: np.ndarray,
    centroids: np.ndarray,
    genre_idx: np.ndarray,
    genres: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Δ_i = s_true − max_{g≠own} s_g

    Returns:
      delta, s_true, s_nearest_other, nearest_other_genre (str array)
    """
    sims = query_emb @ centroids.T  # (N, G)
    n, g = sims.shape
    rows = np.arange(n)
    s_true = sims[rows, genre_idx]

    sims_other = sims.copy()
    sims_other[rows, genre_idx] = -np.inf
    nearest_other_idx = np.argmax(sims_other, axis=1)
    s_nearest_other = sims_other[rows, nearest_other_idx]
    delta = s_true - s_nearest_other
    nearest_other = np.array([genres[j] for j in nearest_other_idx], dtype=object)
    return delta, s_true, s_nearest_other, nearest_other


def _first_relevant_rank(sim: np.ndarray, rel: torch.Tensor) -> np.ndarray:
    """Protocol-B: 1-basierter Rang des ersten gleich-Genre-Kandidaten pro Query."""
    n = sim.shape[0]
    ranks = np.zeros(n, dtype=np.float64)
    for i in range(n):
        order = np.argsort(-sim[i])
        for r, j in enumerate(order, start=1):
            if bool(rel[i, int(j)]):
                ranks[i] = float(r)
                break
    return ranks


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    r, p = spearmanr(x, y)
    return float(r), float(p)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    r, p = pearsonr(x, y)
    return float(r), float(p)


def _stats_row(
    model_key: str,
    model_label: str,
    direction: str,
    genre: str,
    delta: np.ndarray,
    ranks: np.ndarray,
) -> dict:
    sp_r, sp_p = _safe_spearman(delta, ranks)
    pe_r, pe_p = _safe_pearson(delta, ranks)
    return {
        "model_key": model_key,
        "model": model_label,
        "direction": direction,
        "genre": genre,
        "n": int(len(delta)),
        "delta_median": float(np.median(delta)),
        "delta_mean": float(np.mean(delta)),
        "delta_q25": float(np.percentile(delta, 25)),
        "delta_q75": float(np.percentile(delta, 75)),
        "delta_p10": float(np.percentile(delta, 10)),
        "delta_p05": float(np.percentile(delta, 5)),
        "frac_delta_neg": float(np.mean(delta < 0)),
        "rank_median": float(np.median(ranks)),
        "spearman_delta_vs_rank": sp_r,
        "spearman_pvalue": sp_p,
        "pearson_delta_vs_rank": pe_r,
        "pearson_pvalue": pe_p,
    }


def _style_ax(ax) -> None:
    ax.tick_params(labelsize=9)
    ax.grid(which="major", axis="y", linestyle="-", linewidth=0.4, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#666666")


# --- Pfade ---
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

for label, p in [
    ("Pair Heads", pair_path),
    ("Genre Heads", genre_path),
    ("AE Pair", ae_pair_path),
    ("AE Genre", ae_genre_path),
]:
    if p is None:
        print(f"FEHLER: {label} nicht gefunden.", flush=True)
        sys.exit(1)

ae_pair_emb_dir = ae_pair_path / "audio_encoder_pair_test_embeddings"
ae_genre_emb_dir = ae_genre_path / "audio_encoder_genre_test_embeddings"
for label, p in [
    ("AE-Pair Test-Embeddings", ae_pair_emb_dir),
    ("AE-Genre Test-Embeddings", ae_genre_emb_dir),
]:
    if not p.is_dir():
        print(f"FEHLER: {label} nicht gefunden: {p}", flush=True)
        sys.exit(1)

# --- Samples laden ---
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
        if v_path.exists() and a_path.exists() and ae_p.exists() and ae_g.exists():
            samples.append((video_id, label, v_path, a_path, ae_p, ae_g))

if not samples:
    print("FEHLER: Keine Test-Samples mit allen Embeddings gefunden.", flush=True)
    sys.exit(1)

labels = np.array([s[1] for s in samples])
genres = sorted(set(labels.tolist()))
genre_to_idx = {g: i for i, g in enumerate(genres)}
genre_idx = np.array([genre_to_idx[g] for g in labels], dtype=np.int64)
n = len(labels)
rel = label_relevance_matrix(labels)

print(f"Dataset-Run:  {run_name}", flush=True)
print(f"Pair-Run:     {pair_path}", flush=True)
print(f"Genre-Run:    {genre_path}", flush=True)
print(f"AE-Pair-Run:  {ae_pair_path}", flush=True)
print(f"AE-Genre-Run: {ae_genre_path}", flush=True)
print(f"Test-Samples: {n} | Genres: {len(genres)}", flush=True)
print(f"Ausgabe:      {output_dir}", flush=True)

V = torch.tensor(np.stack([np.load(s[2]) for s in samples]), dtype=torch.float32, device=config.DEVICE)
A = torch.tensor(np.stack([np.load(s[3]) for s in samples]), dtype=torch.float32, device=config.DEVICE)
A_ae_pair = torch.tensor(np.stack([np.load(s[4]) for s in samples]), dtype=torch.float32, device=config.DEVICE)
A_ae_genre = torch.tensor(np.stack([np.load(s[5]) for s in samples]), dtype=torch.float32, device=config.DEVICE)

video_head_pair, audio_head_pair = load_projection_heads_pair(pair_path)
video_head_genre, audio_head_genre = load_projection_heads_genre(genre_path)
video_head_ae_pair, audio_head_ae_pair = load_audio_encoder_heads_pair(ae_pair_path)
video_head_ae_genre, audio_head_ae_genre = load_audio_encoder_heads_genre(ae_genre_path)

configs = [
    ("pair", "E1", *_project(video_head_pair, audio_head_pair, V, A)),
    ("genre", "E2", *_project(video_head_genre, audio_head_genre, V, A)),
    ("audio_encoder_pair", "E3a", *_project(video_head_ae_pair, audio_head_ae_pair, V, A_ae_pair)),
    ("audio_encoder_genre", "E3b", *_project(video_head_ae_genre, audio_head_ae_genre, V, A_ae_genre)),
]

# --- Margin + Rang ---
all_rows: list[dict] = []
summary_rows: list[dict] = []
delta_by_model_dir: dict[tuple[str, str], np.ndarray] = {}
rank_by_model_dir: dict[tuple[str, str], np.ndarray] = {}
median_delta_by_model_dir_genre: dict[tuple[str, str], dict[str, float]] = {}

for model_key, model_label, v_emb, a_emb in configs:
    mu_v = _genre_centroids(v_emb, labels, genres)
    mu_a = _genre_centroids(a_emb, labels, genres)
    sim_va = v_emb @ a_emb.T

    for direction, query, cents, sim in (
        ("V2A", v_emb, mu_a, sim_va),
        ("A2V", a_emb, mu_v, sim_va.T),
    ):
        delta, s_true, s_other, nearest_other = _centroid_margin(query, cents, genre_idx, genres)
        ranks = _first_relevant_rank(sim, rel if direction == "V2A" else rel.T)
        delta_by_model_dir[(model_key, direction)] = delta
        rank_by_model_dir[(model_key, direction)] = ranks

        genre_medians: dict[str, float] = {}
        genre_stat_rows: list[dict] = []
        for g in genres:
            mask = labels == g
            row = _stats_row(model_key, model_label, direction, g, delta[mask], ranks[mask])
            genre_stat_rows.append(row)
            summary_rows.append(row)
            genre_medians[g] = row["delta_median"]

        # Macro over genres (gleiche Gewichtung je Genre)
        macro = {
            "model_key": model_key,
            "model": model_label,
            "direction": direction,
            "genre": "ALL",
            "n": n,
            "delta_median": float(np.mean([r["delta_median"] for r in genre_stat_rows])),
            "delta_mean": float(np.mean([r["delta_mean"] for r in genre_stat_rows])),
            "delta_q25": float(np.mean([r["delta_q25"] for r in genre_stat_rows])),
            "delta_q75": float(np.mean([r["delta_q75"] for r in genre_stat_rows])),
            "delta_p10": float(np.mean([r["delta_p10"] for r in genre_stat_rows])),
            "delta_p05": float(np.mean([r["delta_p05"] for r in genre_stat_rows])),
            "frac_delta_neg": float(np.mean([r["frac_delta_neg"] for r in genre_stat_rows])),
            "rank_median": float(np.mean([r["rank_median"] for r in genre_stat_rows])),
            "spearman_delta_vs_rank": float(np.nanmean([r["spearman_delta_vs_rank"] for r in genre_stat_rows])),
            "spearman_pvalue": float("nan"),
            "pearson_delta_vs_rank": float(np.nanmean([r["pearson_delta_vs_rank"] for r in genre_stat_rows])),
            "pearson_pvalue": float("nan"),
        }
        summary_rows.append(macro)
        median_delta_by_model_dir_genre[(model_key, direction)] = genre_medians

        print(
            f"{model_label} {direction}: macro_median(Δ)={macro['delta_median']:.3f}  "
            f"macro_p05={macro['delta_p05']:.3f}  frac(Δ<0)={macro['frac_delta_neg']:.3f}  "
            f"macro_Spearman={macro['spearman_delta_vs_rank']:.3f}",
            flush=True,
        )

        for i in range(n):
            all_rows.append({
                "model_key": model_key,
                "model": model_label,
                "direction": direction,
                "query_idx": i,
                "video_id": samples[i][0],
                "query_genre": labels[i],
                "s_true": float(s_true[i]),
                "s_nearest_other": float(s_other[i]),
                "nearest_other_genre": nearest_other[i],
                "delta": float(delta[i]),
                "rank_first_relevant": float(ranks[i]),
            })

csv_path = output_dir / "centroid_margin.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model_key", "model", "direction", "query_idx", "video_id", "query_genre",
            "s_true", "s_nearest_other", "nearest_other_genre", "delta", "rank_first_relevant",
        ],
    )
    writer.writeheader()
    writer.writerows(all_rows)
print(f"Gespeichert: {csv_path}", flush=True)

summary_path = output_dir / "centroid_margin_summary.csv"
with open(summary_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)
print(f"Gespeichert: {summary_path}", flush=True)

# --- Boxplot Δ_i (Query-Pool, pro Richtung) ---
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=True)
for ax, direction, title in zip(axes, ("V2A", "A2V"), ("V→A", "A→V")):
    data, colors, labels_plot = [], [], []
    for model_key, model_label in MODEL_ORDER:
        data.append(delta_by_model_dir[(model_key, direction)])
        colors.append(COLORS[model_label])
        labels_plot.append(model_label)

    bp = ax.boxplot(
        data,
        labels=labels_plot,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.2},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
        boxprops={"linewidth": 0.8},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    for i, vals in enumerate(data, start=1):
        ax.plot(i, float(np.percentile(vals, 5)), marker="v", color="#333333", markersize=5, zorder=5)

    ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_title(f"Protocol B — {title}", fontsize=10)
    ax.set_ylabel(r"$\Delta_i = s_{\mathrm{true}} - s_{\mathrm{nearest\,other}}$" if direction == "V2A" else "")
    _style_ax(ax)

fig.suptitle("Cross-modal centroid margin", fontsize=11, y=1.02)
fig.tight_layout()
box_base = output_dir / "centroid_margin_boxplot"
fig.savefig(f"{box_base}.pdf", bbox_inches="tight")
fig.savefig(f"{box_base}.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Gespeichert: {box_base}.pdf / .png", flush=True)

# --- Median-Δ pro Genre (Balken) ---
width = 0.18
x = np.arange(len(genres))
offsets = np.linspace(-1.5, 1.5, len(MODEL_ORDER)) * width
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, direction, title in zip(axes, ("V2A", "A2V"), ("V→A", "A→V")):
    for off, (model_key, model_label) in zip(offsets, MODEL_ORDER):
        vals = [median_delta_by_model_dir_genre[(model_key, direction)][g] for g in genres]
        ax.bar(x + off, vals, width, label=model_label, color=COLORS[model_label], edgecolor="white", linewidth=0.4)
    ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([GENRE_SHORT.get(g, g) for g in genres], rotation=45, ha="right", fontsize=8)
    if direction == "V2A":
        ax.set_ylabel(r"Median $\Delta_i$ (per genre)", fontsize=10)
    _style_ax(ax)

handles, leg_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, leg_labels, loc="lower center", ncol=4, fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Centroid margin by genre", fontsize=11, y=1.02)
fig.tight_layout(rect=[0, 0.06, 1, 1])
genre_base = output_dir / "centroid_margin_per_genre"
fig.savefig(f"{genre_base}.pdf", bbox_inches="tight")
fig.savefig(f"{genre_base}.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Gespeichert: {genre_base}.pdf / .png", flush=True)

# --- Scatter Δ vs. Rank ---
fig, axes = plt.subplots(2, 4, figsize=(12, 5.5), sharex=True, sharey=True)
for col, (model_key, model_label) in enumerate(MODEL_ORDER):
    for row, direction in enumerate(("V2A", "A2V")):
        ax = axes[row, col]
        delta = delta_by_model_dir[(model_key, direction)]
        ranks = rank_by_model_dir[(model_key, direction)]
        sp_r, _ = _safe_spearman(delta, ranks)

        ax.scatter(
            delta, ranks,
            s=4, alpha=0.25, color=COLORS[model_label],
            edgecolors="none", rasterized=True,
        )
        ax.axvline(0.0, color="#888888", linestyle="--", linewidth=0.7)
        ax.set_yscale("log")
        if row == 0:
            ax.set_title(f"{model_label}\nSpearman={sp_r:.2f}", fontsize=9)
        else:
            ax.set_title(f"Spearman={sp_r:.2f}", fontsize=9)
        if col == 0:
            ax.set_ylabel(f"{'V→A' if direction == 'V2A' else 'A→V'}\nfirst relevant rank", fontsize=9)
        if row == 1:
            ax.set_xlabel(r"$\Delta_i$", fontsize=9)
        _style_ax(ax)

fig.suptitle(r"$\Delta_i$ vs. Protocol-B first relevant rank", fontsize=11, y=1.01)
fig.tight_layout()
corr_base = output_dir / "centroid_margin_rank_corr"
fig.savefig(f"{corr_base}.pdf", bbox_inches="tight")
fig.savefig(f"{corr_base}.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Gespeichert: {corr_base}.pdf / .png", flush=True)

print("\nFertig.", flush=True)
