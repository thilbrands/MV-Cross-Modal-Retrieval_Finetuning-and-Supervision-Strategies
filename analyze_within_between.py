"""
Within- vs. Between-Genre Cosine-Similarity (projiziert, E1/E2, Audio/Video).

Ausgabe:
  within_between_analysis.csv
  within_between_plot.pdf / .png

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
from models import load_projection_heads_genre, load_projection_heads_pair

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

COLOR_WITHIN = "#2ca02c"
COLOR_BETWEEN = "#d62728"


def _head_run(env_name: str, training_run_dir, filename: str):
    if os.environ.get(env_name):
        return Path(os.environ[env_name])
    if training_run_dir and (Path(training_run_dir) / filename).exists():
        return Path(training_run_dir)
    return config.get_latest_training_run_with(filename)


def _project_modality(head, raw: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return F.normalize(head(raw), p=2, dim=-1).cpu().numpy()


def _within_between(emb: np.ndarray, labels: np.ndarray, genres: list[str]):
    rows = []
    within_vals = []
    between_vals = []
    for g in genres:
        idx_g = np.where(labels == g)[0]
        idx_o = np.where(labels != g)[0]
        if len(idx_g) < 2:
            continue
        eg = emb[idx_g]
        sim_w = eg @ eg.T
        n_g = len(idx_g)
        within = float(sim_w[np.triu_indices(n_g, k=1)].mean())
        between = float((emb[idx_g] @ emb[idx_o].T).mean())
        sep = within - between
        rows.append({"genre": g, "within": within, "between": between, "separation": sep})
        within_vals.append(within)
        between_vals.append(between)
    overall_within = float(np.mean(within_vals))
    overall_between = float(np.mean(between_vals))
    rows.append({
        "genre": "ALL",
        "within": overall_within,
        "between": overall_between,
        "separation": overall_within - overall_between,
    })
    return rows, overall_within, overall_between


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

print(f"Dataset-Run:  {run_name}", flush=True)
print(f"Pair-Run:     {pair_path}", flush=True)
print(f"Genre-Run:    {genre_path}", flush=True)
print(f"Test-Samples: {len(samples)}", flush=True)
print(f"Ausgabe:      {output_dir}", flush=True)

V = torch.tensor(np.stack([np.load(s[1]) for s in samples]), dtype=torch.float32, device=config.DEVICE)
A = torch.tensor(np.stack([np.load(s[2]) for s in samples]), dtype=torch.float32, device=config.DEVICE)
video_head_pair, audio_head_pair = load_projection_heads_pair(pair_path)
video_head_genre, audio_head_genre = load_projection_heads_genre(genre_path)

configs = [
    ("pair", "E1", video_head_pair, audio_head_pair),
    ("genre", "E2", video_head_genre, audio_head_genre),
]

csv_rows = []
panel_stats = {}

for model_key, model_label, v_head, a_head in configs:
    for modality, emb in (
        ("audio", _project_modality(a_head, A)),
        ("video", _project_modality(v_head, V)),
    ):
        stats, ow, ob = _within_between(emb, labels, genres)
        panel_stats[(model_key, modality)] = (ow, ob, ow - ob)
        for row in stats:
            csv_rows.append({
                "model_key": model_key,
                "modality": modality,
                "genre": row["genre"],
                "within": row["within"],
                "between": row["between"],
                "separation": row["separation"],
            })

csv_path = output_dir / "within_between_analysis.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["model_key", "modality", "genre", "within", "between", "separation"]
    )
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"Gespeichert: {csv_path}", flush=True)

print("\n=== Summary (overall) ===", flush=True)
print(f"{'Model':<8} {'Modality':<8} {'within':>8} {'between':>8} {'separation':>11} {'ratio':>8}", flush=True)
for model_key, model_label in (("pair", "E1"), ("genre", "E2")):
    for modality in ("audio", "video"):
        ow, ob, sep = panel_stats[(model_key, modality)]
        ratio = ow / ob if ob != 0 else float("nan")
        print(
            f"{model_label:<8} {modality:<8} {ow:8.3f} {ob:8.3f} {sep:11.3f} {ratio:8.3f}",
            flush=True,
        )

genre_labels = [GENRE_SHORT.get(g, g) for g in genres]
x = np.arange(len(genres))
width = 0.35

fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
panels = [
    (0, 0, "pair", "audio", "E1"),
    (0, 1, "genre", "audio", "E2"),
    (1, 0, "pair", "video", "E1"),
    (1, 1, "genre", "video", "E2"),
]
for row, col, model_key, modality, model_label in panels:
    ax = axes[row, col]
    per_genre = [r for r in csv_rows if r["model_key"] == model_key and r["modality"] == modality and r["genre"] != "ALL"]
    within_bars = [r["within"] for r in per_genre]
    between_bars = [r["between"] for r in per_genre]
    ow, ob, sep = panel_stats[(model_key, modality)]
    ax.bar(x - width / 2, within_bars, width, label="within", color=COLOR_WITHIN)
    ax.bar(x + width / 2, between_bars, width, label="between", color=COLOR_BETWEEN)
    ax.axhline(ow, color=COLOR_WITHIN, linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(ob, color=COLOR_BETWEEN, linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_title(f"{model_label} {modality} (Δ = {sep:.2f})", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(genre_labels, rotation=45, ha="right", fontsize=7)
    if col == 0:
        ax.set_ylabel("Cosine similarity", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=7, loc="upper right")

fig.tight_layout()
out_base = output_dir / "within_between_plot"
fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Gespeichert: {out_base}.pdf", flush=True)
print(f"Gespeichert: {out_base}.png", flush=True)
