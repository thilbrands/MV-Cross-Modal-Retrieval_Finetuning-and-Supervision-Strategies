"""
Training-Curves-Plot (E1/E2/E3a/E3b) aus results_*.csv.

Run:
  TRAINING_RUN_DIR=/path/to/run python3 plot_training_curves.py
  TRAINING_RUN_DIR=~/Desktop/ba_results/genre_retuned/csv python3 plot_training_curves.py
"""
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "configs"))
import config

COLOR_HEAD = "#2166ac"
COLOR_ENCODER = "#d95f02"


def _resolve_run_dir() -> Path:
    if os.environ.get("TRAINING_RUN_DIR"):
        return Path(os.environ["TRAINING_RUN_DIR"]).expanduser()
    for candidate in [
        Path.home() / "Desktop/ba_results/genre_retuned/csv",
        config.TRAINING_RUNS_ROOT / "2026-07-01_13-41",
        Path.home() / "Desktop/ba_results/2026-06-24_17-10",
    ]:
        if (candidate / "results_pair.csv").exists():
            return candidate
    print("FEHLER: TRAINING_RUN_DIR nicht gesetzt und keine results_pair.csv gefunden.", flush=True)
    sys.exit(1)


RUN_DIR = _resolve_run_dir()
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", RUN_DIR / "training_curves"))


def load_metrics(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    epochs = [int(r["epoch"]) for r in rows]
    train = [float(r["train_loss"]) for r in rows]
    val = [float(r["val_loss"]) for r in rows]
    is_best = [int(r["is_best"]) for r in rows]
    best_ep = [e for e, b in zip(epochs, is_best) if b][-1]
    return epochs, train, val, best_ep


def plot_experiment(ax, csv_name, label, color):
    epochs, train, val, best_ep = load_metrics(RUN_DIR / csv_name)
    ax.plot(epochs, train, ls="--", lw=1.0, alpha=0.55, color=color, label=f"{label} train")
    ax.plot(epochs, val, ls="-", lw=1.8, color=color, label=f"{label} val")
    best_idx = epochs.index(best_ep)
    ax.plot(best_ep, val[best_idx], "o", ms=6, color=color, markeredgecolor="white", markeredgewidth=1.2, zorder=5)


def style_axis(ax, ylabel):
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlim(0, 20)
    ax.tick_params(axis="both", labelsize=9)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(which="major", linestyle="-", linewidth=0.4, alpha=0.5)
    ax.grid(which="minor", linestyle=":", linewidth=0.25, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#666666")
    ax.legend(fontsize=7.5, frameon=True, framealpha=0.9, edgecolor="#cccccc")


print(f"RUN_DIR: {RUN_DIR}", flush=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))

plot_experiment(ax1, "results_pair.csv", "E1", COLOR_HEAD)
plot_experiment(ax1, "results_audio_encoder_pair.csv", "E3a", COLOR_ENCODER)
ax1.set_title("pair-based", fontsize=10)
style_axis(ax1, "InfoNCE Loss")

plot_experiment(ax2, "results_genre.csv", "E2", COLOR_HEAD)
plot_experiment(ax2, "results_audio_encoder_genre.csv", "E3b", COLOR_ENCODER)
ax2.set_title("genre-based", fontsize=10)
style_axis(ax2, "SupCon Loss")

fig.tight_layout()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(f"{OUTPUT_PATH}.pdf", bbox_inches="tight")
fig.savefig(f"{OUTPUT_PATH}.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Gespeichert: {OUTPUT_PATH}.pdf", flush=True)
print(f"Gespeichert: {OUTPUT_PATH}.png", flush=True)
