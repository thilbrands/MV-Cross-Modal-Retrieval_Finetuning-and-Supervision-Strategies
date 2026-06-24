import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

RUN_DIR = Path.home() / "Desktop/ba_results/2026-06-24_17-10"
OUTPUT_PATH = RUN_DIR / "training_curves"

COLOR_HEAD = "#2166ac"
COLOR_ENCODER = "#d95f02"


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
