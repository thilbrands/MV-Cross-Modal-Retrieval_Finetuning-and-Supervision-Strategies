"""
Genre-Breakdown-Plot aus results_genre_breakdown.csv (ohne Cluster/GPU).

Run:
  python3 plot_genre_breakdown.py ~/Desktop/ba_results/e4_exploration
  python3 plot_genre_breakdown.py ~/Desktop/ba_results/e4_interpolation
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PLOT_MODEL_ORDER = [
    ("baseline", "Baseline"),
    ("pair", "E1"),
    ("genre", "E2"),
    ("audio_encoder_pair", "E3a"),
    ("audio_encoder_genre", "E3b"),
]
_BAR_GROUPS = [
    ("seen_mean", "Seen avg.", "#2166ac"),
    ("unseen_mean", "Unseen avg.", "#d95f02"),
    ("overall_mean", "Overall avg.", "#55A868"),
]


def _load_mrr(csv_path: Path) -> dict:
    data: dict = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["protocol"] != "B":
                continue
            if row["row_type"] not in {g[0] for g in _BAR_GROUPS}:
                continue
            key = (row["model_key"], row["direction"], row["row_type"])
            data[key] = float(row["mrr"])
    return data


def _style_bar_axis(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(which="major", axis="y", linestyle="-", linewidth=0.4, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#666666")


def _add_figure_legend(fig, axes) -> None:
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=3,
        fontsize=7.5,
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
    )


def plot_genre_breakdown(run_dir: Path) -> None:
    csv_path = run_dir / "results_genre_breakdown.csv"
    if not csv_path.exists():
        print(f"FEHLER: {csv_path} nicht gefunden.", flush=True)
        sys.exit(1)

    mrr = _load_mrr(csv_path)
    plot_models = [(k, label) for k, label in _PLOT_MODEL_ORDER if any(mrr.get((k, d, g)) is not None for d in ("V2A", "A2V") for g, _, _ in _BAR_GROUPS)]
    if not plot_models:
        print(f"FEHLER: Keine Protocol-B-Daten in {csv_path}.", flush=True)
        sys.exit(1)

    model_keys, model_labels = zip(*plot_models)
    x = np.arange(len(model_keys))
    bar_width = 0.22
    offsets = [-bar_width, 0.0, bar_width]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    for direction_key, ax, title in [("V2A", axes[0], "V→A"), ("A2V", axes[1], "A→V")]:
        panel_vals = []
        for i, (row_type, bucket_label, color) in enumerate(_BAR_GROUPS):
            vals = [mrr.get((m, direction_key, row_type), 0.0) for m in model_keys]
            panel_vals.extend(vals)
            ax.bar(x + offsets[i], vals, bar_width, label=bucket_label, color=color, edgecolor="white", linewidth=0.6)

        ax.set_title(f"Protocol B — {title}", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=9)
        ax.set_ylim(0, max(panel_vals) * 1.12 if panel_vals else 1.0)
        _style_bar_axis(ax, "MRR")

    _add_figure_legend(fig, axes)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_base = run_dir / "genre_breakdown_plot"
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Gespeichert: {out_base}.pdf", flush=True)
    print(f"Gespeichert: {out_base}.png", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 plot_genre_breakdown.py <run_dir>", flush=True)
        sys.exit(1)
    plot_genre_breakdown(Path(sys.argv[1]).expanduser())
