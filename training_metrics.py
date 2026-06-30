"""CSV-Logging von Trainings- und Evaluationsmetriken (für Plots/Tabellen in der Thesis)."""
import csv
from pathlib import Path

METRICS_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "best_val",
    "is_best",
    "epochs_without_improvement",
]

EVALUATION_FIELDS = [
    "protocol",
    "protocol_name",
    "direction",
    "model_key",
    "model",
    "mrr",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "precision_at_1",
    "precision_at_10",
    "mean_rank",
]

GENRE_BREAKDOWN_FIELDS = [
    "model_key",
    "model",
    "protocol",
    "protocol_name",
    "direction",
    "row_type",
    "genre",
    "is_seen",
    "n",
    "mrr",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "mean_rank",
]


def save_training_metrics_csv(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_evaluation_results_csv(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVALUATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_genre_breakdown_csv(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GENRE_BREAKDOWN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
