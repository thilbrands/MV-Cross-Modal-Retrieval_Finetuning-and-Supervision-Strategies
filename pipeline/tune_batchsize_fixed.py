"""
Batchsize-Tuning für feste Top-Konfigurationen aus einer bestehenden results.csv.

Auswahl der Basiskonfigurationen:
- Beste Konfiguration insgesamt (Top-1)
- Beste MLP-Konfiguration (falls vorhanden und nicht identisch)

Dann Sweep über BATCH_SIZES (default: 32,64,128,256).
Selection bleibt task-aligned über eval_single_head.py:
- pair  -> Protokoll A (Recall@10 avg)
- genre -> Protokoll B (Recall@10 avg)
"""
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import config


def _run(cmd, env):
    p = subprocess.run(cmd, env=env, cwd=_REPO_ROOT)
    if p.returncode != 0:
        raise RuntimeError(f"Fehlgeschlagen: {' '.join(cmd)}")


def _load_and_select_configs(results_csv: Path) -> List[Dict[str, str]]:
    rows = list(csv.DictReader(open(results_csv, "r", encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"Leere CSV: {results_csv}")
    rows.sort(key=lambda r: float(r["score_recall_at_10_avg"]), reverse=True)

    selected = [rows[0]]  # best overall
    best_mlp = next((r for r in rows if r.get("head_type") == "mlp"), None)
    if best_mlp is not None and best_mlp.get("trial") != rows[0].get("trial"):
        selected.append(best_mlp)
    return selected


def _parse_batch_sizes(s: str) -> List[int]:
    out = []
    for x in s.replace(",", " ").split():
        out.append(int(x))
    if not out:
        raise ValueError("BATCH_SIZES ist leer.")
    return out


def main():
    training_type = os.environ.get("TRAINING_TYPE", "").strip().lower()
    if training_type not in {"pair", "genre"}:
        raise ValueError("TRAINING_TYPE muss 'pair' oder 'genre' sein.")

    results_csv_env = os.environ.get("RESULTS_CSV", "").strip()
    if not results_csv_env:
        raise ValueError("RESULTS_CSV ist erforderlich.")
    results_csv = Path(results_csv_env)
    if not results_csv.exists():
        raise FileNotFoundError(f"RESULTS_CSV nicht gefunden: {results_csv}")

    run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
    if not run_name:
        raise RuntimeError("DATASET_RUN_NAME nicht gesetzt und kein Dataset-Run gefunden.")

    batch_sizes = _parse_batch_sizes(os.environ.get("BATCH_SIZES", "32 64 128 256"))
    max_epochs = int(os.environ.get("HP_MAX_EPOCHS", "20"))
    patience = int(os.environ.get("HP_PATIENCE", "3"))

    selected_cfgs = _load_and_select_configs(results_csv)
    train_script = _REPO_ROOT / "pipeline" / ("pair_based_training.py" if training_type == "pair" else "genre_based_training.py")
    eval_script = _REPO_ROOT / "pipeline" / "eval_single_head.py"

    out_root = config.TRAINING_RUNS_ROOT / f"batchsize_tuning_{training_type}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Starte Batchsize-Tuning: type={training_type} dataset_run={run_name}", flush=True)
    print(f"results_csv={results_csv}", flush=True)
    print(f"batch_sizes={batch_sizes} max_epochs={max_epochs} patience={patience}", flush=True)
    print("Ausgewählte Basiskonfigurationen:", flush=True)
    for i, cfg in enumerate(selected_cfgs, start=1):
        print(
            f"  cfg{i}: trial={cfg['trial']} score={cfg['score_recall_at_10_avg']} "
            f"lr={cfg['lr']} out_dim={cfg['out_dim']} temp={cfg['temp']} head={cfg['head_type']}",
            flush=True,
        )

    rows = []
    best_score = float("-inf")
    best_row = None

    for cfg_idx, cfg in enumerate(selected_cfgs, start=1):
        for bs in batch_sizes:
            run_dir = out_root / f"cfg{cfg_idx}_trial{cfg['trial']}_bs{bs}"
            run_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[cfg{cfg_idx} trial={cfg['trial']} bs={bs}] lr={cfg['lr']} out_dim={cfg['out_dim']} "
                f"temp={cfg['temp']} head={cfg['head_type']}",
                flush=True,
            )

            env = os.environ.copy()
            env.update(
                {
                    "DATASET_RUN_NAME": run_name,
                    "TRAINING_RUN_DIR": str(run_dir),
                    "HP_LR": str(cfg["lr"]),
                    "HP_OUT_DIM": str(cfg["out_dim"]),
                    "HP_TEMP": str(cfg["temp"]),
                    "HP_HEAD_TYPE": str(cfg["head_type"]),
                    "HP_HIDDEN_DIM": str(cfg.get("hidden_dim", "256")),
                    "HP_BATCH_SIZE": str(bs),
                    "HP_MAX_EPOCHS": str(max_epochs),
                    "HP_PATIENCE": str(patience),
                }
            )
            _run([sys.executable, str(train_script)], env=env)

            metrics_json = run_dir / "val_metrics.json"
            eval_env = os.environ.copy()
            eval_env.update(
                {
                    "DATASET_RUN_NAME": run_name,
                    "TRAINING_TYPE": training_type,
                    "EVAL_SPLIT": "val",
                    "EVAL_MODEL_PATH": str(run_dir),
                    "EVAL_METRICS_JSON": str(metrics_json),
                }
            )
            _run([sys.executable, str(eval_script)], env=eval_env)
            with open(metrics_json, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            score = float(metrics["selection_score_recall_at_10_avg"])
            row = {
                "training_type": training_type,
                "base_trial": cfg["trial"],
                "base_score_recall_at_10_avg": cfg["score_recall_at_10_avg"],
                "selection_protocol": metrics["selection_protocol"],
                "score_recall_at_10_avg": score,
                "lr": cfg["lr"],
                "out_dim": cfg["out_dim"],
                "temp": cfg["temp"],
                "head_type": cfg["head_type"],
                "hidden_dim": cfg.get("hidden_dim", "256"),
                "batch_size": bs,
                "max_epochs": max_epochs,
                "patience": patience,
                "run_dir": str(run_dir),
            }
            rows.append(row)
            if score > best_score:
                best_score = score
                best_row = row
            print(f"  -> r10_score={score:.6f} (best={best_score:.6f})", flush=True)

    rows.sort(key=lambda r: float(r["score_recall_at_10_avg"]), reverse=True)
    results_csv_out = out_root / "results.csv"
    with open(results_csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "training_type": training_type,
        "dataset_run": run_name,
        "source_results_csv": str(results_csv),
        "batch_sizes": batch_sizes,
        "total_runs": len(rows),
        "best_run": best_row,
        "results_csv": str(results_csv_out),
    }
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Batchsize-Tuning abgeschlossen.", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
