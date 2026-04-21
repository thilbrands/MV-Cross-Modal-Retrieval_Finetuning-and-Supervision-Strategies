"""
Grid-Tuning für Pair- und Genre-Training mit task-aligned Selection:
- pair  -> Protokoll A (Recall@10 avg V->A/A->V)
- genre -> Protokoll B (Recall@10 avg V->A/A->V)

Nutzung:
  TRAINING_TYPE=pair  python -m pipeline.tune_hyperparams
  TRAINING_TYPE=genre python -m pipeline.tune_hyperparams
"""
import csv
import concurrent.futures
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import config


def _run(cmd, env):
    p = subprocess.run(cmd, env=env, cwd=_REPO_ROOT)
    if p.returncode != 0:
        raise RuntimeError(f"Fehlgeschlagen: {' '.join(cmd)}")


def _run_trial(
    trial_idx: int,
    total: int,
    combo,
    tuning_root: Path,
    run_name: str,
    training_type: str,
    hidden_dim: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
    python_exe: str,
    train_script: Path,
    eval_script: Path,
):
    lr, out_dim, temp, head_type = combo
    trial_dir = tuning_root / f"trial_{trial_idx:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[{trial_idx}/{total}] lr={lr} out_dim={out_dim} temp={temp} head_type={head_type}",
        flush=True,
    )

    env = os.environ.copy()
    env.update(
        {
            "DATASET_RUN_NAME": run_name,
            "TRAINING_RUN_DIR": str(trial_dir),
            "HP_LR": str(lr),
            "HP_OUT_DIM": str(out_dim),
            "HP_TEMP": str(temp),
            "HP_HEAD_TYPE": head_type,
            "HP_HIDDEN_DIM": str(hidden_dim),
            "HP_BATCH_SIZE": str(batch_size),
            "HP_MAX_EPOCHS": str(max_epochs),
            "HP_PATIENCE": str(patience),
            "HP_SEED": str(seed),
        }
    )
    _run([python_exe, str(train_script)], env=env)

    metrics_json = trial_dir / "val_metrics.json"
    eval_env = os.environ.copy()
    eval_env.update(
        {
            "DATASET_RUN_NAME": run_name,
            "TRAINING_TYPE": training_type,
            "EVAL_SPLIT": "val",
            "EVAL_MODEL_PATH": str(trial_dir),
            "EVAL_METRICS_JSON": str(metrics_json),
        }
    )
    _run([python_exe, str(eval_script)], env=eval_env)
    with open(metrics_json, "r", encoding="utf-8") as f:
        m = json.load(f)

    score = float(m["selection_score_recall_at_10_avg"])
    row = {
        "trial": trial_idx,
        "training_type": training_type,
        "score_recall_at_10_avg": score,
        "selection_protocol": m["selection_protocol"],
        "lr": lr,
        "out_dim": out_dim,
        "temp": temp,
        "head_type": head_type,
        "hidden_dim": hidden_dim,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "seed": seed,
        "trial_dir": str(trial_dir),
    }
    return row


def _parse_batch_sizes(s: str):
    vals = []
    for x in s.replace(",", " ").split():
        vals.append(int(x))
    if not vals:
        raise ValueError("HP_BATCH_SIZE_SWEEP ist leer.")
    return vals


def _select_top_configs_for_batchsize(rows_sorted):
    selected = rows_sorted[:3]
    selected_trials = {str(r["trial"]) for r in selected}
    best_mlp = next((r for r in rows_sorted if str(r.get("head_type", "")).lower() == "mlp"), None)
    if best_mlp is not None and str(best_mlp["trial"]) not in selected_trials:
        selected.append(best_mlp)
    return selected


def _run_batchsize_sweep(
    base_rows_sorted,
    tuning_root: Path,
    run_name: str,
    training_type: str,
    max_epochs: int,
    patience: int,
    seed: int,
    python_exe: str,
    train_script: Path,
    eval_script: Path,
):
    selected = _select_top_configs_for_batchsize(base_rows_sorted)
    batch_sizes = _parse_batch_sizes(os.environ.get("HP_BATCH_SIZE_SWEEP", "32 64 128 256"))
    out_root = tuning_root / "batchsize_sweep"
    out_root.mkdir(parents=True, exist_ok=True)

    print(
        f"Starte Batch-Size-Sweep: selected_cfgs={len(selected)} batch_sizes={batch_sizes}",
        flush=True,
    )
    for i, cfg in enumerate(selected, start=1):
        print(
            f"  cfg{i}: trial={cfg['trial']} score={cfg['score_recall_at_10_avg']} "
            f"lr={cfg['lr']} out_dim={cfg['out_dim']} temp={cfg['temp']} head={cfg['head_type']}",
            flush=True,
        )

    rows = []
    best_score = float("-inf")
    best_row = None
    run_idx = 0
    total = len(selected) * len(batch_sizes)
    for cfg_idx, cfg in enumerate(selected, start=1):
        for bs in batch_sizes:
            run_idx += 1
            run_dir = out_root / f"cfg{cfg_idx}_trial{cfg['trial']}_bs{bs}"
            run_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[bs {run_idx}/{total}] cfg{cfg_idx} trial={cfg['trial']} bs={bs}",
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
                    "HP_SEED": str(seed),
                }
            )
            _run([python_exe, str(train_script)], env=env)

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
            _run([python_exe, str(eval_script)], env=eval_env)
            with open(metrics_json, "r", encoding="utf-8") as f:
                m = json.load(f)

            score = float(m["selection_score_recall_at_10_avg"])
            row = {
                "training_type": training_type,
                "base_trial": cfg["trial"],
                "base_score_recall_at_10_avg": cfg["score_recall_at_10_avg"],
                "selection_protocol": m["selection_protocol"],
                "score_recall_at_10_avg": score,
                "lr": cfg["lr"],
                "out_dim": cfg["out_dim"],
                "temp": cfg["temp"],
                "head_type": cfg["head_type"],
                "hidden_dim": cfg.get("hidden_dim", "256"),
                "batch_size": bs,
                "max_epochs": max_epochs,
                "patience": patience,
                "seed": seed,
                "run_dir": str(run_dir),
            }
            rows.append(row)
            if score > best_score:
                best_score = score
                best_row = row
            print(f"  -> r10_score={score:.6f} (best={best_score:.6f})", flush=True)

    rows_sorted = sorted(rows, key=lambda x: x["score_recall_at_10_avg"], reverse=True)
    csv_path = out_root / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        w.writeheader()
        w.writerows(rows_sorted)

    summary = {
        "training_type": training_type,
        "dataset_run": run_name,
        "seed": seed,
        "batch_sizes": batch_sizes,
        "selected_base_trials": [r["trial"] for r in selected],
        "total_trials": len(rows_sorted),
        "best_trial": best_row,
        "results_csv": str(csv_path),
    }
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    training_type = os.environ.get("TRAINING_TYPE", "pair").strip().lower()
    if training_type not in {"pair", "genre"}:
        raise ValueError("TRAINING_TYPE muss 'pair' oder 'genre' sein.")

    run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
    if not run_name:
        raise RuntimeError("DATASET_RUN_NAME nicht gesetzt und kein Dataset-Run gefunden.")

    lr_values = [1e-3, 1e-4, 1e-5]
    out_dims = [64, 128, 256]
    temps = [0.07, 0.1, 0.3]
    head_types = ["linear", "mlp"]
    hidden_dim = 256
    batch_size = int(os.environ.get("HP_BATCH_SIZE", "128"))
    max_epochs = int(os.environ.get("HP_MAX_EPOCHS", "20"))
    patience = int(os.environ.get("HP_PATIENCE", "3"))
    seed = int(os.environ.get("HP_SEED", "42"))
    workers = int(os.environ.get("HP_TUNE_WORKERS", "12"))
    if workers < 1:
        workers = 1

    tuning_root = config.TRAINING_RUNS_ROOT / f"tuning_{training_type}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    tuning_root.mkdir(parents=True, exist_ok=True)

    train_script = _REPO_ROOT / "pipeline" / ("pair_based_training.py" if training_type == "pair" else "genre_based_training.py")
    eval_script = _REPO_ROOT / "pipeline" / "eval_single_head.py"

    rows = []
    best_score = float("-inf")
    best_trial = None

    combos = list(itertools.product(lr_values, out_dims, temps, head_types))
    print(
        f"Starte Tuning: type={training_type} | combos={len(combos)} | workers={workers} | dataset_run={run_name}",
        flush=True,
    )
    if workers > 1:
        print("Hinweis: Mehrere Worker parallel. Auf 1 GPU ggf. auf workers=1 setzen.", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = []
        for i, combo in enumerate(combos, start=1):
            fut = ex.submit(
                _run_trial,
                i,
                len(combos),
                combo,
                tuning_root,
                run_name,
                training_type,
                hidden_dim,
                batch_size,
                max_epochs,
                patience,
                seed,
                sys.executable,
                train_script,
                eval_script,
            )
            futures.append(fut)
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            score = row["score_recall_at_10_avg"]
            rows.append(row)
            if score > best_score:
                best_score = score
                best_trial = row
            print(f"  -> trial={row['trial']} r10_score={score:.6f} (best={best_score:.6f})", flush=True)

    rows_sorted = sorted(rows, key=lambda x: x["score_recall_at_10_avg"], reverse=True)
    csv_path = tuning_root / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        w.writeheader()
        w.writerows(rows_sorted)

    summary = {
        "training_type": training_type,
        "dataset_run": run_name,
        "seed": seed,
        "total_trials": len(rows_sorted),
        "best_trial": best_trial,
        "results_csv": str(csv_path),
    }
    with open(tuning_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Stufe 2: Batch-Size-Feinsuche auf Top-3 + bestes MLP
    batchsize_summary = _run_batchsize_sweep(
        rows_sorted,
        tuning_root,
        run_name,
        training_type,
        max_epochs,
        patience,
        seed,
        sys.executable,
        train_script,
        eval_script,
    )
    with open(tuning_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump({**summary, "batchsize_sweep": batchsize_summary}, f, indent=2)

    print("Tuning abgeschlossen.", flush=True)
    print(json.dumps({**summary, "batchsize_sweep": batchsize_summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
