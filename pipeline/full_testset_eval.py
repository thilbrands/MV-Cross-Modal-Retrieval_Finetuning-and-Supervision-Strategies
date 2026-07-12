#!/usr/bin/env python3
"""
Full-Testset-Evaluation: Protocol A + B auf dem gesamten Test-Split.

Gleiche Metriken/Modelle wie evaluation.py, aber separates Skript und Output:
  - results_full_testset_eval.csv
  - meta_full_testset_eval.json
  - full_testset_eval_output.txt

evaluation.py bleibt unverändert.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
import config
from dataset import PairDataset
from models import (
    ProjectionHead,
    load_audio_encoder_heads_genre,
    load_audio_encoder_heads_pair,
    load_projection_heads_genre,
    load_projection_heads_pair,
)
from metrics import (
    MRR,
    label_relevance_matrix,
    labels_from_split_csv,
    mean_rank,
    pair_relevance_matrix,
    precision_at_k,
    recall_at_k,
)
from training_metrics import save_evaluation_results_csv

run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
if not run_name:
    print("FEHLER: DATASET_RUN_NAME nicht gesetzt.", flush=True)
    sys.exit(1)

run_dir = config.DATASETS_ROOT / run_name
EMBEDDINGS_DIR = run_dir / "embeddings"
SPLIT_CSV = Path(os.environ.get("SPLIT_CSV", run_dir / "train_val_test_split.csv"))
DEVICE = config.DEVICE

if os.environ.get("TRAINING_RUN_DIR"):
    shared_run_dir = Path(os.environ["TRAINING_RUN_DIR"])
    pair_path = shared_run_dir if (shared_run_dir / "projection_heads_pair.pt").exists() else None
    genre_path = shared_run_dir if (shared_run_dir / "projection_heads_genre.pt").exists() else None
    pair_run_dir = shared_run_dir if pair_path else None
    genre_run_dir = shared_run_dir if genre_path else None
else:
    pair_run_dir = config.get_latest_training_run_with("projection_heads_pair.pt")
    genre_run_dir = config.get_latest_training_run_with("projection_heads_genre.pt")
    pair_path = pair_run_dir if pair_run_dir else None
    genre_path = genre_run_dir if genre_run_dir else None

ae_pair_path = (
    Path(os.environ["AE_PAIR_RUN_DIR"])
    if os.environ.get("AE_PAIR_RUN_DIR")
    else config.get_latest_training_run_with("audio_encoder_pair.pt")
)
ae_genre_path = (
    Path(os.environ["AE_GENRE_RUN_DIR"])
    if os.environ.get("AE_GENRE_RUN_DIR")
    else config.get_latest_training_run_with("audio_encoder_genre.pt")
)


def _meta_commit(run_dir: Optional[Path], meta_name: str = "meta.json") -> str:
    if run_dir is None:
        return ""
    for name in (meta_name, "meta.json"):
        p = run_dir / name
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f).get("git_commit", "") or ""
            except Exception:
                pass
    return ""


def _eval_output_dir() -> Path:
    if os.environ.get("TRAINING_RUN_DIR"):
        return Path(os.environ["TRAINING_RUN_DIR"])
    if os.environ.get("EVAL_OUTPUT_DIR"):
        return Path(os.environ["EVAL_OUTPUT_DIR"])
    if pair_run_dir:
        return pair_run_dir
    return config.TRAINING_RUNS_ROOT / f"full_testset_eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


def _load_ae_embeddings(run_path, subdir, samples):
    emb_dir = run_path / subdir
    embs = [
        torch.tensor(np.load(emb_dir / f"{video_id}.npy"), dtype=torch.float32)
        for video_id, *_ in samples
    ]
    return torch.stack(embs).to(DEVICE)


_out_lines: List[str] = []


def _out(msg: str) -> None:
    print(msg, flush=True)
    _out_lines.append(msg)


_out(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
_out(f"Git-Commit (Eval): {config.get_git_commit()}")
_out(f"Dataset-Run: {run_name}")
_out(f"Split-CSV: {SPLIT_CSV}")
_out(f"Pair-based Head: {pair_path or config.PROJECTION_HEADS_PATH}")
_out(f"Genre-based Head: {genre_path or config.PROJECTION_HEADS_GENRE_PATH}")
_out(f"Audio-Encoder Pair: {ae_pair_path or '-'}")
_out(f"Audio-Encoder Genre: {ae_genre_path or '-'}")

test_ds = PairDataset("test", SPLIT_CSV, EMBEDDINGS_DIR)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
labels = labels_from_split_csv(SPLIT_CSV, "test", relevance_column="label", embeddings_dir=EMBEDDINGS_DIR)
_out(f"Test-Samples: {len(test_ds)}")

video_head_pair, audio_head_pair = load_projection_heads_pair(pair_path)
video_head_genre, audio_head_genre = load_projection_heads_genre(genre_path)
video_head_ae_pair, audio_head_ae_pair = load_audio_encoder_heads_pair(ae_pair_path)
video_head_ae_genre, audio_head_ae_genre = load_audio_encoder_heads_genre(ae_genre_path)

V_list, A_list = [], []
for v, a in test_loader:
    V_list.append(v)
    A_list.append(a)
V = torch.cat(V_list, dim=0).to(DEVICE)
A = torch.cat(A_list, dim=0).to(DEVICE)
Vn = F.normalize(V, p=2, dim=-1)
An = F.normalize(A, p=2, dim=-1)

A_ae_pair = _load_ae_embeddings(ae_pair_path, "audio_encoder_pair_test_embeddings", test_ds.samples)
A_ae_genre = _load_ae_embeddings(ae_genre_path, "audio_encoder_genre_test_embeddings", test_ds.samples)

sim_baseline = (Vn @ An.T).cpu()
with torch.no_grad():
    video_head_rand = ProjectionHead().to(DEVICE).eval()
    audio_head_rand = ProjectionHead().to(DEVICE).eval()
    v_rand = F.normalize(video_head_rand(V), p=2, dim=-1)
    a_rand = F.normalize(audio_head_rand(A), p=2, dim=-1)

    v_pair = F.normalize(video_head_pair(V), p=2, dim=-1)
    a_pair = F.normalize(audio_head_pair(A), p=2, dim=-1)
    v_genre = F.normalize(video_head_genre(V), p=2, dim=-1)
    a_genre = F.normalize(audio_head_genre(A), p=2, dim=-1)

    v_ae_pair = F.normalize(video_head_ae_pair(V), p=2, dim=-1)
    a_ae_pair = F.normalize(audio_head_ae_pair(A_ae_pair), p=2, dim=-1)

    v_ae_genre = F.normalize(video_head_ae_genre(V), p=2, dim=-1)
    a_ae_genre = F.normalize(audio_head_ae_genre(A_ae_genre), p=2, dim=-1)

sim_rand = (v_rand @ a_rand.T).cpu()
sim_pair = (v_pair @ a_pair.T).cpu()
sim_genre = (v_genre @ a_genre.T).cpu()
sim_ae_pair = (v_ae_pair @ a_ae_pair.T).cpu()
sim_ae_genre = (v_ae_genre @ a_ae_genre.T).cpu()

rel_pair = pair_relevance_matrix(sim_baseline.size(0))
rel_label = label_relevance_matrix(labels)

EVAL_MODELS = [
    ("baseline", "Baseline", sim_baseline),
    ("untrained", "Untrained heads", sim_rand),
    ("pair", "Pair-based", sim_pair),
    ("genre", "Genre-based", sim_genre),
    ("audio_encoder_pair", "Audio-Encoder Pair", sim_ae_pair),
    ("audio_encoder_genre", "Audio-Encoder Genre", sim_ae_genre),
]
EVAL_PROTOCOLS = [
    ("A", "pair", "Pair-basierte Relevanz (exaktes Video-Audio-Paar)", rel_pair),
    ("B", "label", "Label-basierte Relevanz (gleiches Genre)", rel_label),
]

results_rows: List[dict] = []


def _compute_metrics(sim, relevance, with_precision: bool = False) -> dict:
    metrics = {
        "mrr": float(MRR(sim, relevance=relevance)),
        "recall_at_1": float(recall_at_k(sim, 1, relevance=relevance)),
        "recall_at_5": float(recall_at_k(sim, 5, relevance=relevance)),
        "recall_at_10": float(recall_at_k(sim, 10, relevance=relevance)),
        "mean_rank": float(mean_rank(sim, relevance=relevance)),
    }
    if with_precision:
        metrics["precision_at_1"] = float(precision_at_k(sim, 1, relevance=relevance))
        metrics["precision_at_10"] = float(precision_at_k(sim, 10, relevance=relevance))
    return metrics


def _record_metrics(protocol, protocol_name, direction, model_key, model, sim, relevance) -> None:
    metrics = _compute_metrics(sim, relevance, with_precision=(protocol == "B"))
    results_rows.append(
        {
            "protocol": protocol,
            "protocol_name": protocol_name,
            "direction": direction,
            "model_key": model_key,
            "model": model,
            **metrics,
        }
    )
    line = (
        f"  {model} | MRR={metrics['mrr']:.4f} R@1={metrics['recall_at_1']:.4f} "
        f"R@10={metrics['recall_at_10']:.4f} MR={metrics['mean_rank']:.1f}"
    )
    if "precision_at_1" in metrics:
        line += f" P@1={metrics['precision_at_1']:.4f} P@10={metrics['precision_at_10']:.4f}"
    _out(line)


for protocol_id, protocol_key, protocol_title, relevance in EVAL_PROTOCOLS:
    _out(f"=== Protokoll {protocol_id}: {protocol_title} ===")
    for direction, sim_getter, rel_getter in (
        ("V2A", lambda s: s, lambda r: r),
        ("A2V", lambda s: s.T, lambda r: r.T),
    ):
        dir_label = "V→A" if direction == "V2A" else "A→V"
        _out(f"--- {dir_label} ---")
        for model_key, model_name, sim in EVAL_MODELS:
            _record_metrics(
                protocol_id,
                protocol_key,
                direction,
                model_key,
                model_name,
                sim_getter(sim),
                rel_getter(relevance),
            )

output_dir = _eval_output_dir()
output_dir.mkdir(parents=True, exist_ok=True)
results_csv = output_dir / "results_full_testset_eval.csv"
save_evaluation_results_csv(results_rows, results_csv)

meta = {
    "timestamp": datetime.now().isoformat(),
    "git_commit_eval": config.get_git_commit(),
    "dataset_run": run_name,
    "split_csv": str(SPLIT_CSV),
    "n_test": len(test_ds),
    "pair_head_path": str(pair_path or config.PROJECTION_HEADS_PATH),
    "genre_head_path": str(genre_path or config.PROJECTION_HEADS_GENRE_PATH),
    "audio_encoder_pair_path": str(ae_pair_path or ""),
    "audio_encoder_genre_path": str(ae_genre_path or ""),
    "pair_train_commit": _meta_commit(pair_run_dir, "meta_pair.json") or None,
    "genre_train_commit": _meta_commit(genre_run_dir, "meta_genre.json") or None,
    "results_csv": str(results_csv),
}
meta_path = output_dir / "meta_full_testset_eval.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

out_path = output_dir / "full_testset_eval_output.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(_out_lines))

_out("")
_out(f"Gespeichert: {results_csv}")
_out(f"Gespeichert: {meta_path}")
_out(f"Gespeichert: {out_path}")
