"""
Evaluiert einen einzelnen trainierten Head (pair oder genre) auf einem Split (val/test)
und schreibt optional Metriken + Selection-Score als JSON.
"""
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import config
from dataset import PairDataset
from metrics import recall_at_k, labels_from_split_csv, label_relevance_matrix, pair_relevance_matrix
from models import load_projection_heads, load_projection_heads_genre


def _avg_bidir(metric_fn, sim: torch.Tensor, rel: torch.Tensor):
    va = metric_fn(sim, relevance=rel)
    av = metric_fn(sim.T, relevance=rel.T)
    return va, av, (va + av) / 2.0


def main():
    training_type = os.environ.get("TRAINING_TYPE", "pair").strip().lower()  # pair|genre
    split_name = os.environ.get("EVAL_SPLIT", "val").strip()  # val|test
    model_path = os.environ.get("EVAL_MODEL_PATH")  # Ordner oder .pt
    output_json = os.environ.get("EVAL_METRICS_JSON")

    if training_type not in {"pair", "genre"}:
        raise ValueError("TRAINING_TYPE muss 'pair' oder 'genre' sein.")
    if not model_path:
        raise ValueError("EVAL_MODEL_PATH ist erforderlich.")

    run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
    if not run_name:
        raise RuntimeError("DATASET_RUN_NAME nicht gesetzt und kein Dataset-Run gefunden.")
    run_dir = config.DATASETS_ROOT / run_name
    embeddings_dir = run_dir / "embeddings"
    split_csv = run_dir / "train_val_test_split.csv"
    device = config.DEVICE

    ds = PairDataset(split_name, split_csv, embeddings_dir)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    if training_type == "pair":
        video_head, audio_head = load_projection_heads(model_path)
    else:
        video_head, audio_head = load_projection_heads_genre(model_path)

    v_list, a_list = [], []
    for v, a in loader:
        v_list.append(v)
        a_list.append(a)
    V = torch.cat(v_list, dim=0).to(device)
    A = torch.cat(a_list, dim=0).to(device)

    with torch.no_grad():
        vp = video_head(V)
        ap = audio_head(A)
    sim = (vp @ ap.T).cpu()

    labels = labels_from_split_csv(split_csv, split_name, relevance_column="label", embeddings_dir=embeddings_dir)
    rel_pair = pair_relevance_matrix(sim.size(0))
    rel_label = label_relevance_matrix(labels)

    r10_a_va, r10_a_av, r10_a_avg = _avg_bidir(
        lambda s, relevance: recall_at_k(s, 10, relevance=relevance), sim, rel_pair
    )
    r10_b_va, r10_b_av, r10_b_avg = _avg_bidir(
        lambda s, relevance: recall_at_k(s, 10, relevance=relevance), sim, rel_label
    )

    # Task-aligned selection score:
    # pair -> Protokoll A, genre -> Protokoll B
    selection_score = r10_a_avg if training_type == "pair" else r10_b_avg

    result = {
        "training_type": training_type,
        "split": split_name,
        "dataset_run": run_name,
        "model_path": model_path,
        "selection_protocol": "A_pair" if training_type == "pair" else "B_label",
        "selection_score_recall_at_10_avg": float(selection_score),
        "protocol_a": {
            "recall_at_10": {"va": float(r10_a_va), "av": float(r10_a_av), "avg": float(r10_a_avg)},
        },
        "protocol_b": {
            "recall_at_10": {"va": float(r10_b_va), "av": float(r10_b_av), "avg": float(r10_b_avg)},
        },
    }

    print(json.dumps(result, indent=2), flush=True)
    if output_json:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
