"""
comparison_eval: Stewart-kompatible Pool-Size Evaluation

Vorgehen:
  - Protocol A (Pair): zwei disjunkte stratifizierte Subsets à 1820 Samples
    (182 pro Genre bei 10 Genres), Metriken über beide Subsets gemittelt.
  - Protocol B (Genre): voller Test-Split (N=3668), inkl. Precision@1/@10.

Wichtig:
  - evaluation.py bleibt unverändert.
  - Dieses Skript ist eine zusätzliche Evaluations-Variante.
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import config
from dataset import PairDataset
from metrics import (
    MRR,
    label_relevance_matrix,
    mean_rank,
    pair_relevance_matrix,
    precision_at_k,
    recall_at_k,
)
from models import (
    ProjectionHead,
    load_audio_encoder_heads_genre,
    load_audio_encoder_heads_pair,
    load_projection_heads_genre,
    load_projection_heads_pair,
)
from training_metrics import save_evaluation_results_csv

EVAL_MODELS_META = [
    ("baseline", "Baseline"),
    ("untrained", "Untrained heads"),
    ("pair", "Pair-based"),
    ("genre", "Genre-based"),
    ("audio_encoder_pair", "Audio-Encoder Pair"),
    ("audio_encoder_genre", "Audio-Encoder Genre"),
]

EVAL_PROTOCOLS = [
    ("A", "pair", "Pair-basierte Relevanz (exaktes Video-Audio-Paar)"),
    ("B", "label", "Label-basierte Relevanz (gleiches Genre)"),
]


def _meta_commit(run_dir: Optional[Path], meta_name: str = "meta.json") -> str:
    if run_dir is None:
        return ""
    for name in (meta_name, "meta.json"):
        p = run_dir / name
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    m = json.load(f)
                return m.get("git_commit", "") or ""
            except Exception:
                pass
    return ""


def _eval_output_dir() -> Path:
    if os.environ.get("TRAINING_RUN_DIR"):
        return Path(os.environ["TRAINING_RUN_DIR"])
    if os.environ.get("EVAL_OUTPUT_DIR"):
        return Path(os.environ["EVAL_OUTPUT_DIR"])
    return config.TRAINING_RUNS_ROOT / f"comparison_eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


def _load_ae_embeddings(
    run_path: Path,
    subdir: str,
    samples: Sequence[Tuple[str, Path, Path, str]],
    device: str,
) -> torch.Tensor:
    emb_dir = run_path / subdir
    embs = [
        torch.tensor(np.load(emb_dir / f"{video_id}.npy"), dtype=torch.float32)
        for video_id, *_rest in samples
    ]
    return torch.stack(embs, dim=0).to(device)


def _load_full_test_tensors(
    test_ds: PairDataset,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    V_list, A_list = [], []
    for v, a in loader:
        V_list.append(v)
        A_list.append(a)
    V = torch.cat(V_list, dim=0).to(device)
    A = torch.cat(A_list, dim=0).to(device)
    labels = np.array([s[3] for s in test_ds.samples])
    return V, A, labels


def _compute_similarities(
    *,
    V: torch.Tensor,
    A: torch.Tensor,
    A_ae_pair: torch.Tensor,
    A_ae_genre: torch.Tensor,
    video_head_pair,
    audio_head_pair,
    video_head_genre,
    audio_head_genre,
    video_head_ae_pair,
    audio_head_ae_pair,
    video_head_ae_genre,
    audio_head_ae_genre,
    video_head_rand,
    audio_head_rand,
) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        Vn = F.normalize(V, p=2, dim=-1)
        An = F.normalize(A, p=2, dim=-1)

        sim_baseline = (Vn @ An.T).cpu()

        v_rand = F.normalize(video_head_rand(V), p=2, dim=-1)
        a_rand = F.normalize(audio_head_rand(A), p=2, dim=-1)
        sim_rand = (v_rand @ a_rand.T).cpu()

        v_pair = F.normalize(video_head_pair(V), p=2, dim=-1)
        a_pair = F.normalize(audio_head_pair(A), p=2, dim=-1)
        sim_pair = (v_pair @ a_pair.T).cpu()

        v_genre = F.normalize(video_head_genre(V), p=2, dim=-1)
        a_genre = F.normalize(audio_head_genre(A), p=2, dim=-1)
        sim_genre = (v_genre @ a_genre.T).cpu()

        v_ae_pair = F.normalize(video_head_ae_pair(V), p=2, dim=-1)
        a_ae_pair = F.normalize(audio_head_ae_pair(A_ae_pair), p=2, dim=-1)
        sim_ae_pair = (v_ae_pair @ a_ae_pair.T).cpu()

        v_ae_genre = F.normalize(video_head_ae_genre(V), p=2, dim=-1)
        a_ae_genre = F.normalize(audio_head_ae_genre(A_ae_genre), p=2, dim=-1)
        sim_ae_genre = (v_ae_genre @ a_ae_genre.T).cpu()

    return {
        "baseline": sim_baseline,
        "untrained": sim_rand,
        "pair": sim_pair,
        "genre": sim_genre,
        "audio_encoder_pair": sim_ae_pair,
        "audio_encoder_genre": sim_ae_genre,
    }


def _extract_submatrix(sim: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    idx = torch.tensor(indices, dtype=torch.long)
    return sim.index_select(0, idx).index_select(1, idx)


def _compute_metrics(sim: torch.Tensor, relevance: torch.Tensor, with_precision: bool) -> Dict[str, float]:
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


def _evaluate(
    similarities: Dict[str, torch.Tensor],
    labels: np.ndarray,
    protocol_ids: Sequence[str],
) -> List[dict]:
    n = len(labels)
    rel_pair = pair_relevance_matrix(n)
    rel_label = label_relevance_matrix(labels)
    relevance_by_protocol = {"A": rel_pair, "B": rel_label}

    results_rows: List[dict] = []
    for protocol_id, protocol_key, _protocol_title in EVAL_PROTOCOLS:
        if protocol_id not in protocol_ids:
            continue
        relevance = relevance_by_protocol[protocol_id]
        with_precision = protocol_id == "B"

        for direction, sim_getter, rel_getter in (
            ("V2A", lambda s: s, lambda r: r),
            ("A2V", lambda s: s.T, lambda r: r.T),
        ):
            for model_key, model_name in EVAL_MODELS_META:
                sim = sim_getter(similarities[model_key])
                rel = rel_getter(relevance)
                metrics = _compute_metrics(sim, rel, with_precision=with_precision)
                results_rows.append(
                    {
                        "protocol": protocol_id,
                        "protocol_name": protocol_key,
                        "direction": direction,
                        "model_key": model_key,
                        "model": model_name,
                        **metrics,
                    }
                )
    return results_rows


def _average_subset_results(rows1: List[dict], rows2: List[dict]) -> List[dict]:
    def key_fn(r: dict) -> Tuple[str, str, str, str]:
        return (r["protocol"], r["direction"], r["model_key"], r["model"])

    d1 = {key_fn(r): r for r in rows1}
    d2 = {key_fn(r): r for r in rows2}
    avg_rows: List[dict] = []

    for k in sorted(d1.keys()):
        r1 = d1[k]
        r2 = d2[k]
        out = {
            "protocol": r1["protocol"],
            "protocol_name": r1.get("protocol_name", ""),
            "direction": r1["direction"],
            "model_key": r1["model_key"],
            "model": r1["model"],
        }
        for m in [
            "mrr",
            "recall_at_1",
            "recall_at_5",
            "recall_at_10",
            "mean_rank",
            "precision_at_1",
            "precision_at_10",
        ]:
            v1 = r1.get(m)
            v2 = r2.get(m)
            if v1 is None or v2 is None:
                continue
            out[m] = (float(v1) + float(v2)) / 2.0
        avg_rows.append(out)
    return avg_rows


def _combine_protocol_results(protocol_a_rows: List[dict], protocol_b_rows: List[dict]) -> List[dict]:
    combined = protocol_a_rows + protocol_b_rows
    combined.sort(key=lambda r: (r["protocol"], r["direction"], r["model_key"]))
    return combined


# --- main ---

run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
if not run_name:
    print("FEHLER: Kein DATASET_RUN_NAME gefunden.", flush=True)
    sys.exit(1)

run_dir = config.DATASETS_ROOT / run_name
EMBEDDINGS_DIR = run_dir / "embeddings"
TRAIN_VAL_TEST_SPLIT_CSV = run_dir / "train_val_test_split.csv"
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

if not pair_path or not genre_path or not ae_pair_path or not ae_genre_path:
    print("FEHLER: Mindestens ein Modell-Checkpoint fehlt.", flush=True)
    sys.exit(1)

output_dir = _eval_output_dir()
output_dir.mkdir(parents=True, exist_ok=True)

target_total = int(os.environ.get("COMPARISON_TARGET_TOTAL", "1820"))
subset_seed = int(os.environ.get("COMPARISON_SUBSET_SEED", "42"))
per_genre_env = os.environ.get("COMPARISON_PER_GENRE")

test_ds = PairDataset("test", TRAIN_VAL_TEST_SPLIT_CSV, EMBEDDINGS_DIR)
all_labels = [s[3] for s in test_ds.samples]
unique_labels = sorted(set(all_labels))
num_genres = len(unique_labels)
if num_genres == 0:
    print("FEHLER: Keine Labels im Testsplit gefunden.", flush=True)
    sys.exit(1)

if per_genre_env:
    per_genre = int(per_genre_env)
    if per_genre * num_genres != target_total:
        print(
            f"FEHLER: COMPARISON_PER_GENRE={per_genre} passt nicht zu target_total={target_total}.",
            flush=True,
        )
        sys.exit(1)
else:
    if target_total % num_genres != 0:
        print(
            f"FEHLER: target_total={target_total} ist nicht teilbar durch num_genres={num_genres}.",
            flush=True,
        )
        sys.exit(1)
    per_genre = target_total // num_genres

indices_by_label: Dict[str, List[int]] = defaultdict(list)
for idx, sample in enumerate(test_ds.samples):
    indices_by_label[sample[3]].append(idx)

for lab in unique_labels:
    if len(indices_by_label[lab]) < 2 * per_genre:
        print(
            f"FEHLER: Genre '{lab}' hat nur {len(indices_by_label[lab])} Samples, "
            f"benötigt mindestens {2 * per_genre}.",
            flush=True,
        )
        sys.exit(1)

rng = np.random.default_rng(subset_seed)
subset1_indices: List[int] = []
subset2_indices: List[int] = []
for lab in unique_labels:
    perm = rng.permutation(indices_by_label[lab])
    subset1_indices.extend(perm[:per_genre].tolist())
    subset2_indices.extend(perm[per_genre : 2 * per_genre].tolist())

subset1_indices_sorted = sorted(subset1_indices)
subset2_indices_sorted = sorted(subset2_indices)
if set(subset1_indices_sorted).intersection(subset2_indices_sorted):
    print("FEHLER: Subsets sind nicht disjunkt.", flush=True)
    sys.exit(1)

print(f"Comparison Eval run: {run_name}")
print(f"  Test samples (Protocol B): {len(test_ds)}")
print(f"  Subset size (Protocol A): {target_total} ({per_genre} pro Genre, {num_genres} Genres)")
print(f"  Subset seed: {subset_seed}")

video_head_pair, audio_head_pair = load_projection_heads_pair(pair_path)
video_head_genre, audio_head_genre = load_projection_heads_genre(genre_path)
video_head_ae_pair, audio_head_ae_pair = load_audio_encoder_heads_pair(ae_pair_path)
video_head_ae_genre, audio_head_ae_genre = load_audio_encoder_heads_genre(ae_genre_path)

torch.manual_seed(subset_seed)
video_head_rand = ProjectionHead().to(DEVICE).eval()
audio_head_rand = ProjectionHead().to(DEVICE).eval()

V, A, full_labels = _load_full_test_tensors(test_ds, DEVICE)
A_ae_pair = _load_ae_embeddings(ae_pair_path, "audio_encoder_pair_test_embeddings", test_ds.samples, DEVICE)
A_ae_genre = _load_ae_embeddings(ae_genre_path, "audio_encoder_genre_test_embeddings", test_ds.samples, DEVICE)

full_similarities = _compute_similarities(
    V=V,
    A=A,
    A_ae_pair=A_ae_pair,
    A_ae_genre=A_ae_genre,
    video_head_pair=video_head_pair,
    audio_head_pair=audio_head_pair,
    video_head_genre=video_head_genre,
    audio_head_genre=audio_head_genre,
    video_head_ae_pair=video_head_ae_pair,
    audio_head_ae_pair=audio_head_ae_pair,
    video_head_ae_genre=video_head_ae_genre,
    audio_head_ae_genre=audio_head_ae_genre,
    video_head_rand=video_head_rand,
    audio_head_rand=audio_head_rand,
)

# Protocol B: voller Test-Split
results_protocol_b_full = _evaluate(full_similarities, full_labels, protocol_ids=["B"])

# Protocol A: stratifizierte Subsets
subset1_similarities = {
    key: _extract_submatrix(sim, subset1_indices_sorted) for key, sim in full_similarities.items()
}
subset2_similarities = {
    key: _extract_submatrix(sim, subset2_indices_sorted) for key, sim in full_similarities.items()
}
labels_subset1 = full_labels[subset1_indices_sorted]
labels_subset2 = full_labels[subset2_indices_sorted]

results_protocol_a_subset1 = _evaluate(subset1_similarities, labels_subset1, protocol_ids=["A"])
results_protocol_a_subset2 = _evaluate(subset2_similarities, labels_subset2, protocol_ids=["A"])
results_protocol_a_avg = _average_subset_results(results_protocol_a_subset1, results_protocol_a_subset2)

combined_results = _combine_protocol_results(results_protocol_a_avg, results_protocol_b_full)

subset1_csv = output_dir / "results_evaluation_stewart_protocol_a_subset1.csv"
subset2_csv = output_dir / "results_evaluation_stewart_protocol_a_subset2.csv"
protocol_a_avg_csv = output_dir / "results_evaluation_stewart_protocol_a_avg.csv"
protocol_b_full_csv = output_dir / "results_evaluation_stewart_protocol_b_full.csv"
combined_csv = output_dir / "results_evaluation_stewart_combined.csv"

save_evaluation_results_csv(results_protocol_a_subset1, subset1_csv)
save_evaluation_results_csv(results_protocol_a_subset2, subset2_csv)
save_evaluation_results_csv(results_protocol_a_avg, protocol_a_avg_csv)
save_evaluation_results_csv(results_protocol_b_full, protocol_b_full_csv)
save_evaluation_results_csv(combined_results, combined_csv)

meta = {
    "timestamp": datetime.now().isoformat(),
    "git_commit_eval": config.get_git_commit(),
    "dataset_run": run_name,
    "n_test_total": len(test_ds),
    "protocol_a_pool_size": target_total,
    "protocol_b_pool_size": len(test_ds),
    "num_genres": num_genres,
    "per_genre": per_genre,
    "subset_seed": subset_seed,
    "subset1_indices_sorted": subset1_indices_sorted,
    "subset2_indices_sorted": subset2_indices_sorted,
    "subset1_video_ids": [test_ds.samples[i][0] for i in subset1_indices_sorted],
    "subset2_video_ids": [test_ds.samples[i][0] for i in subset2_indices_sorted],
    "pair_head_path": str(pair_path),
    "genre_head_path": str(genre_path),
    "audio_encoder_pair_path": str(ae_pair_path),
    "audio_encoder_genre_path": str(ae_genre_path),
    "pair_train_commit": _meta_commit(pair_run_dir, "meta_pair.json") or None,
    "genre_train_commit": _meta_commit(genre_run_dir, "meta_genre.json") or None,
    "results_csv_combined": str(combined_csv),
    "results_csv_protocol_a_avg": str(protocol_a_avg_csv),
    "results_csv_protocol_b_full": str(protocol_b_full_csv),
    "results_csv_protocol_a_subset1": str(subset1_csv),
    "results_csv_protocol_a_subset2": str(subset2_csv),
}
meta_path = output_dir / "meta_comparison_eval.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("")
print(f"Saved Protocol A subset1: {subset1_csv}", flush=True)
print(f"Saved Protocol A subset2: {subset2_csv}", flush=True)
print(f"Saved Protocol A avg:     {protocol_a_avg_csv}", flush=True)
print(f"Saved Protocol B full:    {protocol_b_full_csv}", flush=True)
print(f"Saved combined:           {combined_csv}", flush=True)
print(f"Saved meta:               {meta_path}", flush=True)
