"""
comparison_eval: Stewart-kompatible Pool-Size Evaluation

Goal:
  Vergleiche Retrieval-Ergebnisse unter kontrollierter Kandidaten-Pool-Größe.

Vorgehen:
  - Nimm den kompletten Test-Split (PairDataset("test", ...)).
  - Stratifiziere den Testsplit nach Genre/Label.
  - Erzeuge zwei disjunkte Subsets mit jeweils gleicher Genre-Abdeckung.
    Standard: insgesamt target_total=1820 Samples => per_genre=182 bei 10 Genres.
  - Führe auf jedem Subset eine vollständige Retrieval-Evaluation aus (V2A, A2V;
    Protocol A und B; Protocol B beinhaltet Precision@1 und Precision@10).
  - Mittlere die Metriken beider Subsets (gleiches N pro Subset).

Wichtig:
  - Die bestehende evaluation.py bleibt unverändert.
  - Diese Datei ist eine zusätzliche Evaluations-Variante.
"""

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

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


@dataclass(frozen=True)
class SubsetSpec:
    subset_id: str  # "subset1" | "subset2"
    seed: int
    per_genre: int
    target_total: int


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
    # Fallback: wie evaluation.py
    return config.TRAINING_RUNS_ROOT / f"comparison_eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


def _ae_runs_differ_from_training_dir(
    train_dir: Optional[Path], ae_pair_path: Optional[Path], ae_genre_path: Optional[Path]
) -> bool:
    if not train_dir:
        return False
    for ae_path in (ae_pair_path, ae_genre_path):
        if ae_path and ae_path.resolve() != train_dir.resolve():
            return True
    return False


def _load_subset_tensors(
    subset_samples: Sequence[Tuple[str, Path, Path, str]],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Lädt V und A Embeddings in exakt der Reihenfolge von subset_samples.

    Rückgabe:
      - V: (n, d) Video-Embeddings
      - A: (n, d) Audio-Embeddings
      - labels: 1D array (n,) in gleicher Reihenfolge
    """
    V_list: List[torch.Tensor] = []
    A_list: List[torch.Tensor] = []
    labels_list: List[str] = []

    # subset_samples Elemente sind (video_id, v_path, a_path, label)
    for video_id, v_path, a_path, label in subset_samples:
        _ = video_id  # nur zur Klarheit
        V_list.append(torch.tensor(np.load(v_path), dtype=torch.float32))
        A_list.append(torch.tensor(np.load(a_path), dtype=torch.float32))
        labels_list.append(label)

    V = torch.stack(V_list, dim=0).to(device)
    A = torch.stack(A_list, dim=0).to(device)
    labels = np.array(labels_list)
    return V, A, labels


def _load_ae_embeddings(
    run_path: Path,
    subdir: str,
    subset_samples: Sequence[Tuple[str, Path, Path, str]],
    device: str,
) -> torch.Tensor:
    emb_dir = run_path / subdir
    embs: List[torch.Tensor] = []
    for video_id, *_rest in subset_samples:
        embs.append(torch.tensor(np.load(emb_dir / f"{video_id}.npy"), dtype=torch.float32))
    return torch.stack(embs, dim=0).to(device)


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


def _record_metrics(
    results_rows: List[dict],
    protocol: str,
    protocol_name: str,
    direction: str,
    model_key: str,
    model: str,
    sim: torch.Tensor,
    relevance: torch.Tensor,
    with_precision: bool,
    out_lines: List[str],
) -> None:
    metrics = _compute_metrics(sim, relevance, with_precision=with_precision)
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

    if model_key in ("baseline", "pair", "genre", "audio_encoder_pair", "audio_encoder_genre", "untrained"):
        out_lines.append(f"  {model} ({protocol},{direction}) mrr={metrics.get('mrr'):.6f}")


def _evaluate_on_subset(
    *,
    subset_samples: Sequence[Tuple[str, Path, Path, str]],
    subset_spec: SubsetSpec,
    device: str,
    video_head_pair,
    audio_head_pair,
    video_head_genre,
    audio_head_genre,
    video_head_ae_pair,
    audio_head_ae_pair,
    video_head_ae_genre,
    audio_head_ae_genre,
    ae_pair_path: Path,
    ae_genre_path: Path,
    video_head_rand,
    audio_head_rand,
) -> List[dict]:
    # Embeddings + Labels
    V, A, labels = _load_subset_tensors(subset_samples, device=device)
    Vn = F.normalize(V, p=2, dim=-1)
    An = F.normalize(A, p=2, dim=-1)

    # Similarities
    with torch.no_grad():
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

        # Audio-Encoder Embeddings (precomputed wav2clip encoder output)
        A_ae_pair = _load_ae_embeddings(
            run_path=ae_pair_path,
            subdir="audio_encoder_pair_test_embeddings",
            subset_samples=subset_samples,
            device=device,
        )
        A_ae_genre = _load_ae_embeddings(
            run_path=ae_genre_path,
            subdir="audio_encoder_genre_test_embeddings",
            subset_samples=subset_samples,
            device=device,
        )

        v_ae_pair = F.normalize(video_head_ae_pair(V), p=2, dim=-1)
        a_ae_pair = F.normalize(audio_head_ae_pair(A_ae_pair), p=2, dim=-1)
        sim_ae_pair = (v_ae_pair @ a_ae_pair.T).cpu()

        v_ae_genre = F.normalize(video_head_ae_genre(V), p=2, dim=-1)
        a_ae_genre = F.normalize(audio_head_ae_genre(A_ae_genre), p=2, dim=-1)
        sim_ae_genre = (v_ae_genre @ a_ae_genre.T).cpu()

    # Relevance
    n = sim_baseline.size(0)
    rel_pair = pair_relevance_matrix(n)
    rel_label = label_relevance_matrix(labels)

    eval_models = [
        ("baseline", "Baseline", sim_baseline),
        ("untrained", "Untrained heads", sim_rand),
        ("pair", "Pair-based", sim_pair),
        ("genre", "Genre-based", sim_genre),
        ("audio_encoder_pair", "Audio-Encoder Pair", sim_ae_pair),
        ("audio_encoder_genre", "Audio-Encoder Genre", sim_ae_genre),
    ]
    eval_protocols = [
        ("A", "pair", "Pair-basierte Relevanz (exaktes Video-Audio-Paar)", rel_pair),
        ("B", "label", "Label-basierte Relevanz (gleiches Genre)", rel_label),
    ]

    results_rows: List[dict] = []
    out_lines: List[str] = []

    for protocol_id, protocol_key, protocol_title, relevance in eval_protocols:
        for direction, sim_getter, rel_getter in (
            ("V2A", lambda s: s, lambda r: r),
            ("A2V", lambda s: s.T, lambda r: r.T),
        ):
            for model_key, model_name, sim in eval_models:
                with_precision = protocol_id == "B"
                _record_metrics(
                    results_rows,
                    protocol=protocol_id,
                    protocol_name=protocol_key,
                    direction=direction,
                    model_key=model_key,
                    model=model_name,
                    sim=sim_getter(sim),
                    relevance=rel_getter(relevance),
                    with_precision=with_precision,
                    out_lines=out_lines,
                )

    # subset_spec aktuell nicht in CSV, aber wir halten es für Meta/Debug im Kopf
    return results_rows


def _average_subset_results(rows1: List[dict], rows2: List[dict]) -> List[dict]:
    fields = [
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

    def key_fn(r: dict) -> Tuple[str, str, str, str]:
        return (r["protocol"], r["direction"], r["model_key"], r["model"])

    d1 = {key_fn(r): r for r in rows1}
    d2 = {key_fn(r): r for r in rows2}
    keys = sorted(d1.keys())

    avg_rows: List[dict] = []
    for k in keys:
        r1 = d1[k]
        r2 = d2[k]
        out = {
            "protocol": r1["protocol"],
            "protocol_name": r1.get("protocol_name", ""),
            "direction": r1["direction"],
            "model_key": r1["model_key"],
            "model": r1["model"],
        }
        # Mittelung: Wenn bei Protocol A keine precision-Spalten existieren, bleiben sie leer/fehlend.
        for m in ["mrr", "recall_at_1", "recall_at_5", "recall_at_10", "mean_rank", "precision_at_1", "precision_at_10"]:
            v1 = r1.get(m, None)
            v2 = r2.get(m, None)
            if v1 is None or v2 is None:
                # nicht vorhanden (typisch bei Protocol A)
                continue
            out[m] = (float(v1) + float(v2)) / 2.0

        avg_rows.append(out)
    return avg_rows


# --- main ---

run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
if not run_name:
    print("FEHLER: Kein DATASET_RUN_NAME gefunden.", flush=True)
    sys.exit(1)

run_dir = config.DATASETS_ROOT / run_name
EMBEDDINGS_DIR = run_dir / "embeddings"
TRAIN_VAL_TEST_SPLIT_CSV = run_dir / "train_val_test_split.csv"
DEVICE = config.DEVICE

# Training runs / heads
train_dir = Path(os.environ["TRAINING_RUN_DIR"]) if os.environ.get("TRAINING_RUN_DIR") else None

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

# Audio-Encoder checkpoints
ae_pair_path = Path(os.environ["AE_PAIR_RUN_DIR"]) if os.environ.get("AE_PAIR_RUN_DIR") else config.get_latest_training_run_with("audio_encoder_pair.pt")
ae_genre_path = Path(os.environ["AE_GENRE_RUN_DIR"]) if os.environ.get("AE_GENRE_RUN_DIR") else config.get_latest_training_run_with("audio_encoder_genre.pt")

if _ae_runs_differ_from_training_dir(train_dir, ae_pair_path, ae_genre_path):
    print("Hinweis: AE_*_RUN_DIR weichen von TRAINING_RUN_DIR ab — keine Dateien im Train-Ordner geschrieben.")
    print("")
    # Wir lassen trotzdem laufen, aber schreiben in EVAL_OUTPUT_DIR oder eigenen Fallback-Ordner

output_dir = _eval_output_dir()
output_dir.mkdir(parents=True, exist_ok=True)

# Pool-size configuration
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
            f"FEHLER: COMPARISON_PER_GENRE={per_genre} ergibt nicht target_total={target_total} "
            f"(num_genres={num_genres} -> {per_genre * num_genres}).",
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

# Build stratified disjoint subsets: subset1 uses first per_genre, subset2 uses next per_genre
indices_by_label: Dict[str, List[int]] = defaultdict(list)
for idx, sample in enumerate(test_ds.samples):
    indices_by_label[sample[3]].append(idx)

for lab in unique_labels:
    if len(indices_by_label[lab]) < 2 * per_genre:
        print(
            f"FEHLER: Genre '{lab}' hat nur {len(indices_by_label[lab])} Samples, benötigt aber mindestens {2*per_genre} für disjunkte Subsets.",
            flush=True,
        )
        sys.exit(1)

rng = np.random.default_rng(subset_seed)
subset1_indices: List[int] = []
subset2_indices: List[int] = []

for lab in unique_labels:
    idxs = indices_by_label[lab]
    perm = rng.permutation(idxs)
    subset1_indices.extend(perm[:per_genre].tolist())
    subset2_indices.extend(perm[per_genre : 2 * per_genre].tolist())

subset1_indices_sorted = sorted(subset1_indices)
subset2_indices_sorted = sorted(subset2_indices)

if set(subset1_indices_sorted).intersection(set(subset2_indices_sorted)):
    print("FEHLER: Subsets sind nicht disjunkt (Index-Overlap).", flush=True)
    sys.exit(1)

subset1_samples = [test_ds.samples[i] for i in subset1_indices_sorted]
subset2_samples = [test_ds.samples[i] for i in subset2_indices_sorted]

print(f"Comparison Eval run: {run_name}")
print(f"  Test samples: {len(test_ds)}")
print(f"  Genres: {num_genres} -> per_genre={per_genre}, target_total={per_genre * num_genres}")
print(f"  Subset seed: {subset_seed}")
print(f"  Subset1: {len(subset1_samples)} | Subset2: {len(subset2_samples)}")

subset_spec1 = SubsetSpec(subset_id="subset1", seed=subset_seed, per_genre=per_genre, target_total=target_total)
subset_spec2 = SubsetSpec(subset_id="subset2", seed=subset_seed, per_genre=per_genre, target_total=target_total)

# Load heads (wie evaluation.py)
if not pair_path or not genre_path:
    print("FEHLER: Projection heads (pair/genre) nicht gefunden.", flush=True)
    sys.exit(1)

video_head_pair, audio_head_pair = load_projection_heads_pair(pair_path)
video_head_genre, audio_head_genre = load_projection_heads_genre(genre_path)
video_head_ae_pair, audio_head_ae_pair = load_audio_encoder_heads_pair(ae_pair_path)
video_head_ae_genre, audio_head_ae_genre = load_audio_encoder_heads_genre(ae_genre_path)

# Deterministische untrained heads für beide Subsets
torch.manual_seed(subset_seed)
video_head_rand = ProjectionHead().to(DEVICE).eval()
audio_head_rand = ProjectionHead().to(DEVICE).eval()

results_subset1 = _evaluate_on_subset(
    subset_samples=subset1_samples,
    subset_spec=subset_spec1,
    device=DEVICE,
    video_head_pair=video_head_pair,
    audio_head_pair=audio_head_pair,
    video_head_genre=video_head_genre,
    audio_head_genre=audio_head_genre,
    video_head_ae_pair=video_head_ae_pair,
    audio_head_ae_pair=audio_head_ae_pair,
    video_head_ae_genre=video_head_ae_genre,
    audio_head_ae_genre=audio_head_ae_genre,
    ae_pair_path=ae_pair_path,
    ae_genre_path=ae_genre_path,
    video_head_rand=video_head_rand,
    audio_head_rand=audio_head_rand,
)

results_subset2 = _evaluate_on_subset(
    subset_samples=subset2_samples,
    subset_spec=subset_spec2,
    device=DEVICE,
    video_head_pair=video_head_pair,
    audio_head_pair=audio_head_pair,
    video_head_genre=video_head_genre,
    audio_head_genre=audio_head_genre,
    video_head_ae_pair=video_head_ae_pair,
    audio_head_ae_pair=audio_head_ae_pair,
    video_head_ae_genre=video_head_ae_genre,
    audio_head_ae_genre=audio_head_ae_genre,
    ae_pair_path=ae_pair_path,
    ae_genre_path=ae_genre_path,
    video_head_rand=video_head_rand,
    audio_head_rand=audio_head_rand,
)

avg_results = _average_subset_results(results_subset1, results_subset2)

# Write outputs (3 CSVs)
subset1_csv = output_dir / "results_evaluation_stewart_pool_subset1.csv"
subset2_csv = output_dir / "results_evaluation_stewart_pool_subset2.csv"
avg_csv = output_dir / "results_evaluation_stewart_pool_avg.csv"

save_evaluation_results_csv(results_subset1, subset1_csv)
save_evaluation_results_csv(results_subset2, subset2_csv)
save_evaluation_results_csv(avg_results, avg_csv)

# Meta
meta = {
    "timestamp": datetime.now().isoformat(),
    "dataset_run": run_name,
    "n_test_total": len(test_ds),
    "num_genres": num_genres,
    "target_total": target_total,
    "per_genre": per_genre,
    "subset_seed": subset_seed,
    "subset1_indices_sorted": subset1_indices_sorted,
    "subset2_indices_sorted": subset2_indices_sorted,
    "subset1_video_ids": [s[0] for s in subset1_samples],
    "subset2_video_ids": [s[0] for s in subset2_samples],
    "pair_head_path": str(pair_path),
    "genre_head_path": str(genre_path),
    "audio_encoder_pair_path": str(ae_pair_path),
    "audio_encoder_genre_path": str(ae_genre_path),
    "pair_train_commit": _meta_commit(pair_run_dir, "meta_pair.json") or None,
    "genre_train_commit": _meta_commit(genre_run_dir, "meta_genre.json") or None,
    "results_csv_avg": str(avg_csv),
    "results_csv_subset1": str(subset1_csv),
    "results_csv_subset2": str(subset2_csv),
}
meta_path = output_dir / "meta_comparison_eval.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("")
print(f"Saved subset1 CSV: {subset1_csv}", flush=True)
print(f"Saved subset2 CSV: {subset2_csv}", flush=True)
print(f"Saved AVG CSV:     {avg_csv}", flush=True)
print(f"Saved meta:        {meta_path}", flush=True)

