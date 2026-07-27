"""
Evaluation: MRR, Recall@1/5/10, Mean Rank. V→A und A→V.

Zusätzlich: 95%-Bootstrap-CIs (Perzentilmethode) über Queries.
  BOOTSTRAP_B=10000 (default; 0 = aus)
  BOOTSTRAP_SEED=42
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "configs"))
import config
from dataset import PairDataset
from models import ProjectionHead, load_projection_heads_pair, load_projection_heads_genre, load_audio_encoder_heads_pair, load_audio_encoder_heads_genre
from metrics import (
    precision_at_k,
    labels_from_split_csv,
    label_relevance_matrix,
    pair_relevance_matrix,
    per_query_first_relevant_rank,
    per_query_reciprocal_rank,
    per_query_recall_at_k,
    bootstrap_mean_ci,
    bootstrap_diff_ci,
)
from training_metrics import save_evaluation_results_csv, save_evaluation_diff_csv

BOOTSTRAP_B = int(os.environ.get("BOOTSTRAP_B", "10000"))
BOOTSTRAP_SEED = int(os.environ.get("BOOTSTRAP_SEED", "42"))
BASELINE_KEY = "baseline"

# Run: aus Umgebung oder neuester
run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
if not run_name:
    print("FEHLER: DATASET_RUN_NAME nicht gesetzt und kein Run unter DATASETS_ROOT.", flush=True)
    sys.exit(1)
run_dir = config.DATASETS_ROOT / run_name
EMBEDDINGS_DIR = run_dir / "embeddings"
TRAIN_VAL_TEST_SPLIT_CSV = run_dir / "train_val_test_split.csv"
DEVICE = config.DEVICE

# Ein Ordner (Pipeline) oder neueste Einzel-Runs für Pair/Genre
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

# Audio-Encoder-Checkpoints (optional)
ae_pair_path = Path(os.environ["AE_PAIR_RUN_DIR"]) if os.environ.get("AE_PAIR_RUN_DIR") else config.get_latest_training_run_with("audio_encoder_pair.pt")
ae_genre_path = Path(os.environ["AE_GENRE_RUN_DIR"]) if os.environ.get("AE_GENRE_RUN_DIR") else config.get_latest_training_run_with("audio_encoder_genre.pt")


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


# Ausgabe sammeln, damit wir sie bei Pipeline in den Run-Ordner schreiben können
_out_lines: List[str] = []


def _out(s: str) -> None:
    print(s, flush=True)
    _out_lines.append(s)


_out(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
_out(f"Git-Commit (Eval): {config.get_git_commit()}")
_out(f"Dataset-Run: {run_name}")
_out(f"Pair-based Head: {pair_path or config.PROJECTION_HEADS_PATH} (train commit: {_meta_commit(pair_run_dir, 'meta_pair.json') or '-'})")
_out(f"Genre-based Head: {genre_path or config.PROJECTION_HEADS_GENRE_PATH} (train commit: {_meta_commit(genre_run_dir, 'meta_genre.json') or '-'})")
_out(f"Audio-Encoder Pair: {ae_pair_path or '-'}")
_out(f"Audio-Encoder Genre: {ae_genre_path or '-'}")
_out(f"Bootstrap: B={BOOTSTRAP_B} seed={BOOTSTRAP_SEED} (95% CI)")

test_ds = PairDataset("test", TRAIN_VAL_TEST_SPLIT_CSV, EMBEDDINGS_DIR)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
video_head_pair, audio_head_pair = load_projection_heads_pair(pair_path)
video_head_genre, audio_head_genre = load_projection_heads_genre(genre_path)
video_head_ae_pair, audio_head_ae_pair = load_audio_encoder_heads_pair(ae_pair_path)
video_head_ae_genre, audio_head_ae_genre = load_audio_encoder_heads_genre(ae_genre_path)

# Relevanz = Genre-Tag (Spalte "label"), gleiche Reihenfolge wie test_ds
labels = labels_from_split_csv(
    TRAIN_VAL_TEST_SPLIT_CSV, "test", relevance_column="label", embeddings_dir=EMBEDDINGS_DIR
)
_out("Labels shape: " + str(labels.shape))

# Pre-computed Embeddings (für head-only Modelle)
V_list, A_list = [], []
for v, a in test_loader:
    V_list.append(v)
    A_list.append(a)
V = torch.cat(V_list, dim=0).to(DEVICE)
A = torch.cat(A_list, dim=0).to(DEVICE)
Vn = F.normalize(V, p=2, dim=-1)
An = F.normalize(A, p=2, dim=-1)

# Pre-computed Audio-Encoder-Embeddings laden (in gleicher Reihenfolge wie test_ds)
def _load_ae_embeddings(run_path, subdir, samples):
    emb_dir = run_path / subdir
    embs = [torch.tensor(np.load(emb_dir / f"{video_id}.npy"), dtype=torch.float32) for video_id, *_ in samples]
    return torch.stack(embs).to(DEVICE)

A_ae_pair = _load_ae_embeddings(ae_pair_path, "audio_encoder_pair_test_embeddings", test_ds.samples) if ae_pair_path else None
A_ae_genre = _load_ae_embeddings(ae_genre_path, "audio_encoder_genre_test_embeddings", test_ds.samples) if ae_genre_path else None

# Ähnlichkeitsmatrizen
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
diff_rows: List[dict] = []
# (protocol, direction, model_key) -> per-query metric arrays
_per_query: Dict[Tuple[str, str, str], Dict[str, np.ndarray]] = {}


def _fmt_ci(mean: float, low: float, high: float) -> str:
    return f"{mean:.4f} [{low:.4f}, {high:.4f}]"


def _compute_metrics_with_ci(sim, relevance, with_precision: bool = False) -> Tuple[dict, Dict[str, np.ndarray]]:
    rr = per_query_reciprocal_rank(sim, relevance)
    r1 = per_query_recall_at_k(sim, 1, relevance)
    r5 = per_query_recall_at_k(sim, 5, relevance)
    r10 = per_query_recall_at_k(sim, 10, relevance)
    ranks = per_query_first_relevant_rank(sim, relevance)
    ranks_pos = ranks[ranks > 0]

    mrr, mrr_lo, mrr_hi = bootstrap_mean_ci(rr, BOOTSTRAP_B, seed=BOOTSTRAP_SEED)
    recall_at_1, r1_lo, r1_hi = bootstrap_mean_ci(r1, BOOTSTRAP_B, seed=BOOTSTRAP_SEED)
    recall_at_5, r5_lo, r5_hi = bootstrap_mean_ci(r5, BOOTSTRAP_B, seed=BOOTSTRAP_SEED)
    recall_at_10, r10_lo, r10_hi = bootstrap_mean_ci(r10, BOOTSTRAP_B, seed=BOOTSTRAP_SEED)
    mr, mr_lo, mr_hi = bootstrap_mean_ci(ranks_pos if len(ranks_pos) else ranks, BOOTSTRAP_B, seed=BOOTSTRAP_SEED)

    metrics = {
        "mrr": mrr,
        "mrr_ci_low": mrr_lo,
        "mrr_ci_high": mrr_hi,
        "recall_at_1": recall_at_1,
        "recall_at_1_ci_low": r1_lo,
        "recall_at_1_ci_high": r1_hi,
        "recall_at_5": recall_at_5,
        "recall_at_5_ci_low": r5_lo,
        "recall_at_5_ci_high": r5_hi,
        "recall_at_10": recall_at_10,
        "recall_at_10_ci_low": r10_lo,
        "recall_at_10_ci_high": r10_hi,
        "mean_rank": mr,
        "mean_rank_ci_low": mr_lo,
        "mean_rank_ci_high": mr_hi,
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    if with_precision:
        metrics["precision_at_1"] = float(precision_at_k(sim, 1, relevance=relevance))
        metrics["precision_at_10"] = float(precision_at_k(sim, 10, relevance=relevance))
    else:
        metrics["precision_at_1"] = ""
        metrics["precision_at_10"] = ""

    per_q = {"mrr": rr, "recall_at_1": r1, "recall_at_5": r5, "recall_at_10": r10, "rank": ranks}
    return metrics, per_q


def _record_metrics(
    protocol: str,
    protocol_name: str,
    direction: str,
    model_key: str,
    model: str,
    sim,
    relevance,
) -> None:
    metrics, per_q = _compute_metrics_with_ci(sim, relevance, with_precision=(protocol == "B"))
    _per_query[(protocol, direction, model_key)] = per_q
    results_rows.append({
        "protocol": protocol,
        "protocol_name": protocol_name,
        "direction": direction,
        "model_key": model_key,
        "model": model,
        **metrics,
    })
    _out(f"  {model}")
    line = (
        "    MRR: " + _fmt_ci(metrics["mrr"], metrics["mrr_ci_low"], metrics["mrr_ci_high"])
        + " | R@1: " + _fmt_ci(metrics["recall_at_1"], metrics["recall_at_1_ci_low"], metrics["recall_at_1_ci_high"])
        + " | R@5: " + _fmt_ci(metrics["recall_at_5"], metrics["recall_at_5_ci_low"], metrics["recall_at_5_ci_high"])
        + " | R@10: " + _fmt_ci(metrics["recall_at_10"], metrics["recall_at_10_ci_low"], metrics["recall_at_10_ci_high"])
    )
    if metrics.get("precision_at_1") != "":
        line += (
            " | P@1: " + str(metrics["precision_at_1"])
            + " | P@10: " + str(metrics["precision_at_10"])
        )
    line += " | MR: " + _fmt_ci(metrics["mean_rank"], metrics["mean_rank_ci_low"], metrics["mean_rank_ci_high"])
    _out(line)


for protocol_id, protocol_key, protocol_title, relevance in EVAL_PROTOCOLS:
    _out(f"=== Protokoll {protocol_id}: {protocol_title} ===")
    for direction, sim_getter, rel_getter in (
        ("V2A", lambda s: s, lambda r: r),
        ("A2V", lambda s: s.T, lambda r: r.T),
    ):
        dir_label = "V→A (Video als Query, Audio retrieval)" if direction == "V2A" else "A→V (Audio als Query, Video retrieval)"
        _out(f"=== {dir_label} ===")
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
        _out("")

# Gepaarte Bootstrap-Differenzen vs. Baseline (MRR, R@10)
_out("=== Bootstrap-Differenzen vs. Baseline (95% CI) ===")
for protocol_id, protocol_key, _, _ in EVAL_PROTOCOLS:
    for direction in ("V2A", "A2V"):
        base_key = (protocol_id, direction, BASELINE_KEY)
        if base_key not in _per_query:
            continue
        base = _per_query[base_key]
        _out(f"  [{protocol_id} {direction}]")
        for model_key, model_name, _ in EVAL_MODELS:
            if model_key == BASELINE_KEY:
                continue
            key = (protocol_id, direction, model_key)
            if key not in _per_query:
                continue
            cur = _per_query[key]
            for metric_name, arr_name in (("mrr", "mrr"), ("recall_at_10", "recall_at_10")):
                diff, lo, hi = bootstrap_diff_ci(
                    cur[arr_name], base[arr_name], BOOTSTRAP_B, seed=BOOTSTRAP_SEED
                )
                excludes0 = int(lo > 0 or hi < 0)
                diff_rows.append({
                    "protocol": protocol_id,
                    "protocol_name": protocol_key,
                    "direction": direction,
                    "model_key": model_key,
                    "model": model_name,
                    "baseline_key": BASELINE_KEY,
                    "metric": metric_name,
                    "diff": diff,
                    "diff_ci_low": lo,
                    "diff_ci_high": hi,
                    "ci_excludes_zero": excludes0,
                    "bootstrap_B": BOOTSTRAP_B,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                })
                star = "*" if excludes0 else ""
                _out(
                    f"    {model_name} − Baseline | {metric_name}: "
                    f"{diff:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}"
                )
        _out("")
_out("  (* = 95%-CI enthält 0 nicht)")


def _eval_output_dir() -> Path:
    # Expliziter Output-Ordner hat Vorrang (alte Eval nicht überschreiben).
    if os.environ.get("EVAL_OUTPUT_DIR"):
        return Path(os.environ["EVAL_OUTPUT_DIR"])
    if os.environ.get("TRAINING_RUN_DIR"):
        return Path(os.environ["TRAINING_RUN_DIR"])
    if pair_run_dir:
        return pair_run_dir
    return config.TRAINING_RUNS_ROOT / f"evaluation_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


def _ae_runs_differ_from_training_dir() -> bool:
    """Nur relevant, wenn Ergebnisse in TRAINING_RUN_DIR geschrieben würden."""
    if os.environ.get("EVAL_OUTPUT_DIR"):
        return False
    if not os.environ.get("TRAINING_RUN_DIR"):
        return False
    train_dir = Path(os.environ["TRAINING_RUN_DIR"]).resolve()
    for ae_path in (ae_pair_path, ae_genre_path):
        if ae_path and ae_path.resolve() != train_dir:
            return True
    return False


if _ae_runs_differ_from_training_dir():
    _out(
        "Hinweis: AE_PAIR_RUN_DIR/AE_GENRE_RUN_DIR weichen von TRAINING_RUN_DIR ab — "
        "keine Dateien im Train-Ordner geschrieben (Ergebnis nur im Slurm-Log). "
        "Oder EVAL_OUTPUT_DIR setzen."
    )
    print("", flush=True)
    sys.exit(0)

output_dir = _eval_output_dir()
output_dir.mkdir(parents=True, exist_ok=True)
_out(f"Eval-Output-Dir: {output_dir}")
results_csv = output_dir / "results_evaluation.csv"
save_evaluation_results_csv(results_rows, results_csv)
diff_csv = output_dir / "results_evaluation_bootstrap_diff.csv"
save_evaluation_diff_csv(diff_rows, diff_csv)

eval_meta = {
    "timestamp": datetime.now().isoformat(),
    "git_commit_eval": config.get_git_commit(),
    "dataset_run": run_name,
    "n_test": len(test_ds),
    "bootstrap_B": BOOTSTRAP_B,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "bootstrap_alpha": 0.05,
    "pair_head_path": str(pair_path or config.PROJECTION_HEADS_PATH),
    "genre_head_path": str(genre_path or config.PROJECTION_HEADS_GENRE_PATH),
    "audio_encoder_pair_path": str(ae_pair_path or ""),
    "audio_encoder_genre_path": str(ae_genre_path or ""),
    "pair_train_commit": _meta_commit(pair_run_dir, "meta_pair.json") or None,
    "genre_train_commit": _meta_commit(genre_run_dir, "meta_genre.json") or None,
    "results_csv": str(results_csv),
    "bootstrap_diff_csv": str(diff_csv),
}
eval_meta_path = output_dir / "meta_evaluation.json"
with open(eval_meta_path, "w", encoding="utf-8") as f:
    json.dump(eval_meta, f, indent=2)

_out(f"Metriken-CSV: {results_csv}")
_out(f"Bootstrap-Diff-CSV: {diff_csv}")
_out(f"Eval-Metadaten: {eval_meta_path}")

out_path = output_dir / "evaluation_output.txt"
_out_lines.append("")
_out_lines.append("Gespeichert: " + str(out_path))
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(_out_lines))
print("", flush=True)
print("Gespeichert: " + str(out_path), flush=True)
print("Gespeichert: " + str(results_csv), flush=True)
print("Gespeichert: " + str(diff_csv), flush=True)
