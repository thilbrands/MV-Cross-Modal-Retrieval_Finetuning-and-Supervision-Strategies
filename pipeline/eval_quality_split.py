"""
Evaluation aufgesplittet nach visueller Qualität (VLM-Score: KEEP_HIGH vs. KEEP_LOW).

Retrieval-Pool = volles Test-Set (alle 3.668 Segmente). Nur die Menge der Queries,
über die gemittelt wird, wird gefiltert (einmal KEEP_HIGH, einmal KEEP_LOW).

Modelle: Baseline (frozen), E3a (Audio-Encoder Pair), E3b (Audio-Encoder Genre).
Metriken: MRR, Recall@10 — Protocol A (Pair) + B (Genre), Richtungen V→A und A→V.

Env-Vars:
  DATASET_RUN_NAME   — Dataset-Run (default: neuester)
  TRAINING_RUN_DIR   — Run mit audio_encoder_*.pt (auch als AE-Default)
  AE_PAIR_RUN_DIR / AE_GENRE_RUN_DIR — überschreiben TRAINING_RUN_DIR pro Encoder
  EVAL_OUTPUT_DIR    — Ausgabeordner (default: TRAINING_RUN_DIR / Dataset-Run)
"""
import csv
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
import config
from dataset import PairDataset
from models import load_audio_encoder_heads_pair, load_audio_encoder_heads_genre
from metrics import MRR, recall_at_k, labels_from_split_csv, label_relevance_matrix, pair_relevance_matrix

run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
if not run_name:
    print("FEHLER: Kein Dataset-Run gefunden.", flush=True)
    sys.exit(1)

run_dir = config.DATASETS_ROOT / run_name
EMBEDDINGS_DIR = run_dir / "embeddings"
SPLIT_CSV = run_dir / "train_val_test_split.csv"
DEVICE = config.DEVICE

ae_pair_path = (
    Path(os.environ["AE_PAIR_RUN_DIR"]) if os.environ.get("AE_PAIR_RUN_DIR")
    else Path(os.environ["TRAINING_RUN_DIR"]) if os.environ.get("TRAINING_RUN_DIR")
    else config.get_latest_training_run_with("audio_encoder_pair.pt")
)
ae_genre_path = (
    Path(os.environ["AE_GENRE_RUN_DIR"]) if os.environ.get("AE_GENRE_RUN_DIR")
    else Path(os.environ["TRAINING_RUN_DIR"]) if os.environ.get("TRAINING_RUN_DIR")
    else config.get_latest_training_run_with("audio_encoder_genre.pt")
)
if not ae_pair_path or not ae_genre_path:
    print("FEHLER: AE-Run nicht gefunden. AE_PAIR_RUN_DIR / AE_GENRE_RUN_DIR / TRAINING_RUN_DIR setzen.", flush=True)
    sys.exit(1)

test_ds = PairDataset("test", SPLIT_CSV, EMBEDDINGS_DIR)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

labels = labels_from_split_csv(SPLIT_CSV, "test", relevance_column="label", embeddings_dir=EMBEDDINGS_DIR)
vlm_scores = labels_from_split_csv(SPLIT_CSV, "test", relevance_column="vlm_score", embeddings_dir=EMBEDDINGS_DIR)

V_list, A_list = [], []
for v, a in test_loader:
    V_list.append(v)
    A_list.append(a)
V = torch.cat(V_list, dim=0).to(DEVICE)
A = torch.cat(A_list, dim=0).to(DEVICE)


def _load_ae_embeddings(run_path, subdir):
    emb_dir = run_path / subdir
    embs = [torch.tensor(np.load(emb_dir / f"{vid}.npy"), dtype=torch.float32) for vid, *_ in test_ds.samples]
    return torch.stack(embs).to(DEVICE)


A_ae_pair = _load_ae_embeddings(ae_pair_path, "audio_encoder_pair_test_embeddings")
A_ae_genre = _load_ae_embeddings(ae_genre_path, "audio_encoder_genre_test_embeddings")

video_head_ae_pair, audio_head_ae_pair = load_audio_encoder_heads_pair(ae_pair_path)
video_head_ae_genre, audio_head_ae_genre = load_audio_encoder_heads_genre(ae_genre_path)


def _sim_frozen(video, audio):
    return (F.normalize(video, p=2, dim=-1) @ F.normalize(audio, p=2, dim=-1).T).cpu()


def _sim_projected(video_head, audio_head, video, audio):
    with torch.no_grad():
        v = F.normalize(video_head(video), p=2, dim=-1)
        a = F.normalize(audio_head(audio), p=2, dim=-1)
    return (v @ a.T).cpu()


sim_baseline = _sim_frozen(V, A)
sim_e3a = _sim_projected(video_head_ae_pair, audio_head_ae_pair, V, A_ae_pair)
sim_e3b = _sim_projected(video_head_ae_genre, audio_head_ae_genre, V, A_ae_genre)

n = sim_baseline.size(0)
rel_pair = pair_relevance_matrix(n)
rel_label = label_relevance_matrix(labels)

MODELS = [
    ("baseline", "Baseline", sim_baseline),
    ("audio_encoder_pair", "E3a", sim_e3a),
    ("audio_encoder_genre", "E3b", sim_e3b),
]
PROTOCOLS = [("A", "pair", rel_pair), ("B", "label", rel_label)]
QUALITY_GROUPS = [
    ("KEEP_HIGH", torch.tensor(np.where(vlm_scores == "KEEP_HIGH")[0], dtype=torch.long)),
    ("KEEP_LOW", torch.tensor(np.where(vlm_scores == "KEEP_LOW")[0], dtype=torch.long)),
]

print(f"Dataset-Run:  {run_name}", flush=True)
print(f"AE-Pair-Run:  {ae_pair_path}", flush=True)
print(f"AE-Genre-Run: {ae_genre_path}", flush=True)
print(f"Test-Samples: {n}", flush=True)
for q_name, q_idx in QUALITY_GROUPS:
    print(f"  {q_name}: {len(q_idx)} Queries", flush=True)


def _metrics_subset(sim, rel, q_idx) -> dict:
    s = sim[q_idx]
    r = rel[q_idx]
    return {
        "mrr": float(MRR(s, relevance=r)),
        "recall_at_10": float(recall_at_k(s, 10, relevance=r)),
    }


rows: List[dict] = []
for protocol_id, protocol_key, rel in PROTOCOLS:
    for direction, sim_fn, rel_t in (("V2A", lambda s: s, rel), ("A2V", lambda s: s.T, rel.T)):
        for model_key, model_name, sim in MODELS:
            sim_d = sim_fn(sim)
            for q_name, q_idx in QUALITY_GROUPS:
                m = _metrics_subset(sim_d, rel_t, q_idx)
                rows.append({
                    "model_key": model_key,
                    "model": model_name,
                    "protocol": protocol_id,
                    "direction": direction,
                    "quality": q_name,
                    "n_queries": int(len(q_idx)),
                    "mrr": m["mrr"],
                    "recall_at_10": m["recall_at_10"],
                })


def _eval_output_dir() -> Path:
    if os.environ.get("EVAL_OUTPUT_DIR"):
        return Path(os.environ["EVAL_OUTPUT_DIR"])
    if os.environ.get("TRAINING_RUN_DIR"):
        return Path(os.environ["TRAINING_RUN_DIR"])
    return run_dir


output_dir = _eval_output_dir()
output_dir.mkdir(parents=True, exist_ok=True)
results_csv = output_dir / "results_quality_split.csv"
_fields = ["model_key", "model", "protocol", "direction", "quality", "n_queries", "mrr", "recall_at_10"]
with open(results_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=_fields)
    writer.writeheader()
    writer.writerows(rows)

print("\n=== Quality-Split Ergebnisse (MRR | R@10) ===", flush=True)
header = f"{'Model':<10} {'Prot':<5} {'Dir':<5} {'HIGH MRR':>9} {'LOW MRR':>9} {'HIGH R@10':>10} {'LOW R@10':>10}"
print(header, flush=True)
print("-" * len(header), flush=True)
by_key = {(r["model"], r["protocol"], r["direction"], r["quality"]): r for r in rows}
for _, model_name, _ in MODELS:
    for protocol_id, _, _ in PROTOCOLS:
        for direction in ("V2A", "A2V"):
            h = by_key[(model_name, protocol_id, direction, "KEEP_HIGH")]
            lo = by_key[(model_name, protocol_id, direction, "KEEP_LOW")]
            print(
                f"{model_name:<10} {protocol_id:<5} {direction:<5} "
                f"{h['mrr']:>9.3f} {lo['mrr']:>9.3f} {h['recall_at_10']:>10.3f} {lo['recall_at_10']:>10.3f}",
                flush=True,
            )

print(f"\nGespeichert: {results_csv}", flush=True)
