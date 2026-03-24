"""
Evaluation: MRR, Recall@1/5/10, Mean Rank. V→A und A→V.
Lädt neueste Heads aus training_runs/<Datum_Uhrzeit>/; oben: Datum, Git-Commit, verwendete Heads.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
import config
from dataset import PairDataset
from models import ProjectionHead, load_projection_heads, load_projection_heads_genre
from metrics import (
    MRR,
    recall_at_k,
    mean_rank,
    labels_from_split_csv,
    label_relevance_matrix,
    pair_relevance_matrix,
)

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
_out(f"Git-Commit (Eval): {config.get_git_commit()}{' (dirty)' if config.get_git_dirty() else ''}")
_out(f"Dataset-Run: {run_name}")
_out(f"Pair-based Head: {pair_path or config.PROJECTION_HEADS_PATH} (train commit: {_meta_commit(pair_run_dir, 'meta_pair.json') or '-'})")
_out(f"Genre-based Head: {genre_path or config.PROJECTION_HEADS_GENRE_PATH} (train commit: {_meta_commit(genre_run_dir, 'meta_genre.json') or '-'})")

test_ds = PairDataset("test", TRAIN_VAL_TEST_SPLIT_CSV, EMBEDDINGS_DIR)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
video_head_pair, audio_head_pair = load_projection_heads(pair_path)
video_head_genre, audio_head_genre = load_projection_heads_genre(genre_path)

# Relevanz = Genre-Tag (Spalte "label"), gleiche Reihenfolge wie test_ds
labels = labels_from_split_csv(
    TRAIN_VAL_TEST_SPLIT_CSV, "test", relevance_column="label", embeddings_dir=EMBEDDINGS_DIR
)
_out("Labels shape: " + str(labels.shape))

# Test-Embeddings laden (V = Video, A = Audio)
V_list, A_list = [], []
for v, a in test_loader:
    V_list.append(v)
    A_list.append(a)
V = torch.cat(V_list, dim=0).to(DEVICE)
A = torch.cat(A_list, dim=0).to(DEVICE)

# Ähnlichkeitsmatrix: sim[i,j] = Video i vs Audio j
sim_baseline = (V @ A.T).cpu()
with torch.no_grad():
    # Baseline: zufällig initialisierte Heads (gleiche Architektur wie trainiert, aber untrained)
    video_head_rand = ProjectionHead().to(DEVICE).eval()
    audio_head_rand = ProjectionHead().to(DEVICE).eval()
    v_rand = video_head_rand(V)
    a_rand = audio_head_rand(A)

    v_pair = video_head_pair(V)
    a_pair = audio_head_pair(A)
    v_genre = video_head_genre(V)
    a_genre = audio_head_genre(A)
sim_rand = (v_rand @ a_rand.T).cpu()
sim_pair = (v_pair @ a_pair.T).cpu()
sim_genre = (v_genre @ a_genre.T).cpu()

# Zwei Relevanz-Protokolle:
# 1) Pair-basiert: nur exaktes Gegenstück (Diagonale)
# 2) Label-basiert: gleiches Genre
rel_pair = pair_relevance_matrix(sim_baseline.size(0))
rel_label = label_relevance_matrix(labels)

_out("sim_baseline shape: " + str(sim_baseline.shape))
_out("sim_rand (untrained heads) shape: " + str(sim_rand.shape))
_out("sim_pair shape: " + str(sim_pair.shape))
_out("sim_genre shape: " + str(sim_genre.shape))


def print_metrics(name, sim, relevance):
    _out(f"  {name}")
    _out(
        "    MRR: " + str(MRR(sim, relevance=relevance))
        + " | Recall@1: " + str(recall_at_k(sim, 1, relevance=relevance))
        + " | Recall@5: " + str(recall_at_k(sim, 5, relevance=relevance))
        + " | Recall@10: " + str(recall_at_k(sim, 10, relevance=relevance))
        + " | Mean Rank: " + str(mean_rank(sim, relevance=relevance))
    )


_out("=== Protokoll A: Pair-basierte Relevanz (exaktes Video-Audio-Paar) ===")
_out("=== V→A (Video als Query, Audio retrieval) ===")
print_metrics("Baseline", sim_baseline, rel_pair)
print_metrics("Untrained heads", sim_rand, rel_pair)
print_metrics("Pair-based", sim_pair, rel_pair)
print_metrics("Genre-based", sim_genre, rel_pair)
_out("")
_out("=== A→V (Audio als Query, Video retrieval) ===")
print_metrics("Baseline", sim_baseline.T, rel_pair.T)
print_metrics("Untrained heads", sim_rand.T, rel_pair.T)
print_metrics("Pair-based", sim_pair.T, rel_pair.T)
print_metrics("Genre-based", sim_genre.T, rel_pair.T)
_out("")
_out("=== Protokoll B: Label-basierte Relevanz (gleiches Genre) ===")
_out("=== V→A (Video als Query, Audio retrieval) ===")
print_metrics("Baseline", sim_baseline, rel_label)
print_metrics("Untrained heads", sim_rand, rel_label)
print_metrics("Pair-based", sim_pair, rel_label)
print_metrics("Genre-based", sim_genre, rel_label)
_out("")
_out("=== A→V (Audio als Query, Video retrieval) ===")
print_metrics("Baseline", sim_baseline.T, rel_label.T)
print_metrics("Untrained heads", sim_rand.T, rel_label.T)
print_metrics("Pair-based", sim_pair.T, rel_label.T)
print_metrics("Genre-based", sim_genre.T, rel_label.T)

# Bei Pipeline: Evaluation-Ausgabe in denselben Run-Ordner schreiben
if os.environ.get("TRAINING_RUN_DIR"):
    out_path = Path(os.environ["TRAINING_RUN_DIR"]) / "evaluation_output.txt"
    _out_lines.append("")
    _out_lines.append("Gespeichert: " + str(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(_out_lines))
    print("", flush=True)
    print("Gespeichert: " + str(out_path), flush=True)
