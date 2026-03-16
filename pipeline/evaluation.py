"""
Evaluation: MRR, Recall@1/5/10, Mean Rank. V→A und A→V. Logik 1:1 aus old/09_evaluation.ipynb.
Run: DATASET_RUN_NAME oder neuester Run. Projektions-Heads aus PROJECTION_HEADS_PATH.
"""
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
import config
from dataset import PairDataset
from models import load_projection_heads, load_projection_heads_genre
from metrics import MRR, recall_at_k, mean_rank, labels_from_split_csv

# Run: aus Umgebung oder neuester
run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
if not run_name:
    print("FEHLER: DATASET_RUN_NAME nicht gesetzt und kein Run unter DATASETS_ROOT.", flush=True)
    sys.exit(1)
run_dir = config.DATASETS_ROOT / run_name
EMBEDDINGS_DIR = run_dir / "embeddings"
TRAIN_VAL_TEST_SPLIT_CSV = run_dir / "train_val_test_split.csv"
PROJECTION_HEADS_PATH = config.PROJECTION_HEADS_PATH
DEVICE = config.DEVICE

test_ds = PairDataset("test", TRAIN_VAL_TEST_SPLIT_CSV, EMBEDDINGS_DIR)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
video_head_pair, audio_head_pair = load_projection_heads()
video_head_genre, audio_head_genre = load_projection_heads_genre()

# Relevanz = Genre-Tag (Spalte "label"), gleiche Reihenfolge wie test_ds
labels = labels_from_split_csv(
    TRAIN_VAL_TEST_SPLIT_CSV, "test", relevance_column="label", embeddings_dir=EMBEDDINGS_DIR
)
print("Labels shape:", labels.shape)

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
    v_pair = video_head_pair(V)
    a_pair = audio_head_pair(A)
    v_genre = video_head_genre(V)
    a_genre = audio_head_genre(A)
sim_pair = (v_pair @ a_pair.T).cpu()
sim_genre = (v_genre @ a_genre.T).cpu()

print("sim_baseline shape:", sim_baseline.shape)
print("sim_pair shape:", sim_pair.shape)
print("sim_genre shape:", sim_genre.shape)


def print_metrics(name, sim, labels):
    print(f"  {name}")
    print("    MRR:", MRR(sim, labels), "| Recall@1:", recall_at_k(sim, 1, labels),
          "| Recall@5:", recall_at_k(sim, 5, labels), "| Recall@10:", recall_at_k(sim, 10, labels),
          "| Mean Rank:", mean_rank(sim, labels))


print("=== V→A (Video als Query, Audio retrieval) ===")
print_metrics("Baseline", sim_baseline, labels)
print_metrics("Pair-based", sim_pair, labels)
print_metrics("Genre-based", sim_genre, labels)
print()
print("=== A→V (Audio als Query, Video retrieval) ===")
print_metrics("Baseline", sim_baseline.T, labels)
print_metrics("Pair-based", sim_pair.T, labels)
print_metrics("Genre-based", sim_genre.T, labels)
