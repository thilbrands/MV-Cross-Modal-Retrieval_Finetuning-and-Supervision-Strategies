"""
Genre-Confusion-Matrix für Cross-Modal Retrieval (V→A, Top-1).
Für jede Video-Query wird der Top-1-Audio-Kandidat per Cosine Similarity retrieved;
aggregiert in einer 10×10-Matrix (Zeile = Query-Genre Video, Spalte = retrieved Audio-Genre).
Zeilen-normalisiert (Prozent).

Zwei Panels: (a) Baseline (frozen), (b) E3a (Audio-Encoder Pair).
Speichert genre_confusion.pdf/.png im Dataset-Run-Ordner.

Run: python3 plot_genre_confusion.py
     TRAINING_RUN_DIR=... AE_PAIR_RUN_DIR=... python3 plot_genre_confusion.py
"""
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
import config
from models import load_audio_encoder_heads_pair

GENRES = [
    "Blues", "Classical music", "Country", "Electronic music", "Funk",
    "Hip hop music", "Jazz", "Pop music", "Reggae", "Rock music",
]
GENRE_LABELS = ["Blues", "Classical", "Country", "Electronic", "Funk",
                "Hip hop", "Jazz", "Pop", "Reggae", "Rock"]

run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
if not run_name:
    print("FEHLER: Kein Dataset-Run gefunden.", flush=True)
    sys.exit(1)

run_dir = config.DATASETS_ROOT / run_name
split_csv = run_dir / "train_val_test_split.csv"
embeddings_dir = run_dir / "embeddings"

ae_pair_path = (
    Path(os.environ["AE_PAIR_RUN_DIR"]) if os.environ.get("AE_PAIR_RUN_DIR")
    else Path(os.environ["TRAINING_RUN_DIR"]) if os.environ.get("TRAINING_RUN_DIR")
    else config.get_latest_training_run_with("audio_encoder_pair.pt")
)
if not ae_pair_path:
    print("FEHLER: AE-Pair-Run nicht gefunden. AE_PAIR_RUN_DIR / TRAINING_RUN_DIR setzen.", flush=True)
    sys.exit(1)

ae_pair_emb_dir = ae_pair_path / "audio_encoder_pair_test_embeddings"
for label, p in [
    ("Dataset embeddings/video", embeddings_dir / "video"),
    ("Dataset embeddings/audio", embeddings_dir / "audio"),
    ("AE-Pair Test-Embeddings", ae_pair_emb_dir),
    ("Split-CSV", split_csv),
]:
    if not p.exists():
        print(f"FEHLER: {label} nicht gefunden: {p}", flush=True)
        sys.exit(1)

samples = []
with open(split_csv, "r", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["split"].strip() != "test":
            continue
        video_id = row["video_id"].strip()
        label = row["label"].strip()
        v_path = embeddings_dir / "video" / f"{video_id}.npy"
        a_path = embeddings_dir / "audio" / f"{video_id}.npy"
        ae_pair_emb = ae_pair_emb_dir / f"{video_id}.npy"
        if v_path.exists() and a_path.exists() and ae_pair_emb.exists():
            samples.append((video_id, label, v_path, a_path, ae_pair_emb))

if not samples:
    print("FEHLER: Keine Test-Samples mit allen Embeddings gefunden.", flush=True)
    sys.exit(1)

print(f"Dataset-Run: {run_name}", flush=True)
print(f"AE-Pair-Run: {ae_pair_path}", flush=True)
print(f"Test-Samples: {len(samples)}", flush=True)

genre_to_idx = {g: i for i, g in enumerate(GENRES)}
labels = [s[1] for s in samples]
missing = sorted(set(labels) - set(GENRES))
if missing:
    print(f"WARNUNG: Labels nicht in GENRES-Liste: {missing}", flush=True)
label_idx = np.array([genre_to_idx.get(l, -1) for l in labels])

V = torch.tensor(np.stack([np.load(s[2]) for s in samples]), dtype=torch.float32, device=config.DEVICE)
A = torch.tensor(np.stack([np.load(s[3]) for s in samples]), dtype=torch.float32, device=config.DEVICE)
A_ae_pair = torch.tensor(np.stack([np.load(s[4]) for s in samples]), dtype=torch.float32, device=config.DEVICE)

video_head_ae_pair, audio_head_ae_pair = load_audio_encoder_heads_pair(ae_pair_path)


def _top1_audio_idx(video, audio, video_head=None, audio_head=None):
    with torch.no_grad():
        v = video_head(video) if video_head is not None else video
        a = audio_head(audio) if audio_head is not None else audio
        vn = F.normalize(v, p=2, dim=-1)
        an = F.normalize(a, p=2, dim=-1)
        sim = vn @ an.T
        return sim.argmax(dim=1).cpu().numpy()


def _confusion(top1_idx):
    n_genres = len(GENRES)
    counts = np.zeros((n_genres, n_genres), dtype=np.float64)
    for query_i, audio_j in enumerate(top1_idx):
        qg = label_idx[query_i]
        ag = label_idx[audio_j]
        if qg < 0 or ag < 0:
            continue
        counts[qg, ag] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return 100.0 * counts / row_sums


cm_baseline = _confusion(_top1_audio_idx(V, A))
cm_e3a = _confusion(_top1_audio_idx(V, A_ae_pair, video_head_ae_pair, audio_head_ae_pair))

for title, cm in [("Baseline", cm_baseline), ("E3a", cm_e3a)]:
    diag = np.mean([cm[i, i] for i in range(len(GENRES))])
    print(f"{title}: mittlere Diagonale (korrekt) = {diag:.1f}%", flush=True)

fig, (ax1, ax2, cax) = plt.subplots(
    1, 3, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1, 1, 0.04]}
)
heatmap_kwargs = dict(
    vmin=0, vmax=100, cmap="YlOrRd", annot=True, fmt=".0f",
    annot_kws={"size": 6}, linewidths=0.4, linecolor="white",
    xticklabels=GENRE_LABELS, yticklabels=GENRE_LABELS,
)

sns.heatmap(cm_baseline, ax=ax1, cbar=False, **heatmap_kwargs)
sns.heatmap(cm_e3a, ax=ax2, cbar=True, cbar_ax=cax, **heatmap_kwargs)

for ax, title in [(ax1, "(a) Baseline"), (ax2, "(b) E3a")]:
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Retrieved audio genre", fontsize=10)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

ax1.set_ylabel("Query genre (video)", fontsize=10)
ax2.set_ylabel("")
cax.tick_params(labelsize=8)
cax.set_ylabel("Row %", fontsize=9)

fig.tight_layout()
out_base = run_dir / "genre_confusion"
fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Gespeichert: {out_base}.pdf", flush=True)
print(f"Gespeichert: {out_base}.png", flush=True)
