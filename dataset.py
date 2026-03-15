"""
Pair-Dataset für Training/Evaluation: lädt Video- und Audio-Embeddings pro Split.
Logik wie in den Notebooks (07_training, 09_evaluation): Split-CSV + EMBEDDINGS_DIR.
"""
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PairDataset(Dataset):
    """Lädt (video_emb, audio_emb) für alle Zeilen der Split-CSV mit split=split_name."""

    def __init__(self, split_name: str, split_csv: Path, embeddings_dir: Path):
        self.embeddings_dir = Path(embeddings_dir)
        self.samples = []
        with open(split_csv, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("split", "").strip() != split_name:
                    continue
                video_id = row.get("video_id", "").strip()
                if not video_id:
                    continue
                v_path = self.embeddings_dir / "video" / f"{video_id}.npy"
                a_path = self.embeddings_dir / "audio" / f"{video_id}.npy"
                if v_path.exists() and a_path.exists():
                    self.samples.append((video_id, v_path, a_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, v_path, a_path = self.samples[idx]
        v = np.load(v_path)
        a = np.load(a_path)
        return torch.tensor(v, dtype=torch.float32), torch.tensor(a, dtype=torch.float32)
