"""
Datasets für Training/Evaluation.

PairDataset        — lädt pre-computed Video- und Audio-Embeddings (head-only Training)
RawAudioPairDataset — lädt pre-computed Video-Embeddings + rohe Audiowaveforms (Audio-Encoder-Training)
"""
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PairDataset(Dataset):
    """Lädt Video-/Audio-Embeddings pro Split-Zeile; optional mit Genre-Label."""

    def __init__(self, split_name: str, split_csv: Path, embeddings_dir: Path, return_label: bool = False):
        self.embeddings_dir = Path(embeddings_dir)
        self.return_label = return_label
        self.samples = []
        with open(split_csv, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("split", "").strip() != split_name:
                    continue
                video_id = row.get("video_id", "").strip()
                if not video_id:
                    continue
                label = row.get("label", "").strip()
                v_path = self.embeddings_dir / "video" / f"{video_id}.npy"
                a_path = self.embeddings_dir / "audio" / f"{video_id}.npy"
                if v_path.exists() and a_path.exists():
                    self.samples.append((video_id, v_path, a_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, v_path, a_path, label = self.samples[idx]
        v = np.load(v_path)
        a = np.load(a_path)
        v_t = torch.tensor(v, dtype=torch.float32)
        a_t = torch.tensor(a, dtype=torch.float32)
        if self.return_label:
            return v_t, a_t, label
        return v_t, a_t


class RawAudioPairDataset(Dataset):
    """
    Lädt pre-computed Video-Embeddings + rohe Audiowaveforms pro Split.
    Wird für das Audio-Encoder-Training verwendet.
    """

    def __init__(self, split_name: str, split_csv: Path, embeddings_dir: Path, return_label: bool = False):
        self.embeddings_dir = Path(embeddings_dir)
        self.return_label = return_label
        self.samples = []
        with open(split_csv, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("split", "").strip() != split_name:
                    continue
                video_id = row.get("video_id", "").strip()
                if not video_id:
                    continue
                label = row.get("label", "").strip()
                v_path = self.embeddings_dir / "video" / f"{video_id}.npy"
                a_path = self.embeddings_dir / "audio_raw" / f"{video_id}.npy"
                if v_path.exists() and a_path.exists():
                    self.samples.append((video_id, v_path, a_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, v_path, a_path, label = self.samples[idx]
        v_t = torch.tensor(np.load(v_path), dtype=torch.float32)
        a = np.load(a_path)[:160000]
        a_t = torch.tensor(np.pad(a, (0, 160000 - len(a))), dtype=torch.float32)
        if self.return_label:
            return v_t, a_t, label
        return v_t, a_t
