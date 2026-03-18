"""
Modell-Definitionen und Lade-Funktionen (1:1 aus old/models.py).
Pipeline: from models import load_models, ProjectionHead, load_projection_heads
"""
from pathlib import Path

import torch
import torch.nn as nn

from config import DEVICE, PROJECTION_HEADS_PATH, PROJECTION_HEADS_GENRE_PATH


def load_models(device=None):
    """Lädt CLIP (ViT-B/32) und Wav2CLIP, frozen. Gibt (clip_model, clip_preprocess, wav2clip_model, device) zurück."""
    import clip
    import wav2clip
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
    wav2clip_model = wav2clip.get_model().to(DEVICE)
    clip_model.eval()
    wav2clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    for p in wav2clip_model.parameters():
        p.requires_grad = False
    return clip_model, clip_preprocess, wav2clip_model, DEVICE


class ProjectionHead(nn.Module):
    """
    Linearer Projection-Head: Linear(in_dim, out_dim).
    CLIP/Wav2CLIP liefern 512-dim;
    """
    def __init__(self, in_dim=512, out_dim=64):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)


def _checkpoint_path(path, default_path, filename_in_dir):
    """path=None → default_path; path=Ordner → path/filename_in_dir; sonst path als Datei."""
    if path is None:
        return default_path
    p = Path(path)
    if p.is_dir():
        return p / filename_in_dir
    return p


def load_projection_heads(path=None, device=None):
    """Lädt Pair-Heads aus path (Ordner/Datei) oder PROJECTION_HEADS_PATH. Gibt (video_head, audio_head) zurück."""
    ckpt_path = _checkpoint_path(path, PROJECTION_HEADS_PATH, "projection_heads_pair.pt")
    video_head = ProjectionHead().to(DEVICE)
    audio_head = ProjectionHead().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    video_head.load_state_dict(ckpt["video_head"])
    audio_head.load_state_dict(ckpt["audio_head"])
    video_head.eval()
    audio_head.eval()
    return video_head, audio_head


def load_projection_heads_genre(path=None, device=None):
    """Lädt Genre-Heads aus path (Ordner/Datei) oder PROJECTION_HEADS_GENRE_PATH. Gibt (video_head, audio_head) zurück."""
    ckpt_path = _checkpoint_path(path, PROJECTION_HEADS_GENRE_PATH, "projection_heads_genre.pt")
    video_head = ProjectionHead().to(DEVICE)
    audio_head = ProjectionHead().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    video_head.load_state_dict(ckpt["video_head"])
    audio_head.load_state_dict(ckpt["audio_head"])
    video_head.eval()
    audio_head.eval()
    return video_head, audio_head
