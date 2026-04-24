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


def load_wav2clip_finetune(device=None):
    """Lädt Wav2CLIP mit trainierbaren Gewichten (scenario='finetune')."""
    from wav2clip import MODEL_URL
    from wav2clip.model.encoder import ResNetExtractor
    _device = device or DEVICE
    checkpoint = torch.hub.load_state_dict_from_url(MODEL_URL, map_location=_device, progress=True)
    model = ResNetExtractor(checkpoint=checkpoint, scenario="finetune", transform=True)
    model.to(_device)
    return model


class ProjectionHead(nn.Module):
    """
    Projection-Head:
    - linear: Linear(in_dim, out_dim)
    - mlp: Linear(in_dim, hidden_dim) + ReLU + Linear(hidden_dim, out_dim)
    CLIP/Wav2CLIP liefern 512-dim;
    """
    def __init__(self, in_dim=512, out_dim=64, head_type="linear", hidden_dim=256):
        super().__init__()
        self.head_type = head_type
        if head_type == "linear":
            self.proj = nn.Linear(in_dim, out_dim)
        elif head_type == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            raise ValueError(f"Unbekannter head_type: {head_type}")

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


def _infer_head_config_from_state_dict(state_dict):
    """Leitet Head-Architektur und Dimensionen aus einem gespeicherten state_dict ab."""
    if "proj.weight" in state_dict:
        w = state_dict["proj.weight"]
        return {
            "head_type": "linear",
            "in_dim": int(w.shape[1]),
            "out_dim": int(w.shape[0]),
            "hidden_dim": 256,
        }
    if "proj.0.weight" in state_dict and "proj.2.weight" in state_dict:
        w1 = state_dict["proj.0.weight"]
        w2 = state_dict["proj.2.weight"]
        return {
            "head_type": "mlp",
            "in_dim": int(w1.shape[1]),
            "out_dim": int(w2.shape[0]),
            "hidden_dim": int(w1.shape[0]),
        }
    raise ValueError("Konnte Head-Architektur aus Checkpoint nicht ableiten.")


def load_projection_heads(path=None, device=None):
    """Lädt Pair-Heads aus path (Ordner/Datei) oder PROJECTION_HEADS_PATH. Gibt (video_head, audio_head) zurück."""
    ckpt_path = _checkpoint_path(path, PROJECTION_HEADS_PATH, "projection_heads_pair.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    cfg = _infer_head_config_from_state_dict(ckpt["video_head"])
    video_head = ProjectionHead(**cfg).to(DEVICE)
    audio_head = ProjectionHead(**cfg).to(DEVICE)
    video_head.load_state_dict(ckpt["video_head"])
    audio_head.load_state_dict(ckpt["audio_head"])
    video_head.eval()
    audio_head.eval()
    return video_head, audio_head


def load_projection_heads_genre(path=None, device=None):
    """Lädt Genre-Heads aus path (Ordner/Datei) oder PROJECTION_HEADS_GENRE_PATH. Gibt (video_head, audio_head) zurück."""
    ckpt_path = _checkpoint_path(path, PROJECTION_HEADS_GENRE_PATH, "projection_heads_genre.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    cfg = _infer_head_config_from_state_dict(ckpt["video_head"])
    video_head = ProjectionHead(**cfg).to(DEVICE)
    audio_head = ProjectionHead(**cfg).to(DEVICE)
    video_head.load_state_dict(ckpt["video_head"])
    audio_head.load_state_dict(ckpt["audio_head"])
    video_head.eval()
    audio_head.eval()
    return video_head, audio_head
