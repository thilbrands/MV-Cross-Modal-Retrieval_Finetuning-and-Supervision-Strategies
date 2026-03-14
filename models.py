"""
Modell-Definitionen und Lade-Funktionen (1:1 aus old/models.py).
Pipeline: from models import load_models, ProjectionHead, load_projection_heads
"""
import torch
import torch.nn as nn

from config import DEVICE, PROJECTION_HEADS_PATH


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
    """2-Layer-MLP: Linear(512, 512) -> ReLU -> Linear(512, 512)."""
    def __init__(self, in_dim=512, out_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)


def load_projection_heads(device=None):
    """Lädt die trainierten Projektions-Heads aus PROJECTION_HEADS_PATH. Gibt (video_head, audio_head) zurück."""
    video_head = ProjectionHead().to(DEVICE)
    audio_head = ProjectionHead().to(DEVICE)
    ckpt = torch.load(PROJECTION_HEADS_PATH, map_location=DEVICE)
    video_head.load_state_dict(ckpt["video_head"])
    audio_head.load_state_dict(ckpt["audio_head"])
    video_head.eval()
    audio_head.eval()
    return video_head, audio_head
