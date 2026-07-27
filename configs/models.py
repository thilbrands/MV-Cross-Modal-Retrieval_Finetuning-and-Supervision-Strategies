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


_UNFREEZE_MODES = frozenset({"full", "layer4_transform", "layer3_4_transform"})


def configure_wav2clip_trainable(model, unfreeze: str = "layer4_transform") -> None:
    """
    Steuert, welche Wav2CLIP-Teile trainierbar sind (ResNet-18 + MLP-Transform).

    unfreeze:
      - 'full': gesamter Encoder + Transform
      - 'layer4_transform': encoder.layer4 + transform
      - 'layer3_4_transform': encoder.layer3 + layer4 + transform
    """
    if unfreeze not in _UNFREEZE_MODES:
        raise ValueError(f"Unbekannter unfreeze-Modus: {unfreeze!r} (erlaubt: {sorted(_UNFREEZE_MODES)})")

    for p in model.parameters():
        p.requires_grad = False

    if unfreeze == "full":
        for p in model.parameters():
            p.requires_grad = True
        return

    if unfreeze in {"layer3_4_transform", "layer4_transform"}:
        if unfreeze == "layer3_4_transform":
            for p in model.encoder.layer3.parameters():
                p.requires_grad = True
        for p in model.encoder.layer4.parameters():
            p.requires_grad = True
    if model.transform is not None:
        for p in model.transform.parameters():
            p.requires_grad = True


def default_encoder_lr(head_lr: float, unfreeze: str) -> float:
    """Default LR für Wav2CLIP je nach Unfreeze-Tiefe (ohne HP_LR_ENCODER)."""
    if unfreeze == "full":
        return head_lr / 10
    if unfreeze == "layer3_4_transform":
        return head_lr / 5
    return head_lr / 3


def wav2clip_trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def count_wav2clip_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_wav2clip_finetune(device=None, unfreeze: str = "layer4_transform"):
    """Lädt Wav2CLIP (scenario='finetune', transform=True) mit konfigurierbarem Partial Unfreeze."""
    from wav2clip import MODEL_URL
    from wav2clip.model.encoder import ResNetExtractor
    _device = device or DEVICE
    checkpoint = torch.hub.load_state_dict_from_url(MODEL_URL, map_location=_device, progress=True)
    model = ResNetExtractor(checkpoint=checkpoint, scenario="finetune", transform=True)
    configure_wav2clip_trainable(model, unfreeze=unfreeze)
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
    """path=None → default_path; path=Ordner → path/filename oder path/checkpoints/filename; sonst Datei."""
    if path is None:
        return default_path
    p = Path(path)
    if p.is_dir():
        direct = p / filename_in_dir
        nested = p / "checkpoints" / filename_in_dir
        if direct.exists():
            return direct
        if nested.exists():
            return nested
        return direct
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


def _load_heads(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    cfg = _infer_head_config_from_state_dict(ckpt["video_head"])
    video_head = ProjectionHead(**cfg).to(DEVICE)
    audio_head = ProjectionHead(**cfg).to(DEVICE)
    video_head.load_state_dict(ckpt["video_head"])
    audio_head.load_state_dict(ckpt["audio_head"])
    video_head.eval()
    audio_head.eval()
    return video_head, audio_head


def load_projection_heads_pair(path=None, device=None):
    """Lädt Pair-Heads aus path (Ordner/Datei) oder PROJECTION_HEADS_PATH. Gibt (video_head, audio_head) zurück."""
    return _load_heads(_checkpoint_path(path, PROJECTION_HEADS_PATH, "projection_heads_pair.pt"))


def load_projection_heads_genre(path=None, device=None):
    """Lädt Genre-Heads aus path (Ordner/Datei) oder PROJECTION_HEADS_GENRE_PATH. Gibt (video_head, audio_head) zurück."""
    return _load_heads(_checkpoint_path(path, PROJECTION_HEADS_GENRE_PATH, "projection_heads_genre.pt"))


def load_audio_encoder_heads_pair(path, device=None):
    """Lädt Heads aus einem Pair-Audio-Encoder-Checkpoint. Gibt (video_head, audio_head) zurück."""
    return _load_heads(_checkpoint_path(path, None, "audio_encoder_pair.pt"))


def load_audio_encoder_heads_genre(path, device=None):
    """Lädt Heads aus einem Genre-Audio-Encoder-Checkpoint. Gibt (video_head, audio_head) zurück."""
    return _load_heads(_checkpoint_path(path, None, "audio_encoder_genre.pt"))
