"""
Run: sbatch jobs/extract_and_embed_videos.sh (Default: neuester Run)
     oder DATASET_RUN_NAME=... sbatch jobs/extract_and_embed_videos.sh
"""

import os
import sys
import csv
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple

# Repo-Root für config (zentrale Pfade)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import config

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import av
import librosa
from PIL import Image
from tqdm import tqdm

# CLIP / Wav2CLIP über models (1:1 wie old/models.py)
import clip
import wav2clip
from models import load_models


WORK_ROOT = config.WORK_ROOT
DATASETS_ROOT = config.DATASETS_ROOT

TARGET_FPS = 1
TARGET_SR = 16000
MAX_FRAME_SECONDS = 10


def log(msg: str) -> None:
    print(msg, flush=True)


def frame_to_clip_embedding(frame, model, preprocess, device):
    """Ein Frame (numpy RGB) -> ein CLIP-Vektor (512), L2-normalisiert. [wie 04_embeddings_train]"""
    # Frame -> PIL -> CLIP-Preprocess -> encode_image -> normalisieren -> numpy
    pil = Image.fromarray(frame)
    tensor = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.encode_image(tensor)
        out = F.normalize(out, dim=-1)
    return out.cpu().numpy().squeeze()


def audio_to_wav2clip_embedding(audio_np, model):
    """Audio (numpy, 16 kHz) -> ein Wav2CLIP-Vektor (512). [wie 04_embeddings_train]"""
    audio_np = np.asarray(audio_np, dtype=np.float32)
    emb = wav2clip.embed_audio(audio_np, model)
    emb = F.normalize(torch.from_numpy(emb).float(), dim=-1).numpy()
    return emb.squeeze()


def build_video_list(raw_csv: Path, downloads_dir: Path) -> List[Dict[str, str]]:
    with open(raw_csv, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    video_list: List[Dict[str, str]] = []
    for row in rows:
        yt_id = row["yt_id"].strip()
        start_s = row["start_seconds"].strip()
        safe = start_s.replace(".", "p")
        mp4_path = downloads_dir / f"{yt_id}_{safe}.mp4"
        if mp4_path.exists():
            video_list.append({
                "yt_id": yt_id,
                "start_seconds": start_s,
                "video_id": f"{yt_id}_{safe}",
                "mp4_path": str(mp4_path),
                "label": row.get("label", ""),
            })
    return video_list


def extract_frames_1fps(video_path: Path, target_fps: int = 1, max_seconds: int = 10):
    """
    Extrahiert Frames mit OpenCV bei 1 FPS (eine pro Sekunde).
    Returns: list of RGB frames     
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0 or frame_count <= 0:
        return []
    duration_sec = frame_count / fps
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    for t in range(min(max_seconds, int(duration_sec))):
        frame_num = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames


def extract_audio_pyav(video_path: Path, target_sr: int = 16000):
    """
    Extrahiert Audio mit PyAV, Mono, optional auf target_sr resampled.
    Returns: (audio_np float32, sample_rate) oder (None, None) bei Fehler.
    """
    try:
        container = av.open(str(video_path))
        audio_stream = None
        for s in container.streams.audio:
            audio_stream = s
            break
        if audio_stream is None:
            container.close()
            return None, None
        audio_frames = []
        for frame in container.decode(audio_stream):
            arr = frame.to_ndarray()
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            audio_frames.append(arr)
        container.close()
        if not audio_frames:
            return None, None
        audio_np = np.concatenate(audio_frames, axis=0).astype(np.float32)
        if audio_np.max() > 1.0 or audio_np.min() < -1.0:
            audio_np = audio_np / 32768.0
        orig_sr = audio_stream.rate
        if orig_sr != target_sr:
            audio_np = librosa.resample(audio_np, orig_sr=orig_sr, target_sr=target_sr)
        return audio_np, target_sr
    except Exception as e:
        log(f"Audio-Fehler {video_path}: {e}")
        return None, None


def process_one_video(
    item: Dict[str, str],
    clip_model,
    clip_preprocess,
    wav2clip_model,
    video_dir: Path,
    audio_dir: Path,
    device,
) -> Tuple[bool, str]:
    video_id = item["video_id"]
    mp4_path = Path(item["mp4_path"])
    video_out = video_dir / f"{video_id}.npy"
    audio_out = audio_dir / f"{video_id}.npy"
    if video_out.exists() and audio_out.exists():
        return True, ""
    # Video: Encode frames with CLIP and calculate mean pool
    if not video_out.exists():
        frames = extract_frames_1fps(mp4_path, target_fps=TARGET_FPS, max_seconds=10)
        if not frames:
            return False, "frames"
        frame_embeddings = []
        for i in range(len(frames)):
            emb = frame_to_clip_embedding(frames[i], clip_model, clip_preprocess, device)
            frame_embeddings.append(emb)
        video_emb = np.array(frame_embeddings).mean(axis=0)
        video_emb = video_emb / np.linalg.norm(video_emb)
        video_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(video_out, video_emb.astype(np.float32))
    # Audio: Encode with Wav2CLIP
    if not audio_out.exists():
        audio_np, sr = extract_audio_pyav(mp4_path, target_sr=TARGET_SR)
        if audio_np is None or len(audio_np) == 0:
            return False, "audio"
        audio_emb = audio_to_wav2clip_embedding(audio_np, wav2clip_model)
        audio_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(audio_out, audio_emb.astype(np.float32))

    return True, ""


def main() -> None:
    run_name = os.environ.get("DATASET_RUN_NAME")
    if not run_name:
        run_name = config.get_latest_run_name()
        if run_name:
            log(f"DATASET_RUN_NAME nicht gesetzt → nutze neuesten Run (Default): {run_name}")
        else:
            existing = sorted([p.name for p in DATASETS_ROOT.glob("*") if p.is_dir()])
            log("FEHLER: DATASET_RUN_NAME ist nicht gesetzt und es gibt keinen Run unter DATASETS_ROOT.")
            log(f"Vorhandene Runs unter {DATASETS_ROOT}:")
            for name in existing:
                log(f"  - {name}")
            raise SystemExit(1)
    else:
        log(f"DATASET_RUN_NAME (aus Umgebung): {run_name}")

    run_dir = DATASETS_ROOT / run_name
    balanced_csv = run_dir / "segments_balanced.csv"
    downloads_dir = run_dir / "downloads"
    embeddings_dir = run_dir / "embeddings"
    video_dir = embeddings_dir / "video"
    audio_dir = embeddings_dir / "audio"

    if not balanced_csv.exists():
        log(f"FEHLER: segments_balanced.csv nicht gefunden: {balanced_csv}")
        raise SystemExit(1)
    if not downloads_dir.exists():
        log(f"FEHLER: downloads-Verzeichnis nicht gefunden: {downloads_dir}")
        raise SystemExit(1)

    video_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    log("Lade CLIP und Wav2CLIP (models.load_models) …")
    clip_model, clip_preprocess, wav2clip_model, device = load_models()

    log("=" * 60)
    log("Extraktion + Embedding (CLIP / Wav2CLIP)")
    log("=" * 60)
    log(f"RUN_NAME: {run_name}")
    log(f"RUN_DIR: {run_dir}")
    log(f"DEVICE: {device}")
    log(f"Embeddings → {embeddings_dir}")
    log(f"CSV (balanced): {balanced_csv}")

    video_list = build_video_list(balanced_csv, downloads_dir)
    log(f"Videos zu verarbeiten: {len(video_list)}")

    # Pro Genre: Anzahl ausgeben
    genre_counts = Counter(item["label"] for item in video_list)
    for genre in sorted(genre_counts.keys()):
        log(f"  {genre}: {genre_counts[genre]} Videos")

    failed: List[Tuple[str, str]] = []
    for item in tqdm(video_list, desc="Extract+Embed"):
        ok, err = process_one_video(
            item, clip_model, clip_preprocess, wav2clip_model, video_dir, audio_dir, device
        )
        if not ok:
            failed.append((item["video_id"], err))

    n_video = len(list(video_dir.glob("*.npy")))
    n_audio = len(list(audio_dir.glob("*.npy")))
    log(f"Fertig. Video-Embeddings: {n_video} | Audio-Embeddings: {n_audio}")
    if failed:
        log(f"Fehlgeschlagen: {len(failed)} (erste 10): {failed[:10]}")


if __name__ == "__main__":
    main()
