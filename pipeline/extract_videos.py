import os
import csv
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import cv2
import av
import librosa
import soundfile as sf
from tqdm import tqdm


"""
Extraktion von Frames (1 FPS) und Audio für alle Videos in einem Dataset-Run.

- Erwartet einen bestehenden Downloader-Run unter:
  /work2/ra39oxet-DatasetAudioSetSubset/datasets/<RUN_NAME>
- Nutzt:
  - segments_raw.csv        (alle Segmente)
  - downloads/              (alle MP4s)
- Schreibt in denselben Run-Ordner:
  - frames/                 (pro Video: <video_id>.npy)
  - audio/                  (pro Video: <video_id>.wav)

- Run-Auswahl:
  - Environment-Variable DATASET_RUN_NAME muss gesetzt sein,
    z.B.: DATASET_RUN_NAME=2026-03-13_18-02-30_audioset

- Ausführung:
  - sbatch --export=DATASET_RUN_NAME=... jobs/extract_videos.sh
"""


WORK_ROOT = Path("/work2/ra39oxet-DatasetAudioSetSubset")
DATASETS_ROOT = WORK_ROOT / "datasets"

# Extraktions-Parameter (wie im Notebook)
TARGET_FPS = 1
TARGET_SR = 16000


def log(msg: str) -> None:
    print(msg, flush=True)


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
            video_list.append(
                {
                    "yt_id": yt_id,
                    "start_seconds": start_s,
                    "video_id": f"{yt_id}_{safe}",
                    "mp4_path": str(mp4_path),
                    "label": row.get("label", ""),
                }
            )
    return video_list


def extract_frames_1fps(
    video_path: Path, target_fps: int = TARGET_FPS, max_seconds: int = 10
) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0 or frame_count <= 0:
        return []
    duration_sec = frame_count / fps
    frames: List[np.ndarray] = []
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


def extract_audio_pyav(
    video_path: Path, target_sr: int = TARGET_SR
) -> Tuple[np.ndarray | None, int | None]:
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


def main() -> None:
    run_name = os.environ.get("DATASET_RUN_NAME")
    if not run_name:
        existing = sorted(
            [p.name for p in DATASETS_ROOT.glob("*") if p.is_dir()]
        )
        log("FEHLER: DATASET_RUN_NAME ist nicht gesetzt.")
        log(f"Vorhandene Runs unter {DATASETS_ROOT}:")
        for name in existing:
            log(f"  - {name}")
        raise SystemExit(1)

    run_dir = DATASETS_ROOT / run_name
    raw_csv = run_dir / "segments_raw.csv"
    downloads_dir = run_dir / "downloads"
    frames_dir = run_dir / "frames"
    audio_dir = run_dir / "audio"

    if not raw_csv.exists():
        log(f"FEHLER: segments_raw.csv nicht gefunden: {raw_csv}")
        raise SystemExit(1)
    if not downloads_dir.exists():
        log(f"FEHLER: downloads-Verzeichnis nicht gefunden: {downloads_dir}")
        raise SystemExit(1)

    frames_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("Extraktion (Frames + Audio) gestartet")
    log("=" * 60)
    log(f"RUN_NAME: {run_name}")
    log(f"RUN_DIR: {run_dir}")
    log(f"RAW_CSV: {raw_csv}")
    log(f"DOWNLOADS: {downloads_dir}")
    log(f"FRAMES_DIR: {frames_dir}")
    log(f"AUDIO_DIR: {audio_dir}")

    video_list = build_video_list(raw_csv, downloads_dir)
    log(f"Videos zum Extrahieren: {len(video_list)}")

    failed: List[Tuple[str, str]] = []
    for item in tqdm(video_list, desc="Extraktion"):
        video_id = item["video_id"]
        mp4_path = Path(item["mp4_path"])
        frames_out = frames_dir / f"{video_id}.npy"
        audio_out = audio_dir / f"{video_id}.wav"

        # Frames
        if not frames_out.exists():
            frames = extract_frames_1fps(mp4_path)
            if frames:
                np.save(frames_out, np.stack(frames), allow_pickle=False)
            else:
                failed.append((video_id, "frames"))

        # Audio
        if not audio_out.exists():
            audio_np, sr = extract_audio_pyav(mp4_path)
            if audio_np is not None and len(audio_np) > 0:
                sf.write(audio_out, audio_np, sr)
            else:
                failed.append((video_id, "audio"))

    num_frames = len(list(frames_dir.glob("*.npy")))
    num_audio = len(list(audio_dir.glob("*.wav")))
    log(f"Fertig. Frames: {num_frames} | Audio: {num_audio}")
    if failed:
        log(f"Fehlgeschlagen: {len(failed)} (erste 10): {failed[:10]}")


if __name__ == "__main__":
    main()

