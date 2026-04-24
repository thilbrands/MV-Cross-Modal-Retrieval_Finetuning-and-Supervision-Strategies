"""
Einmalig: Extrahiert rohe Audiowaveforms für bestehende Datensätze.
Speichert sie nach embeddings/audio_raw/{video_id}.npy (16 kHz, mono, float32).
Wird benötigt für das Audio-Encoder-Training.

Nutzung:
  python3 pipeline/extract_raw_audio.py
  DATASET_RUN_NAME=... python3 pipeline/extract_raw_audio.py
"""
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
import config

import numpy as np
from tqdm import tqdm

from pipeline.extract_and_embed_videos import (
    build_video_list,
    extract_audio_pyav,
    TARGET_SR,
)


def main():
    run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
    if not run_name:
        print("FEHLER: DATASET_RUN_NAME nicht gesetzt.", flush=True)
        sys.exit(1)

    run_dir = config.DATASETS_ROOT / run_name
    audio_raw_dir = run_dir / "embeddings" / "audio_raw"
    audio_raw_dir.mkdir(parents=True, exist_ok=True)

    video_list = build_video_list(run_dir / "segments_balanced.csv", run_dir / "downloads")
    print(f"Videos: {len(video_list)} | Output: {audio_raw_dir}", flush=True)

    skipped, failed = 0, 0
    for item in tqdm(video_list, desc="extract_raw_audio"):
        out = audio_raw_dir / f"{item['video_id']}.npy"
        if out.exists():
            skipped += 1
            continue
        audio_np, _ = extract_audio_pyav(Path(item["mp4_path"]), target_sr=TARGET_SR)
        if audio_np is None or len(audio_np) == 0:
            failed += 1
            continue
        np.save(out, audio_np.astype(np.float32))

    print(f"Fertig. Gespeichert: {len(video_list) - skipped - failed} | Übersprungen: {skipped} | Fehler: {failed}", flush=True)


if __name__ == "__main__":
    main()
