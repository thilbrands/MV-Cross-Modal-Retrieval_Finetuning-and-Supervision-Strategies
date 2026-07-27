#!/usr/bin/env python3
"""
Exportiert Frame-Strips für zufällige KEEP_HIGH-Segmente (wie im VLM-Filter).

Standard (Cluster, Repo-Root): neuester Dataset-Run unter config.DATASETS_ROOT.
Output: <run_dir>/keep_high_examples/

Voraussetzung: Venv „ba“ (opencv-python), wie bei extract_and_embed:
  module load Python/3.11.5-GCCcore-13.2.0
  source ~/venv/ba/bin/activate
  python3 figures/export_keep_high_frame_examples.py

Oder: sbatch jobs/export_keep_high_examples.sh
"""
from __future__ import annotations

import csv
import os
import random
import sys
from pathlib import Path

import cv2
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "configs"))
import config  # noqa: E402

N_EXAMPLES = int(os.environ.get("N_EXAMPLES", "10"))
SEED = int(os.environ.get("SEED", "42"))
TARGET_FPS = 1
MAX_FRAME_SECONDS = 10


def extract_frames_1fps(video_path: Path, max_seconds: int = MAX_FRAME_SECONDS) -> list:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return []
    frames = []
    for t in range(max_seconds):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def choose_vlm_frames(frames_rgb: list) -> list[Image.Image]:
    n = len(frames_rgb)
    if n < 3:
        return []
    idxs = [0, 2, 4, 6, 8] if n >= 9 else [0, n // 2, n - 1]
    return [Image.fromarray(frames_rgb[i].astype("uint8")) for i in idxs]


def save_strip(images: list[Image.Image], out_path: Path, thumb_h: int = 180) -> None:
    resized = []
    for img in images:
        w = max(1, int(img.width * thumb_h / img.height))
        resized.append(img.resize((w, thumb_h), Image.Resampling.LANCZOS))
    total_w = sum(i.width for i in resized)
    strip = Image.new("RGB", (total_w, thumb_h))
    x = 0
    for img in resized:
        strip.paste(img, (x, 0))
        x += img.width
    out_path.parent.mkdir(parents=True, exist_ok=True)
    strip.save(out_path)


def resolve_run_dir() -> Path:
    run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
    if not run_name:
        print("FEHLER: DATASET_RUN_NAME nicht gesetzt und kein Run unter DATASETS_ROOT.", file=sys.stderr)
        sys.exit(1)
    return config.DATASETS_ROOT / run_name


def main() -> None:
    run_dir = resolve_run_dir()
    scored_csv = Path(os.environ.get("SCORED_CSV", str(run_dir / "segments_unbalanced_vlm_scored.csv")))
    downloads = Path(os.environ.get("DOWNLOADS_DIR", str(run_dir / "downloads")))
    output_dir = Path(
        os.environ.get(
            "OUTPUT_DIR",
            str(config.resolve_plot_output_dir(run_dir) / "keep_high_examples"),
        )
    )

    if not scored_csv.is_file():
        print(f"FEHLER: {scored_csv} nicht gefunden.", file=sys.stderr)
        sys.exit(1)
    if not downloads.is_dir():
        print(f"FEHLER: {downloads} nicht gefunden.", file=sys.stderr)
        sys.exit(1)

    print(f"Dataset-Run: {run_dir.name}", flush=True)
    print(f"CSV:         {scored_csv}", flush=True)
    print(f"Downloads:   {downloads}", flush=True)
    print(f"Output:      {output_dir}", flush=True)

    keep_high_rows = []
    with open(scored_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("vlm_score", "").strip() == "KEEP_HIGH":
                keep_high_rows.append(row)

    if not keep_high_rows:
        print(f"Keine KEEP_HIGH-Zeilen in {scored_csv}", file=sys.stderr)
        sys.exit(1)

    random.seed(SEED)
    random.shuffle(keep_high_rows)
    picked = keep_high_rows[:N_EXAMPLES]

    manifest_path = output_dir / "keep_high_examples_manifest.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["index", "video_id", "label", "vlm_score", "strip_png", "mp4_path"],
        )
        w.writeheader()

        saved = 0
        for i, row in enumerate(picked, start=1):
            yt_id = row["yt_id"].strip()
            start_s = row["start_seconds"].strip()
            video_id = f"{yt_id}_{start_s.replace('.', 'p')}"
            mp4 = downloads / f"{video_id}.mp4"
            if not mp4.is_file():
                print(f"Überspringe (keine MP4): {video_id}", flush=True)
                continue

            frames = extract_frames_1fps(mp4)
            pil_frames = choose_vlm_frames(frames)
            if not pil_frames:
                print(f"Überspringe (<3 Frames): {video_id}", flush=True)
                continue

            strip_path = output_dir / f"{i:02d}_{video_id}_keep_high.png"
            save_strip(pil_frames, strip_path)
            w.writerow(
                {
                    "index": i,
                    "video_id": video_id,
                    "label": row.get("label", ""),
                    "vlm_score": "KEEP_HIGH",
                    "strip_png": str(strip_path),
                    "mp4_path": str(mp4),
                }
            )
            saved += 1
            print(f"[{saved}/{N_EXAMPLES}] {row.get('label', '')} -> {strip_path.name}", flush=True)

    print(f"\nFertig: {saved} Strips unter {output_dir}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
