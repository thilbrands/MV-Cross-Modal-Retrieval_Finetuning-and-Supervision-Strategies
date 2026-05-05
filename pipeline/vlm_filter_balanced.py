"""
VLM-Scoring auf bereits geladene Videos eines Dataset-Runs.

Input:
- segments_raw.csv
- downloads/*.mp4

Output:
- segments_raw_vlm_scored.csv (nur Zeilen mit existierender MP4 + vlm_score)
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import cv2
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import config  # noqa: E402


MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
TARGET_FPS = 1
MAX_FRAME_SECONDS = 10
ALLOWED = {"KEEP_HIGH", "KEEP_LOW", "REMOVE"}

PROMPT = """You are a video content classifier specializing in music-related visual content.
You are given 3-5 frames sampled from the same video. Based only on what you can see in the frames, classify the video into one of three categories.
Answer KEEP_HIGH if the frames show:
- A music video, whether performance-based or narrative/cinematic in style
- People visibly performing, singing, or dancing
- A live concert or stage performance
- Visually styled or cinematically produced content that appears to be made for a music track

Answer KEEP_LOW if the frames show:
- A static album cover or music artwork
- An animated visualizer or motion graphic clearly associated with music
- Lo-fi style artwork or simple music-related illustration
- Still images or footage of musical instruments
- Any other music-related visual content that does not show people or cinematic production

Answer REMOVE if the frames show:
- Plain text or lyrics on a plain or black background without styled visuals
- Gameplay footage
- A screen recording or tutorial of any kind
- Random everyday footage such as home videos, street footage, or a person talking to a camera without any performance context

Look only at the visual content. Do not make assumptions about the audio.
If unsure, answer KEEP_LOW.
Respond with only one word: KEEP_HIGH, KEEP_LOW, or REMOVE."""


def extract_frames_1fps(video_path: Path, target_fps: int = 1, max_seconds: int = 10):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0 or frame_count <= 0:
        return []
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    for t in range(max_seconds):
        frame_num = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def choose_five_or_three(frames_rgb):
    n = len(frames_rgb)
    if n < 3:
        return []
    if n >= 9:
        idxs = [0, 2, 4, 6, 8]
    else:
        idxs = [0, n // 2, n - 1]
    return [Image.fromarray(frames_rgb[i].astype("uint8")) for i in idxs]


def parse_label(text: str) -> str:
    # robust gegen Nebentext; bei Unsicherheit KEEP_LOW (wie im Prompt gefordert)
    for token in re.findall(r"[A-Z_]+", text.upper()):
        if token in ALLOWED:
            return token
    return "KEEP_LOW"


def classify(processor, model, pil_images) -> str:
    contents = [{"type": "image", "image": img} for img in pil_images]
    contents.append({"type": "text", "text": PROMPT})
    messages = [{"role": "user", "content": contents}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=16)
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    raw = output_text[0] if output_text else ""
    return parse_label(raw)


def main() -> None:
    run_name = config.get_latest_run_name()
    if not run_name:
        print("FEHLER: Kein Dataset-Run unter DATASETS_ROOT.", flush=True)
        sys.exit(1)

    run_dir = config.DATASETS_ROOT / run_name
    raw_csv = run_dir / "segments_raw.csv"
    downloads = run_dir / "downloads"
    scored_csv = run_dir / "segments_raw_vlm_scored.csv"

    if not raw_csv.exists():
        print(f"FEHLER: fehlt {raw_csv}", flush=True)
        sys.exit(1)
    if not downloads.exists():
        print(f"FEHLER: fehlt {downloads}", flush=True)
        sys.exit(1)

    print(f"Run: {run_name}", flush=True)
    print(f"Input: {raw_csv}", flush=True)
    print(f"Downloads: {downloads}", flush=True)

    with open(raw_csv, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        in_fields = list(rows[0].keys()) if rows else []
    out_fields = in_fields + ["vlm_score"]

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    scored_rows = []
    missing_count = 0
    total = len(rows)
    for i, row in enumerate(rows, start=1):
        yt_id = row["yt_id"].strip()
        start_s = row["start_seconds"].strip()
        video_id = f"{yt_id}_{start_s.replace('.', 'p')}"
        mp4 = downloads / f"{video_id}.mp4"

        if not mp4.exists():
            missing_count += 1
            continue

        frames = extract_frames_1fps(mp4, target_fps=TARGET_FPS, max_seconds=MAX_FRAME_SECONDS)
        pil_images = choose_five_or_three(frames)
        score = "REMOVE" if not pil_images else classify(processor, model, pil_images)

        new_row = dict(row)
        new_row["vlm_score"] = score
        scored_rows.append(new_row)

        if i % 100 == 0 or i == total:
            print(
                f"Fortschritt: {i}/{total} | geschrieben={len(scored_rows)} | fehlende_mp4={missing_count}",
                flush=True,
            )

    with open(scored_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(scored_rows)

    print(f"Geschrieben: {scored_csv} ({len(scored_rows)} Zeilen)", flush=True)
    print(f"Uebersprungen (fehlende MP4): {missing_count}", flush=True)


if __name__ == "__main__":
    main()

