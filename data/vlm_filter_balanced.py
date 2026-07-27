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
if str(_REPO_ROOT / "configs") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "configs"))
import config  # noqa: E402


MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
TARGET_FPS = 1
MAX_FRAME_SECONDS = 10
ALLOWED = {"KEEP_HIGH", "KEEP_LOW", "REMOVE"}

PROMPT = """You are a video content classifier specializing in music-related visual content. Your task is to analyze 3-5 frames sampled from the same video and classify it into exactly one of three categories: KEEP_HIGH, KEEP_LOW, or REMOVE.
The quality of your classification directly impacts the training of a machine learning model for audio-video retrieval. Incorrectly kept videos with no music-related visual content will introduce noise into the training data and degrade model performance. It is therefore critical that you classify accurately and do not hesitate to answer REMOVE when the visual content is clearly unrelated to music, even if music may be playing in the background.

Answer KEEP_HIGH if the frames show any of the following:
A music video, whether performance-based or narrative/cinematic in style
People visibly performing, singing, playing instruments, or dancing
A DJ or producer actively working with turntables, a mixing board, or music equipment
A live concert or stage performance
Choreographed dance clearly tied to a music performance
Visually styled or cinematically produced content that appears intentionally made for a music track, even if no one is performing

Answer KEEP_LOW if the frames show any of the following:

A static or near-static photo of an artist or band without action or movement
A static album cover or music artwork
An animated visualizer or motion graphic clearly associated with music
Lo-fi style artwork or simple music-related illustration
Still images or footage of musical instruments without a person actively playing them
Any other music-related visual content that does not show people performing or cinematic production

Answer REMOVE if the frames show any of the following:

Plain text or lyrics on a plain, black, or single-color background, even if the text is colorful or stylized
A screen recording of any software or application, even if music is playing in the background
Someone drawing, painting, or creating artwork on screen using software, even if music is audible
Gameplay footage of any kind
A tutorial, lecture, or instructional video of any kind
Random everyday footage such as home videos, street footage, or a person talking to a camera without any performance context
Any video where the primary visual content is clearly unrelated to music, regardless of whether music is playing


Important rules:

Look only at the visual content. Do not make assumptions about the audio.
If the frames show both music-related and unrelated content, classify based on what dominates visually.
A screen recording of software or games is always REMOVE, even if music-related content is visible in the background.
A static artist photo without movement or action is always KEEP_LOW, never KEEP_HIGH.
Do not hesitate to answer REMOVE. Keeping low-quality or unrelated videos is more harmful than removing them.
If genuinely unsure, answer KEEP_LOW.

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
    raw_csv = run_dir / "segments_unbalanced.csv"
    downloads = run_dir / "downloads"
    scored_csv = run_dir / "segments_unbalanced_vlm_scored.csv"

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
    total = len(rows)
    for i, row in enumerate(rows, start=1):
        yt_id = row["yt_id"].strip()
        start_s = row["start_seconds"].strip()
        video_id = f"{yt_id}_{start_s.replace('.', 'p')}"
        mp4 = downloads / f"{video_id}.mp4"

        frames = extract_frames_1fps(mp4, target_fps=TARGET_FPS, max_seconds=MAX_FRAME_SECONDS)
        pil_images = choose_five_or_three(frames)
        score = "REMOVE" if not pil_images else classify(processor, model, pil_images)

        new_row = dict(row)
        new_row["vlm_score"] = score
        scored_rows.append(new_row)

        if i % 100 == 0 or i == total:
            print(f"Fortschritt: {i}/{total}", flush=True)

    with open(scored_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(scored_rows)

    print(f"Geschrieben: {scored_csv} ({len(scored_rows)} Zeilen)", flush=True)


if __name__ == "__main__":
    main()

