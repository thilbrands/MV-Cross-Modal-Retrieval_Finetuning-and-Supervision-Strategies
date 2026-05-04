"""
Ein MP4 aus einem Dataset-Run: Frame-Extraktion wie extract_and_embed_videos (1 FPS, max. 10 s),
dann drei Frames an Qwen3-VL-2B-Instruct (Sekunde 0 / 5 / 8 der 1-FPS-Folge, sonst geklemmt bzw. Anfang-Mitte-Ende).

(Frame-Schleife hier mit fester Obergrenze max_seconds — extract_and_embed_videos.py kürzt mit
int(duration_sec) und kann dadurch weniger als 9 Frames liefern; volles extract-Modul lädt CLIP.)

Voraussetzungen wie vlm_smoke_test (transformers aus Git, GPU).

Video: erste *.mp4 (alphabetisch) in datasets/<neuester Run laut config>/downloads/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import config  # noqa: E402

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: E402

# Wie pipeline/extract_and_embed_videos.py
TARGET_FPS = 1
MAX_FRAME_SECONDS = 10


def extract_frames_1fps(video_path: Path, target_fps: int = 1, max_seconds: int = 10):
    """Eine Zeile pro Sekunde für t=0 … max_seconds-1. Nicht mit int(duration_sec) kappen — sonst z.B. 8,9s → 8 Frames und Index 8 fehlt."""
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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


def main() -> None:
    run_name = config.get_latest_run_name()
    if not run_name:
        print("FEHLER: Kein Dataset-Run unter DATASETS_ROOT.", flush=True)
        sys.exit(1)
    downloads = config.DATASETS_ROOT / run_name / "downloads"
    mp4s = sorted(downloads.glob("*.mp4"))
    if not mp4s:
        print(f"FEHLER: Keine .mp4 unter {downloads}", flush=True)
        sys.exit(1)
    mp4 = mp4s[0]
    print(f"Run: {run_name} | Video (erste .mp4): {mp4}", flush=True)

    frames_rgb = extract_frames_1fps(mp4, target_fps=TARGET_FPS, max_seconds=MAX_FRAME_SECONDS)
    n = len(frames_rgb)
    if n < 3:
        print(f"FEHLER: Mindestens 3 Frames nötig, erhalten: {n}.", flush=True)
        sys.exit(1)
    # Ziel Sekunde 0 / 5 / 8 — wenn weniger als 9 Samples (z. B. nur 8× read ok), letzten Index clampen.
    i0, i1, i2 = 0, min(5, n - 1), min(8, n - 1)
    if len({i0, i1, i2}) < 3:
        i0, i1, i2 = 0, n // 2, n - 1
    if n < 9:
        print(f"Hinweis: {n} Frames (nicht 10) → Indizes fürs VLM: {i0}, {i1}, {i2}.", flush=True)
    pil_images = [Image.fromarray(frames_rgb[i].astype("uint8")) for i in (i0, i1, i2)]

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_images[0]},
                {"type": "image", "image": pil_images[1]},
                {"type": "image", "image": pil_images[2]},
                {
                    "type": "text",
                    "text": (
                        "Drei Bilder aus dem Segment (Anfang, Mitte, gegen Ende; 1 FPS wo möglich). "
                        "Beschreibe kurz auf Deutsch, was man sieht (Szene, Instrumente, Stil)."
                    ),
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(output_text, flush=True)


if __name__ == "__main__":
    main()
