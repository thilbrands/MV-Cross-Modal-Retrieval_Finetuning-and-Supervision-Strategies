#!/usr/bin/env python3
"""
Kombiniert REMOVE-Frame-Strips zu einer PNG/PDF-Figure (PIL, kein Browser).

Standard: 4 Beispiele — Rock, Pop, Electronic (Text), Reggae. Zeilen 4,5,6,3 im Manifest.

  INPUT_DIR=/path/to/remove_examples \\
  python3 figures/combine_remove_example_strips.py

Output: remove_examples_figure.png + .pdf
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path.home() / "Desktop" / "remove_examples"
DEFAULT_ROWS = "4,5,6,3"
OUTPUT_BASENAME = os.environ.get("OUTPUT_BASENAME", "remove_examples_figure")

LABEL_W = 160
STRIP_W = 900
STRIP_H = 130
NUM_FRAMES = 5
ROW_GAP = 0
ROW_PAD = 1
FONT_SIZE = 16


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def resolve_strip_path(row: dict, input_dir: Path) -> Path | None:
    strip_field = (row.get("strip_png") or "").strip()
    if strip_field:
        p = Path(strip_field)
        if p.is_file():
            return p
        local = input_dir / p.name
        if local.is_file():
            return local

    index = (row.get("index") or "").strip()
    video_id = (row.get("video_id") or "").strip()
    if index and video_id:
        guess = input_dir / f"{int(index):02d}_{video_id}_remove.png"
        if guess.is_file():
            return guess
    return None


def load_all_rows(input_dir: Path) -> list[dict]:
    manifest = input_dir / "remove_examples_manifest.csv"
    if not manifest.is_file():
        print(f"FEHLER: {manifest} nicht gefunden.", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    with open(manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx = int((row.get("index") or "0").strip())
            strip_path = resolve_strip_path(row, input_dir)
            if strip_path is None:
                continue
            rows.append(
                {
                    "index": idx,
                    "label": (row.get("label") or "").strip(),
                    "strip_path": strip_path,
                }
            )

    rows.sort(key=lambda r: r["index"])
    if not rows:
        print("FEHLER: Keine Strips gefunden.", file=sys.stderr)
        sys.exit(1)
    return rows


def pick_rows(all_rows: list[dict]) -> list[dict]:
    if os.environ.get("SELECT_INDICES", "").strip():
        wanted = set(parse_int_list(os.environ["SELECT_INDICES"]))
        return [r for r in all_rows if r["index"] in wanted]
    positions = parse_int_list(os.environ.get("SELECT_ROWS", DEFAULT_ROWS))
    picked = []
    for pos in positions:
        if pos < 1 or pos > len(all_rows):
            print(f"FEHLER: SELECT_ROWS Position {pos} ungültig (1..{len(all_rows)}).", file=sys.stderr)
            sys.exit(1)
        picked.append(all_rows[pos - 1])
    if not picked:
        print("FEHLER: Keine Zeilen ausgewählt.", file=sys.stderr)
        sys.exit(1)
    return picked


def split_strip(strip: Image.Image, n: int = NUM_FRAMES) -> list[Image.Image]:
    w, h = strip.size
    cell_w = w // n
    return [strip.crop((i * cell_w, 0, w if i == n - 1 else (i + 1) * cell_w, h)) for i in range(n)]


def cover_crop(img: Image.Image, tw: int, th: int) -> Image.Image:
    scale = max(tw / img.width, th / img.height)
    sw = max(1, int(img.width * scale))
    sh = max(1, int(img.height * scale))
    resized = img.resize((sw, sh), Image.Resampling.LANCZOS)
    left = (sw - tw) // 2
    top = (sh - th) // 2
    return resized.crop((left, top, left + tw, top + th))


def uniform_frame_strip(strip: Image.Image, n: int = NUM_FRAMES) -> Image.Image:
    """5 (oder n) gleich breite Zellen — object-fit: cover pro Frame."""
    frames = split_strip(strip, n)
    cell_w = STRIP_W // n
    out = Image.new("RGB", (STRIP_W, STRIP_H), "white")
    for i, frame in enumerate(frames):
        out.paste(cover_crop(frame, cell_w, STRIP_H), (i * cell_w, 0))
    return out


def build_row(strip: Image.Image, genre: str, font) -> Image.Image:
    strip = uniform_frame_strip(strip)
    row_h = STRIP_H + 2 * ROW_PAD
    row_w = LABEL_W + STRIP_W
    row = Image.new("RGB", (row_w, row_h), "white")
    draw = ImageDraw.Draw(row)
    bbox = draw.textbbox((0, 0), genre, font=font)
    text_h = bbox[3] - bbox[1]
    draw.text((2, (row_h - text_h) // 2), genre, fill="#222222", font=font)
    row.paste(strip, (LABEL_W, ROW_PAD))
    return row


def save_pdf_from_image(img: Image.Image, pdf_path: Path) -> None:
    try:
        img.save(pdf_path, "PDF", resolution=300.0)
    except Exception:
        import matplotlib.pyplot as plt
        import numpy as np

        dpi = 150
        fig_w = img.width / dpi
        fig_h = img.height / dpi
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax.imshow(np.asarray(img))
        ax.axis("off")
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0, facecolor="white")
        plt.close(fig)


def combine(input_dir: Path, output_path: Path) -> int:
    rows = pick_rows(load_all_rows(input_dir))
    font = load_font(FONT_SIZE)

    row_imgs = []
    for row in rows:
        strip = Image.open(row["strip_path"]).convert("RGB")
        row_imgs.append(build_row(strip, row["label"], font))

    width = max(r.width for r in row_imgs)
    height = sum(r.height for r in row_imgs) + ROW_GAP * (len(row_imgs) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for row_img in row_imgs:
        canvas.paste(row_img, (0, y))
        y += row_img.height + ROW_GAP

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    pdf_path = output_path.with_suffix(".pdf")
    save_pdf_from_image(canvas, pdf_path)
    print(f"PDF: {pdf_path}", flush=True)
    return len(row_imgs)


def main() -> None:
    input_dir = Path(os.environ.get("INPUT_DIR", str(DEFAULT_INPUT)))
    output_path = Path(os.environ.get("OUTPUT_PATH", str(input_dir / f"{OUTPUT_BASENAME}.png")))

    sel = os.environ.get("SELECT_INDICES") or os.environ.get("SELECT_ROWS", DEFAULT_ROWS)
    print(f"Input:   {input_dir}", flush=True)
    print(f"Auswahl: {sel}", flush=True)
    print(f"Output:  {output_path}", flush=True)

    n = combine(input_dir, output_path)
    print(f"Fertig: {n} Beispiele -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
