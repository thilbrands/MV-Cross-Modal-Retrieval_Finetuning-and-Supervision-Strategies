#!/usr/bin/env python3
"""
Zählt „geladene“ Videos pro Genre: Zeilen aus train_val_test_split.csv,
bei denen video- und audio-Embedding existieren (wie PairDataset).

Gibt aus: Gesamt pro Genre und eine Tabelle train / val / test pro Genre
(sowie Summenzeilen).

Auf dem Cluster vom Repo-Root:
  python3 scripts/count_videos_per_genre.py
  DATASET_RUN_NAME=2026-03-13_18-12-31_audioset python3 scripts/count_videos_per_genre.py
"""
from __future__ import annotations

import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
import config  # noqa: E402


def main() -> None:
    run_name = os.environ.get("DATASET_RUN_NAME") or config.get_latest_run_name()
    if not run_name:
        print("FEHLER: DATASET_RUN_NAME nicht gesetzt und kein Run unter DATASETS_ROOT.", file=sys.stderr)
        sys.exit(1)

    run_dir = config.DATASETS_ROOT / run_name
    split_csv = run_dir / "train_val_test_split.csv"
    emb_dir = run_dir / "embeddings"

    if not split_csv.exists():
        print(f"FEHLER: {split_csv} nicht gefunden.", file=sys.stderr)
        sys.exit(1)
    if not emb_dir.is_dir():
        print(f"FEHLER: {emb_dir} existiert nicht.", file=sys.stderr)
        sys.exit(1)

    SPLIT_ORDER = ("train", "val", "test")

    # Wie PairDataset: nur zählen, wenn beide .npy da sind
    per_genre_total: Counter[str] = Counter()
    per_genre_split: dict[str, Counter[str]] = defaultdict(Counter)
    split_totals_all: Counter[str] = Counter()  # über alle Genres (geladen)
    skipped_no_emb = 0
    rows_total = 0

    with open(split_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_total += 1
            vid = (row.get("video_id") or "").strip()
            split = (row.get("split") or "").strip()
            label = (row.get("label") or "").strip() or "(leer)"
            if not vid:
                continue
            v_path = emb_dir / "video" / f"{vid}.npy"
            a_path = emb_dir / "audio" / f"{vid}.npy"
            if not (v_path.is_file() and a_path.is_file()):
                skipped_no_emb += 1
                continue
            per_genre_total[label] += 1
            if split:
                per_genre_split[label][split] += 1
                split_totals_all[split] += 1
            else:
                per_genre_split[label]["(kein split)"] += 1
                split_totals_all["(kein split)"] += 1

    print(f"Dataset-Run: {run_name}")
    print(f"CSV:         {split_csv}")
    print(f"Embeddings:  {emb_dir}")
    print()
    loaded = sum(per_genre_total.values())
    print(f"Zeilen in CSV:              {rows_total}")
    print(f"Mit Video+Audio-Embedding: {loaded}")
    print(f"Ohne vollständige Embeds:  {skipped_no_emb}")
    print(f"Unique Genres (label):     {len(per_genre_total)}")
    print()
    print("--- Summe geladene Videos pro Split (alle Genres) ---")
    for s in SPLIT_ORDER:
        if s in split_totals_all:
            print(f"  {s:>5}: {split_totals_all[s]}")
    if "(kein split)" in split_totals_all:
        print(f"  {'(kein split)':>5}: {split_totals_all['(kein split)']}")
    other_splits = [k for k in split_totals_all if k not in SPLIT_ORDER and k != "(kein split)"]
    for s in sorted(other_splits):
        print(f"  {s:>5}: {split_totals_all[s]}")
    print(f"  {'TOTAL':>5}: {loaded}")
    print()

    # Tabelle: Genre | train | val | test | sonstige? | total
    rows_table: list[tuple[str, int, int, int, int, int]] = []
    for label, total in per_genre_total.most_common():
        by_s = per_genre_split[label]
        t = by_s.get("train", 0)
        v = by_s.get("val", 0)
        te = by_s.get("test", 0)
        other = total - t - v - te
        rows_table.append((label, t, v, te, other, total))

    # Spaltenbreiten
    w_label = max(24, min(56, max(len(r[0]) for r in rows_table) + 2)) if rows_table else 24
    colw = 8

    def fmt_row(genre: str, a: int, b: int, c: int, o: int, tot: int, is_sep: bool = False) -> str:
        if is_sep:
            return "-" * (w_label + colw * 5 + 15)
        g = genre if len(genre) <= w_label else genre[: w_label - 3] + "..."
        return f"{g:<{w_label}} {a:>{colw}} {b:>{colw}} {c:>{colw}} {o:>{colw}} {tot:>{colw}}"

    print("--- Pro Genre: Aufteilung train / val / test (nur geladene mit Embeddings) ---")
    hdr = (
        f"{'Genre':<{w_label}} {'train':>{colw}} {'val':>{colw}} {'test':>{colw}} "
        f"{'sonst':>{colw}} {'total':>{colw}}"
    )
    print(hdr)
    print(fmt_row("", 0, 0, 0, 0, 0, is_sep=True))
    for label, t, v, te, o, tot in rows_table:
        print(fmt_row(label, t, v, te, o, tot))
    print(fmt_row("", 0, 0, 0, 0, 0, is_sep=True))
    sum_t = sum(r[1] for r in rows_table)
    sum_v = sum(r[2] for r in rows_table)
    sum_te = sum(r[3] for r in rows_table)
    sum_o = sum(r[4] for r in rows_table)
    print(fmt_row("GESAMT", sum_t, sum_v, sum_te, sum_o, loaded))
    print()
    print("(sonst = Zeilen ohne train/val/test oder unbekannter split-Wert; sollte 0 sein.)")


if __name__ == "__main__":
    main()
