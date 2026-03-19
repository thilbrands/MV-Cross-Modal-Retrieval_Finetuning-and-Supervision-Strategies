#!/usr/bin/env python3
"""
Zählt in der Roh-AudioSet-CSV Segmente pro Genre (Logik wie build_subset_csv im Downloader).

Standard-CSV: config.DATA_CSV → unbalanced_train_segments-2.csv (wie pipeline/downloader.py).
Andere Datei z.B.: DATA_CSV=/pfad/zur/balanced_train_segments.csv

Teil 1 — Roh-Pool ohne used_keys:
  Pro Genre: eindeutige Segmente (ytid, start, end), deren positive_labels die Ontology-MID
  dieses Genres enthalten. pure = nur dieses MID, mixed = weitere MIDs dazu.

Teil 2 — Simulation wie build_subset_csv:
  used_keys + NUM_VIDEOS_PER_LABEL (default 800): Reihenfolge MUSIC_GENRES, pro Genre
  zuerst pure, dann mixed.

Auf dem Cluster (Repo-Root):
  python3 scripts/count_audioset_genre_pool.py
  NUM_VIDEOS_PER_LABEL=800 python3 scripts/count_audioset_genre_pool.py
  DATA_CSV=/work2/.../balanced_train_segments.csv python3 scripts/count_audioset_genre_pool.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
import config  # noqa: E402


def load_label_map(ontology_path: Path) -> Dict[str, str]:
    with open(ontology_path, "r", encoding="utf-8") as f:
        ontology = json.load(f)
    return {item["name"]: item["id"] for item in ontology}


def parse_positive_labels(s: str) -> List[str]:
    if not s or len(s.strip()) == 0:
        return []
    return [x.strip().strip('"') for x in s.split(",") if x.strip()]


def read_segments(data_csv: Path) -> List[Tuple[str, float, float, str, List[str]]]:
    out: List[Tuple[str, float, float, str, List[str]]] = []
    with open(data_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) < 4:
                continue
            ytid = row[0].strip()
            try:
                start_s = float(row[1].strip())
                end_s = float(row[2].strip())
            except ValueError:
                continue
            pl_str = ",".join(row[3:]).strip().strip('"')
            mids = parse_positive_labels(pl_str)
            out.append((ytid, start_s, end_s, pl_str, mids))
    return out


# Muss mit pipeline/downloader.py main() übereinstimmen
MUSIC_GENRES = [
    "Classical music",
    "Jazz",
    "Rock music",
    "Electronic music",
    "Country",
    "Reggae",
    "Folk music",
    "Hip hop music",
    "Pop music",
    "Blues",
]


def main() -> None:
    data_csv = Path(os.environ.get("DATA_CSV", str(config.DATA_CSV)))
    ontology_path = Path(os.environ.get("ONTOLOGY_JSON", str(config.ONTOLOGY_JSON)))
    cap = int(os.environ.get("NUM_VIDEOS_PER_LABEL", "800"))

    if not data_csv.is_file():
        print(f"FEHLER: DATA_CSV nicht gefunden: {data_csv}", file=sys.stderr)
        sys.exit(1)
    if not ontology_path.is_file():
        print(f"FEHLER: ONTOLOGY_JSON nicht gefunden: {ontology_path}", file=sys.stderr)
        sys.exit(1)

    label_map = load_label_map(ontology_path)
    segments = read_segments(data_csv)

    pure_segments: Dict[str, List[Tuple[str, float, float, str]]] = {g: [] for g in MUSIC_GENRES}
    mixed_segments: Dict[str, List[Tuple[str, float, float, str]]] = {g: [] for g in MUSIC_GENRES}

    for ytid, start_s, end_s, pl_str, mids in segments:
        for label in MUSIC_GENRES:
            mid = label_map.get(label)
            if mid is None or mid not in mids:
                continue
            seg = (ytid, start_s, end_s, pl_str)
            if set(mids) == {mid}:
                pure_segments[label].append(seg)
            else:
                mixed_segments[label].append(seg)

    print("=== Roh-Pool in AudioSet-CSV (gleiche Genre-/MID-Logik wie downloader) ===")
    print(f"CSV (Default = unbalanced, siehe config.DATA_CSV): {data_csv}")
    print(f"Ontology: {ontology_path}")
    print(f"Segmente gelesen: {len(segments)}")
    print(f"NUM_VIDEOS_PER_LABEL (Simulation): {cap}")
    print()

    w = max(len(g) for g in MUSIC_GENRES) + 2
    print("--- Pro Genre: eindeutige Segmente mit diesem MID (ohne used_keys) ---")
    print(f"{'Genre':<{w}} {'pure':>8} {'mixed':>8} {'total':>8}")
    print("-" * (w + 8 + 8 + 8 + 3))

    pool_keys_per_genre: Dict[str, Set[Tuple[str, float, float]]] = {}
    for label in MUSIC_GENRES:
        keys_pure = {(t, s, e) for t, s, e, _ in pure_segments[label]}
        keys_mixed = {(t, s, e) for t, s, e, _ in mixed_segments[label]}
        keys_all = keys_pure | keys_mixed
        pool_keys_per_genre[label] = keys_all
        print(f"{label:<{w}} {len(keys_pure):>8} {len(keys_mixed):>8} {len(keys_all):>8}")

    print()
    print(f"--- Simulation build_subset_csv: used_keys + max {cap} pro Genre ---")
    used_keys: Set[Tuple[str, float, float]] = set()
    taken_per_genre = {g: 0 for g in MUSIC_GENRES}
    taken_pure = {g: 0 for g in MUSIC_GENRES}
    taken_mixed = {g: 0 for g in MUSIC_GENRES}

    for label in MUSIC_GENRES:
        if label_map.get(label) is None:
            continue
        count = 0
        for seg_list, is_pure in ((pure_segments[label], True), (mixed_segments[label], False)):
            for ytid, start_s, end_s, _pl in seg_list:
                if count >= cap:
                    break
                key = (ytid, start_s, end_s)
                if key in used_keys:
                    continue
                used_keys.add(key)
                count += 1
                taken_per_genre[label] += 1
                if is_pure:
                    taken_pure[label] += 1
                else:
                    taken_mixed[label] += 1
            if count >= cap:
                break

    print(f"{'Genre':<{w}} {'genommen':>10} {'pure':>12} {'mixed':>12} {'Pool':>12}")
    print("-" * (w + 10 + 12 + 12 + 12 + 5))
    for label in MUSIC_GENRES:
        pool = len(pool_keys_per_genre[label])
        print(
            f"{label:<{w}} {taken_per_genre[label]:>10} "
            f"{taken_pure[label]:>12} {taken_mixed[label]:>12} {pool:>12}"
        )
    print()
    print(f"Summe unique genommen (Simulation): {len(used_keys)}")
    print()
    print("Hinweis: Kleiner Pool = wenig Treffer in dieser CSV. Kleines 'genommen' bei großem Pool")
    print("         = Effekt von used_keys + Genre-Reihenfolge + Kappung.")


if __name__ == "__main__":
    main()
