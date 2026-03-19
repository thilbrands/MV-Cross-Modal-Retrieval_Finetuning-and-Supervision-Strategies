#!/usr/bin/env python3
"""
Zählt in der Roh-AudioSet-CSV Segmente pro Genre (Logik wie build_subset_csv im Downloader).

Standard-CSV: config.DATA_CSV → unbalanced_train_segments-2.csv (wie pipeline/downloader.py).
Andere Datei z.B.: DATA_CSV=/pfad/zur/balanced_train_segments.csv

Teil 1 — Roh-Pool ohne used_keys:
  Genre-Reihenfolge wie in downloader.py (MUSIC_GENRES).
  Pro Genre: pure / mixed / total (eindeutige Segmente).

Teil 2 — Balanciertes Maximum (ermittelt, kein festes Cap):
  Genre-Reihenfolge für die Simulation: aufsteigend nach „mixed“-Pool (eindeutige mixed-Segmente),
  d.h. Genres mit vielen Mixed-Treffern stehen hinten (weniger Konkurrenz für andere zuerst).
  Binärsuche: größtes N, sodass bei used_keys + Cap N pro Genre gilt min(genommen) >= N.
  (= maximale Zahl, mit der du pro Genre gleich viele Segmente wie beim Downloader füllen kannst.)

Optional: SIM_CAP=3000 — zusätzlich eine Simulation mit fester Obergrenze (Vergleich / Debug).

Auf dem Cluster (Repo-Root):
  python3 scripts/count_audioset_genre_pool.py
  DATA_CSV=... SIM_CAP=3000 python3 scripts/count_audioset_genre_pool.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
import config  # noqa: E402


def load_label_map(ontology_path: Path) -> Dict[str, str]:
    import json

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
    "Funk",
    "Hip hop music",
    "Pop music",
    "Blues",
]


def simulate_subset(
    cap: int,
    genre_order: List[str],
    pure_segments: Dict[str, List[Tuple[str, float, float, str]]],
    mixed_segments: Dict[str, List[Tuple[str, float, float, str]]],
    label_map: Dict[str, str],
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], int]:
    """Wie build_subset_csv: used_keys, pro Genre max cap, pure vor mixed."""
    used_keys: Set[Tuple[str, float, float]] = set()
    taken = {g: 0 for g in MUSIC_GENRES}
    taken_pure = {g: 0 for g in MUSIC_GENRES}
    taken_mixed = {g: 0 for g in MUSIC_GENRES}

    for label in genre_order:
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
                taken[label] += 1
                if is_pure:
                    taken_pure[label] += 1
                else:
                    taken_mixed[label] += 1
            if count >= cap:
                break

    return taken, taken_pure, taken_mixed, len(used_keys)


def max_balanced_cap(
    genre_order: List[str],
    pool_sizes: Dict[str, int],
    pure_segments: Dict[str, List[Tuple[str, float, float, str]]],
    mixed_segments: Dict[str, List[Tuple[str, float, float, str]]],
    label_map: Dict[str, str],
) -> int:
    """
    Größtes N mit: simulate(N) erfüllt min_g taken[g] >= N.
    Obere Schranke: min Pool-Größe (mehr kann kein Genre allein schon nicht haben).
    """
    hi = min(pool_sizes[g] for g in MUSIC_GENRES)
    lo = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        taken, _, _, _ = simulate_subset(mid, genre_order, pure_segments, mixed_segments, label_map)
        if min(taken[g] for g in MUSIC_GENRES) >= mid:
            lo = mid
        else:
            hi = mid - 1
    return lo


def main() -> None:
    data_csv = Path(os.environ.get("DATA_CSV", str(config.DATA_CSV)))
    ontology_path = Path(os.environ.get("ONTOLOGY_JSON", str(config.ONTOLOGY_JSON)))
    sim_cap_env = os.environ.get("SIM_CAP", "").strip()

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

    pool_keys_per_genre: Dict[str, Set[Tuple[str, float, float]]] = {}
    mixed_unique_count: Dict[str, int] = {}
    for label in MUSIC_GENRES:
        keys_pure = {(t, s, e) for t, s, e, _ in pure_segments[label]}
        keys_mixed = {(t, s, e) for t, s, e, _ in mixed_segments[label]}
        pool_keys_per_genre[label] = keys_pure | keys_mixed
        mixed_unique_count[label] = len(keys_mixed)

    pool_sizes = {g: len(pool_keys_per_genre[g]) for g in MUSIC_GENRES}

    # Simulation: wenig „mixed“ zuerst, viel „mixed“ zuletzt; bei Gleichstand downloader-Reihenfolge
    genre_order_mixed = sorted(
        MUSIC_GENRES,
        key=lambda g: (mixed_unique_count[g], MUSIC_GENRES.index(g)),
    )

    w = max(len(g) for g in MUSIC_GENRES) + 2

    print("=== Roh-Pool in AudioSet-CSV (gleiche Genre-/MID-Logik wie downloader) ===")
    print(f"CSV (Default = unbalanced, siehe config.DATA_CSV): {data_csv}")
    print(f"Ontology: {ontology_path}")
    print(f"Segmente gelesen: {len(segments)}")
    print()

    print("--- Pro Genre: eindeutige Segmente mit diesem MID (ohne used_keys) ---")
    print("(Reihenfolge wie MUSIC_GENRES / downloader)")
    print(f"{'Genre':<{w}} {'pure':>8} {'mixed':>8} {'total':>8}")
    print("-" * (w + 8 + 8 + 8 + 3))
    for label in MUSIC_GENRES:
        keys_pure = {(t, s, e) for t, s, e, _ in pure_segments[label]}
        keys_mixed = {(t, s, e) for t, s, e, _ in mixed_segments[label]}
        print(f"{label:<{w}} {len(keys_pure):>8} {len(keys_mixed):>8} {len(pool_keys_per_genre[label]):>8}")

    print()
    print("--- Balanciertes Maximum N (used_keys, Cap N pro Genre) ---")
    print("Genre-Reihenfolge (Simulation): nach aufsteigendem mixed-Pool; viele Mixed-Treffer → weiter hinten.")
    print("Reihenfolge:", " → ".join(genre_order_mixed))
    print()

    n_bal = max_balanced_cap(
        genre_order_mixed, pool_sizes, pure_segments, mixed_segments, label_map
    )
    taken, taken_pure, taken_mixed, n_unique = simulate_subset(
        n_bal, genre_order_mixed, pure_segments, mixed_segments, label_map
    )

    print(f"Ermitteltes maximales balanciertes N: {n_bal}")
    print("(Größtes N mit: jedes Genre bekommt mindestens N Segmente unter dieser greedy-Regel.)")
    print(f"Obere Schranke (min Pool-Größe): {min(pool_sizes.values())}")
    print()

    print(f"--- Detail bei Cap = {n_bal} (Reihenfolge = Simulation) ---")
    print(f"{'Genre':<{w}} {'genommen':>10} {'pure':>12} {'mixed':>12} {'Pool':>12}")
    print("-" * (w + 10 + 12 + 12 + 12 + 5))
    for label in genre_order_mixed:
        pool = pool_sizes[label]
        print(
            f"{label:<{w}} {taken[label]:>10} "
            f"{taken_pure[label]:>12} {taken_mixed[label]:>12} {pool:>12}"
        )
    print()
    print(f"Summe unique Segmente (Simulation): {n_unique}")
    print(f"min(genommen) = max(genommen) = {n_bal}: {all(taken[g] == n_bal for g in MUSIC_GENRES)}")

    if sim_cap_env:
        cap_fix = int(sim_cap_env)
        print()
        print(f"--- Optional: fester Cap SIM_CAP={cap_fix}, Reihenfolge wie downloader (MUSIC_GENRES) ---")
        t2, tp2, tm2, u2 = simulate_subset(
            cap_fix, MUSIC_GENRES, pure_segments, mixed_segments, label_map
        )
        print(f"{'Genre':<{w}} {'genommen':>10} {'pure':>12} {'mixed':>12} {'Pool':>12}")
        print("-" * (w + 10 + 12 + 12 + 12 + 5))
        for label in MUSIC_GENRES:
            print(
                f"{label:<{w}} {t2[label]:>10} "
                f"{tp2[label]:>12} {tm2[label]:>12} {pool_sizes[label]:>12}"
            )
        print()
        print(f"Summe unique: {u2} | min(genommen) = {min(t2[g] for g in MUSIC_GENRES)}")


if __name__ == "__main__":
    main()
