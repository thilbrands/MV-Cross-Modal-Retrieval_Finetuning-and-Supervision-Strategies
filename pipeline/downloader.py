#!/usr/bin/env python3
import os
import shutil
import sys
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Repo-Root für Imports (config liegt im Repo-Root)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config


"""
AudioSet-Cluster-Download (downloader2):
- Phase 1: Ontology laden
- Phase 2: segments_raw.csv bauen (identisch zu pipeline/downloader.py)
- Phase 3: Video-Partial-Download (angepasst)
- Phase 4: segments_balanced.csv bauen (identisch zu pipeline/downloader.py)
"""


WORK_ROOT = config.WORK_ROOT
DATA_DIR = config.DATA_DIR
DATA_CSV = config.DATA_CSV
ONTOLOGY_JSON = config.ONTOLOGY_JSON
DATASETS_ROOT = config.DATASETS_ROOT

# Run-Name: Datum_Uhrzeit + Task-Name
_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_NAME = f"{_timestamp}_audioset"

# Run-Ordner und Unterordner für Downloads
RUN_DIR = DATASETS_ROOT / RUN_NAME
DOWNLOAD_DIR = RUN_DIR / "downloads"

DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Klar benannte Ergebnisdateien pro Run
RAW_CSV = RUN_DIR / "segments_raw.csv"
BALANCED_CSV = RUN_DIR / "segments_balanced.csv"
FAILED_CSV = RUN_DIR / "failed_downloads.csv"
CONFIG_PATH = RUN_DIR / "config.json"


def log(msg: str) -> None:
    # Ausgabe sofort in Slurm-Log (flush), damit man sieht, wo der Job hängt.
    print(msg, flush=True)


def write_run_config(
    run_name: str,
    start_time: str,
    git_commit: str,
    status: str,
    job_id: Optional[str],
    end_time: Optional[str] = None,
) -> None:
    cfg = {
        "run_name": run_name,
        "start_time": start_time,
        "git_commit": git_commit,
        "status": status,
        "job_id": job_id,
        "end_time": end_time,
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    log(f"[Config] run.config geschrieben: {CONFIG_PATH}")


def load_label_map(ontology_path: Path) -> Dict[str, str]:
    with open(ontology_path, "r", encoding="utf-8") as f:
        ontology = json.load(f)
    return {item["name"]: item["id"] for item in ontology}


def parse_positive_labels(s: str) -> List[str]:
    if not s or len(s.strip()) == 0:
        return []
    return [x.strip().strip('"') for x in s.split(",") if x.strip()]


def build_subset_csv(
    data_csv: Path,
    label_map: Dict[str, str],
    out_csv: Path,
    music_genres: List[str],
    num_videos_per_label: int,
) -> None:
    # IDENTISCH zu pipeline/downloader.py (CSV-Erstellung)
    log("[Subset-CSV] Lese Eingabe-CSV …")
    segments = []
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
            segments.append((ytid, start_s, end_s, pl_str))

    log(f"[Subset-CSV] {len(segments)} Segmente gelesen. Sortiere nach rein/gemischten Labels …")
    pure_segments: Dict[str, List[tuple]] = {g: [] for g in music_genres}
    mixed_segments: Dict[str, List[tuple]] = {g: [] for g in music_genres}

    for ytid, start_s, end_s, pl_str in segments:
        mids = parse_positive_labels(pl_str)
        for label in music_genres:
            mid = label_map.get(label)
            if mid is None or mid not in mids:
                continue
            seg = (ytid, start_s, end_s, pl_str)
            if set(mids) == {mid}:
                pure_segments[label].append(seg)
            else:
                mixed_segments[label].append(seg)

    used_keys = set()
    all_rows: List[Dict[str, str]] = []

    for label in music_genres:
        mid = label_map.get(label)
        if mid is None:
            print(f"Label '{label}' nicht in Ontology, überspringe.", file=sys.stderr)
            continue
        count = 0
        for seg_list in (pure_segments[label], mixed_segments[label]):
            for ytid, start_s, end_s, pl_str in seg_list:
                if count >= num_videos_per_label:
                    break
                key = (ytid, start_s, end_s)
                if key in used_keys:
                    continue
                used_keys.add(key)
                all_rows.append(
                    {
                        "yt_id": ytid,
                        "start_seconds": str(start_s),
                        "end_seconds": str(end_s),
                        "positive_labels": mid,
                        "label": label,
                    }
                )
                count += 1

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "yt_id",
                "start_seconds",
                "end_seconds",
                "positive_labels",
                "label",
            ],
        )
        w.writeheader()
        w.writerows(all_rows)

    log(f"[Subset-CSV] Geschrieben: {out_csv} mit {len(all_rows)} Zeilen.")


def run(cmd: List[str], timeout: int = 600) -> Tuple[bool, str]:
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        raw = (p.stderr or "").strip()
        if not raw:
            err = ""
        elif "ffmpeg exited" in raw.lower():
            err = raw[:2000]
        elif "ERROR:" in raw:
            for line in raw.split("\n"):
                if "ERROR:" in line:
                    err = line.strip()
                    break
            else:
                err = raw[:1500]
        else:
            err = raw[:1500]
        return p.returncode == 0, err
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


VIDEO_QUALITY = "18"  # wie im Notebook


@lru_cache(maxsize=1)
def _yt_dlp_cookie_args() -> List[str]:
    cookie_file = os.environ.get("YT_DLP_COOKIES", "").strip()
    cookie_from_browser = os.environ.get("YT_DLP_COOKIES_FROM_BROWSER", "").strip()

    if cookie_file and cookie_from_browser:
        raise ValueError("Bitte entweder YT_DLP_COOKIES oder YT_DLP_COOKIES_FROM_BROWSER setzen, nicht beides.")

    if cookie_from_browser:
        log(f"[Download][yt-dlp] Nutze Cookies aus Browser: {cookie_from_browser}")
        return ["--cookies-from-browser", cookie_from_browser]

    if cookie_file:
        p = Path(cookie_file)
        if not p.is_file():
            raise FileNotFoundError(f"YT_DLP_COOKIES gesetzt, aber Datei existiert nicht: {p}")
        log(f"[Download][yt-dlp] Nutze Cookies-Datei: {p}")
        return ["--cookies", str(p)]

    return []


def download_video_segment_partial(
    youtube_id: str,
    start_sec: float,
    end_sec: float,
    out_path: Path,
) -> Tuple[bool, str]:
    """
    Partial download via ffmpeg external_downloader:
    - yt-dlp übernimmt Media-Download/Feeding an ffmpeg
    - ffmpeg bekommt -ss / -t, um nur den gewünschten Abschnitt zu extrahieren
    """
    cookie_args = _yt_dlp_cookie_args()

    duration = max(0.0, float(end_sec) - float(start_sec))

    # ffmpeg external downloader args:
    # -ss start seek
    # -t duration stop after duration
    ext_down_args_str = f"-ss {start_sec} -t {duration} -loglevel quiet"


    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-check-certificates",
        "--no-playlist",
        "--remote-components",
        "ejs:github",
        "--js-runtimes",
        "node",
        *cookie_args,
        "-f",
        f"{VIDEO_QUALITY}/worst",
        "--merge-output-format",
        "mp4",
        "--external-downloader",
        "ffmpeg",
        "--external-downloader-args",
        ext_down_args_str,
        "--no-warnings",
        "-o",
        str(out_path),
        f"https://www.youtube.com/watch?v={youtube_id}",
    ]
    return run(cmd, timeout=600)


def _download_one(row: Dict[str, str], download_dir: Path) -> Tuple[Dict[str, str], bool, str]:
    yt_id = row["yt_id"].strip()
    try:
        start_sec = float(row["start_seconds"])
        end_sec = float(row["end_seconds"])
    except (ValueError, KeyError):
        return (row, False, "Ungültige start_seconds/end_seconds")

    safe_start = str(start_sec).replace(".", "p")
    out_path = download_dir / f"{yt_id}_{safe_start}.mp4"

    if out_path.exists():
        return (row, True, "")

    ok, err = download_video_segment_partial(yt_id, start_sec, end_sec, out_path)
    if ok and out_path.exists():
        return (row, True, "")

    return (row, False, err or "Unbekannt")


def download_all_segments(subset_csv: Path, download_dir: Path) -> None:
    if not shutil.which("ffmpeg"):
        log("[Download] FEHLER: ffmpeg nicht im PATH. Bitte im Job-Skript 'conda_ffmpeg' aktivieren.")
        sys.exit(1)

    log("[Download] Hinweis: Partial Download via external_downloader ffmpeg (-ss/-t).")
    log(f"[Download] ffmpeg: {shutil.which('ffmpeg')}")

    with open(subset_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    n_workers = int(os.environ.get("AUDIOSET_DOWNLOAD_WORKERS", "2"))
    log(f"[Download] Subset-CSV geladen: {total} Zeilen. Parallele Downloads: {n_workers} Worker.")
    log("[Download] Fortschritt alle 100 Segmente …")

    success_count = 0
    failed: List[Dict[str, str]] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_download_one, row, download_dir): row for row in rows}
        for fut in as_completed(futures):
            completed += 1
            try:
                row, ok, err = fut.result()
                yt_id = row.get("yt_id", "").strip()
                if ok:
                    success_count += 1
                else:
                    failed.append(
                        {
                            "yt_id": yt_id,
                            "start_seconds": row.get("start_seconds", ""),
                            "end_seconds": row.get("end_seconds", ""),
                            "positive_labels": row.get("positive_labels", ""),
                            "reason": (err or "")[:500],
                        }
                    )
                    err_short = (err or "").strip().replace("\n", " ")[:400]
                    if len(failed) <= 30 or len(failed) % 50 == 0:
                        log(f"[Download] FEHLER {yt_id}: {err_short or '(keine Meldung)'}")
            except Exception as e:
                row = futures[fut]
                failed.append(
                    {
                        "yt_id": row.get("yt_id", ""),
                        "start_seconds": row.get("start_seconds", ""),
                        "end_seconds": row.get("end_seconds", ""),
                        "positive_labels": row.get("positive_labels", ""),
                        "reason": str(e)[:500],
                    }
                )

            if completed % 100 == 0 or completed == total:
                log(f"[Download] Fortschritt: {completed}/{total} (ok={success_count}, failed={len(failed)})")

    log(f"[Download] Ende: {success_count}/{total} erfolgreich, {len(failed)} fehlgeschlagen.")

    if failed:
        with open(FAILED_CSV, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["yt_id", "start_seconds", "end_seconds", "positive_labels", "reason"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(failed)
        log(f"[Download] Fehlgeschlagene Einträge gespeichert: {FAILED_CSV} (Gründe in Spalte 'reason')")


def build_cleaned_balanced_csv(
    subset_csv: Path,
    cleaned_csv: Path,
    download_dir: Path,
    genres: List[str],
) -> None:
    # IDENTISCH zu pipeline/downloader.py (Balanced CSV)
    log("[Balanced-CSV] Lese Subset-CSV und prüfe existierende MP4s …")
    with open(subset_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    by_genre: Dict[str, List[Dict[str, str]]] = {g: [] for g in genres}

    for row in rows:
        g = row.get("label", "").strip()
        if g not in by_genre:
            continue
        yt_id = row["yt_id"].strip()
        start_s = row["start_seconds"].strip()
        safe = start_s.replace(".", "p")
        if (download_dir / f"{yt_id}_{safe}.mp4").exists():
            by_genre[g].append(dict(row))

    min_per_genre = min(len(by_genre[g]) for g in genres)
    balanced_rows: List[Dict[str, str]] = []
    for g in genres:
        for row in by_genre[g][:min_per_genre]:
            row_with_label = {**row, "label": g}
            balanced_rows.append(row_with_label)

    with open(cleaned_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "yt_id",
                "start_seconds",
                "end_seconds",
                "positive_labels",
                "label",
            ],
        )
        w.writeheader()
        w.writerows(balanced_rows)

    log(f"[Balanced-CSV] Geschrieben: {cleaned_csv}")
    log(f"[Balanced-CSV] {len(balanced_rows)} Zeilen, je {min_per_genre} pro Genre (balanced).")
    log(f"[Balanced-CSV] Pro Genre vor Kappung: {[len(by_genre[g]) for g in genres]}")


def main() -> None:
    start_time = datetime.now().isoformat(timespec="seconds")
    git_commit = config.get_git_commit()
    job_id = os.environ.get("SLURM_JOB_ID")

    write_run_config(
        run_name=RUN_NAME,
        start_time=start_time,
        git_commit=git_commit,
        status="running",
        job_id=job_id,
        end_time=None,
    )

    log("=" * 60)
    log("AudioSet-Cluster-Run gestartet (downloader2)")
    log("=" * 60)

    log(f"WORK_ROOT: {WORK_ROOT}")
    log(f"DATA_DIR: {DATA_DIR}")
    log(f"DATA_CSV: {DATA_CSV} (exists={DATA_CSV.exists()})")
    log(f"ONTOLOGY_JSON: {ONTOLOGY_JSON} (exists={ONTOLOGY_JSON.exists()})")
    log(f"DATASETS_ROOT: {DATASETS_ROOT}")
    log(f"RUN_NAME: {RUN_NAME}")
    log(f"RUN_DIR: {RUN_DIR}")
    log(f"DOWNLOAD_DIR: {DOWNLOAD_DIR}")
    log(f"RAW_CSV: {RAW_CSV}")
    log(f"BALANCED_CSV: {BALANCED_CSV}")
    log(f"FAILED_CSV: {FAILED_CSV}")
    log(f"GIT_COMMIT: {git_commit}")

    if not DATA_CSV.exists():
        log("FEHLER: DATA_CSV fehlt. COPY_TO_WORK2.md ausführen?")
        sys.exit(1)
    if not ONTOLOGY_JSON.exists():
        log("FEHLER: ONTOLOGY_JSON fehlt. COPY_TO_WORK2.md ausführen?")
        sys.exit(1)

    log("")
    log("--- Phase 1: Ontology laden ---")
    label_map = load_label_map(ONTOLOGY_JSON)
    log(f"Ontology geladen: {len(label_map)} Labels.")

    music_genres = [
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
    num_videos_per_label = int(3107 * 1.1)
    log(f"Genres: {music_genres}")
    log(f"Videos pro Label: {num_videos_per_label}")

    log("")
    log("--- Phase 2: Subset-CSV bauen (aus unbalanced_train_segments) ---")
    build_subset_csv(
        data_csv=DATA_CSV,
        label_map=label_map,
        out_csv=RAW_CSV,
        music_genres=music_genres,
        num_videos_per_label=num_videos_per_label,
    )

    log("")
    log("--- Phase 3: Downloads (yt-dlp + partial via ffmpeg external downloader) ---")
    download_all_segments(RAW_CSV, DOWNLOAD_DIR)

    log("")
    log("--- Phase 4: Bereinigte balanced CSV ---")
    build_cleaned_balanced_csv(
        subset_csv=RAW_CSV,
        cleaned_csv=BALANCED_CSV,
        download_dir=DOWNLOAD_DIR,
        genres=music_genres,
    )

    log("")
    log("=" * 60)
    log("AudioSet-Cluster-Run fertig.")
    log("=" * 60)

    end_time = datetime.now().isoformat(timespec="seconds")
    write_run_config(
        run_name=RUN_NAME,
        start_time=start_time,
        git_commit=git_commit,
        status="finished",
        job_id=job_id,
        end_time=end_time,
    )


if __name__ == "__main__":
    main()