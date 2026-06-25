"""
Zentrale Pfad-Config für die Cluster-Pipeline (work2).

Alle festen Pfade stehen hier; run-spezifische Dinge (z. B. welcher Dataset-Run)
werden per Umgebungsvariable oder Default „neuester Run“ gesteuert.
"""
import subprocess
from pathlib import Path
from typing import Optional


# Fester Work-Root auf dem Cluster (nicht verhandelbar)
WORK_ROOT = Path("/work2/ra39oxet-DatasetAudioSetSubset")

# Alle Dataset-Runs liegen darunter (pro Run ein Ordner: Datum_Uhrzeit_audioset)
DATASETS_ROOT = WORK_ROOT / "datasets"

# Training-Runs: jeder Lauf bekommt einen Ordner Datum_Uhrzeit (Heads + meta.json)
TRAINING_RUNS_ROOT = WORK_ROOT / "training_runs"

# Eingabedaten für den Downloader (AudioSet CSV + Ontology)
DATA_DIR = WORK_ROOT / "AudioSetData"
DATA_CSV = DATA_DIR / "unbalanced_train_segments-2.csv"
ONTOLOGY_JSON = DATA_DIR / "ontology.json"

# Device für PyTorch (CLIP, Wav2CLIP): hier zentral festgelegt, überall importierbar.
# Bevorzugung: "cuda" wenn verfügbar, sonst "cpu". Bei fehlendem torch: "cpu".
DEVICE_PREFER = "cuda"
try:
    import torch
    DEVICE = "cuda" if (DEVICE_PREFER == "cuda" and torch.cuda.is_available()) else "cpu"
except ImportError:
    DEVICE = "cpu"

# Fallback-Pfade, wenn keine training_runs existieren (Rückwärtskompatibilität)
PROJECTION_HEADS_PATH = WORK_ROOT / "projection_heads.pt"
PROJECTION_HEADS_GENRE_PATH = WORK_ROOT / "projection_heads_genre.pt"


def get_git_commit() -> str:
    """Git-Commit (kurz) des Repos; leer wenn kein Git/Fehler."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        return (out.stdout or "").strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def get_new_training_run_dir() -> Path:
    """Erstellt TRAINING_RUNS_ROOT und gibt einen neuen Unterordner zurück: YYYY-MM-DD_HH-MM, bei Kollision _1, _2, …"""
    from datetime import datetime
    TRAINING_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    base = datetime.now().strftime("%Y-%m-%d_%H-%M")
    name = base
    n = 0
    while (TRAINING_RUNS_ROOT / name).exists():
        n += 1
        name = f"{base}_{n}"
    run_dir = TRAINING_RUNS_ROOT / name
    run_dir.mkdir(parents=True)
    return run_dir


def _run_name_sort_key(path: Path) -> tuple:
    """Sort key: Timestamp aus Ordnernamen (YYYY-MM-DD_HH-MM-SS), sonst mtime."""
    from datetime import datetime

    parts = path.name.split("_")
    if len(parts) >= 2:
        try:
            ts = datetime.strptime(f"{parts[0]}_{parts[1]}", "%Y-%m-%d_%H-%M-%S")
            return (1, ts.timestamp(), path.stat().st_mtime)
        except ValueError:
            pass
    return (0, path.stat().st_mtime, 0.0)


def get_latest_run_name() -> Optional[str]:
    """
    Name des neuesten Dataset-Runs unter DATASETS_ROOT.
    Primär nach Timestamp im Ordnernamen (YYYY-MM-DD_HH-MM-SS_audioset),
    nicht nach Dateisystem-mtime (sonst kann ein alter Run durch spätere Jobs „neu“ wirken).
  """
    if not DATASETS_ROOT.exists():
        return None
    runs = [p for p in DATASETS_ROOT.iterdir() if p.is_dir()]
    if not runs:
        return None
    runs.sort(key=_run_name_sort_key, reverse=True)
    return runs[0].name


def get_latest_training_run_with(head_filename: str) -> Optional[Path]:
    """Neuester Ordner unter TRAINING_RUNS_ROOT, der head_filename enthält; sonst None."""
    if not TRAINING_RUNS_ROOT.exists():
        return None
    candidates = [d for d in TRAINING_RUNS_ROOT.iterdir() if d.is_dir() and (d / head_filename).exists()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]
