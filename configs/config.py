"""
Zentrale Pfad-Config für die Cluster-Pipeline (work2).

Alle festen Pfade stehen hier; run-spezifische Dinge (z. B. welcher Dataset-Run)
werden per Umgebungsvariable oder Default „neuester Run“ gesteuert.

WORK_ROOT: Default = Cluster-Pfad; überschreibbar per Env, z. B.
  WORK_ROOT=~/ba_work  oder  WORK_ROOT=/pfad/zu/work
"""
import os
import subprocess
from pathlib import Path
from typing import Optional


WORK_ROOT = Path(
    os.environ.get("WORK_ROOT", "/work2/ra39oxet-DatasetAudioSetSubset")
).expanduser()

# Alle Dataset-Runs liegen darunter (pro Run ein Ordner: Datum_Uhrzeit_audioset)
DATASETS_ROOT = WORK_ROOT / "datasets"

# Live-Training/Eval/Plots während der Pipeline (Prozess-Ausgabe)
TRAINING_RUNS_ROOT = WORK_ROOT / "training_runs"

# Kuratierte Abgabe-Ergebnisse: results/<run>/{checkpoints,results,outputs,meta}/
RESULTS_ROOT = WORK_ROOT / "results"

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


def resolve_plot_output_dir(fallback: Path) -> Path:
    """
    Ausgabeordner für Plots/Eval-Figuren.
    Priorität: PLOT_OUTPUT_DIR → TRAINING_RUN_DIR → EVAL_OUTPUT_DIR → fallback
    (meist Dataset-Run). So landen Eval + Figures im gleichen Training-Run-Ordner.
    """
    for key in ("PLOT_OUTPUT_DIR", "TRAINING_RUN_DIR", "EVAL_OUTPUT_DIR"):
        raw = os.environ.get(key)
        if raw:
            out = Path(raw).expanduser()
            out.mkdir(parents=True, exist_ok=True)
            return out
    fallback = Path(fallback)
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def get_git_commit() -> str:
    """Git-Commit (kurz) des Repos; leer wenn kein Git/Fehler."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parents[1],
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
    """Neuester Ordner unter TRAINING_RUNS_ROOT, der head_filename (ggf. unter checkpoints/) enthält."""
    if not TRAINING_RUNS_ROOT.exists():
        return None

    def _has_head(d: Path) -> bool:
        return (d / head_filename).exists() or (d / "checkpoints" / head_filename).exists()

    candidates = [d for d in TRAINING_RUNS_ROOT.iterdir() if d.is_dir() and _has_head(d)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# Eval-CSVs → results/<run>/results/; Trainings-/Analyse-CSVs → results/<run>/outputs/
_EVAL_CSV_NAMES = {
    "results_evaluation.csv",
    "results_evaluation_bootstrap_diff.csv",
    "results_evaluation_for_comparison_with_related_work.csv",
    "results_quality_split.csv",
    "results_genre_breakdown.csv",
}


def package_run_to_results(
    training_run_dir: Path | str,
    dest: Path | str | None = None,
) -> Path:
    """
    Kopiert kuratierte Artefakte aus einem training_runs/<run>/ nach
    results/<run>/{checkpoints,results,outputs,meta}/
    (Figuren bleiben zusätzlich im training_run).

    dest: optionaler Zielordner (z.B. results/<main>/e4_exploration);
          Default: RESULTS_ROOT / <run-name>.
    """
    import shutil

    src = Path(training_run_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"Training-Run nicht gefunden: {src}")

    dest_path = Path(dest) if dest is not None else RESULTS_ROOT / src.name
    ckpt_dir = dest_path / "checkpoints"
    results_dir = dest_path / "results"
    outputs_dir = dest_path / "outputs"
    meta_dir = dest_path / "meta"
    for d in (ckpt_dir, results_dir, outputs_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    for pt in src.glob("*.pt"):
        shutil.copy2(pt, ckpt_dir / pt.name)

    csv_candidates = (
        list(src.glob("results_*.csv"))
        + list(src.glob("centroid_margin*.csv"))
        + list(src.glob("nearest_other*.csv"))
    )
    for csv_path in csv_candidates:
        target_dir = results_dir if csv_path.name in _EVAL_CSV_NAMES else outputs_dir
        shutil.copy2(csv_path, target_dir / csv_path.name)

    for meta in src.glob("meta_*.json"):
        shutil.copy2(meta, meta_dir / meta.name)

    return dest_path
