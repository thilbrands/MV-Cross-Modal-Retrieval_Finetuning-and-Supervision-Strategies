"""
Zentrale Pfad-Config für die Cluster-Pipeline (work2).

Alle festen Pfade stehen hier; run-spezifische Dinge (z. B. welcher Dataset-Run)
werden per Umgebungsvariable oder Default „neuester Run“ gesteuert.
"""
from pathlib import Path


# Fester Work-Root auf dem Cluster (nicht verhandelbar)
WORK_ROOT = Path("/work2/ra39oxet-DatasetAudioSetSubset")

# Alle Dataset-Runs liegen darunter (pro Run ein Ordner: Datum_Uhrzeit_audioset)
DATASETS_ROOT = WORK_ROOT / "datasets"

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

# Projektions-Heads (Training), z. B. pro Run oder zentral
PROJECTION_HEADS_PATH = WORK_ROOT / "projection_heads.pt"


def get_latest_run_name() -> str | None:
    """
    Name des zuletzt erstellten/geänderten Dataset-Runs (Ordner unter DATASETS_ROOT).
    Nutzbar als Default für Extract und (später) Training, wenn DATASET_RUN_NAME
    nicht gesetzt ist.
    """
    if not DATASETS_ROOT.exists():
        return None
    runs = [p for p in DATASETS_ROOT.iterdir() if p.is_dir()]
    if not runs:
        return None
    # Neuester zuerst (nach mtime)
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0].name
