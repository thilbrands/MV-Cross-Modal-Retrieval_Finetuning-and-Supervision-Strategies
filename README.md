# Music-Video Cross-Modal Retrieval

Code and curated results for the bachelor's thesis *"Music-Video Cross-Modal Retrieval: Finetuning and Supervision Strategies for Lightweight Encoder Alignment"*
Theo Hilbrands · Universität Leipzig · ScaDS.AI

## Repository structure

```
configs/                     # paths (cluster_env.sh), models, metrics
data/                        # data preparation and cleaning scripts
datasets/                    # versioned split CSV only (no media)
eval/                        # evaluation jobs
figures/                     # plots and frame exports
generalization_experiment/   # E4
pipelines/                   # run_dataset / run_train_and_eval / run_full
results/                     # curated thesis run (in git)
training/                    # training scripts for experiments E1–E3 (pair-based and genre-based training) + encoder finetuning
```

## Overview

Unlike everyday audio-visual data, where sound and image share a physical cause, the link between music and its video is driven by artistic factors — style, mood, genre. This project asks whether that correspondence can be captured cheaply, by aligning two already pretrained encoders rather than training new ones from scratch.

Concretely, it aligns CLIP (visual) and Wav2CLIP (audio) through lightweight projection heads and compares two supervision strategies:

- Pair-based (InfoNCE) — each video-audio pair is its own positive.
- Genre-based (SupCon) — all same-genre pairs count as positives.

On top of this, it finetunes the audio encoder (E3) and tests generalization to genres unseen during training (E4). The dataset is ~24,500 ten-second music-video segments across ten genres, built from AudioSet and filtered for visual relevance with a vision-language model.

What this repository contains: the reproducible pipeline and the curated results of the canonical run. It does not redistribute AudioSet media — segments are downloaded from YouTube on demand, and only split definitions and curated outputs are versioned.

## Look at results (no cluster)

```sh
ls results/2026-07-04_19-06/results/
```

Main CSVs: `results_evaluation.csv`, `results_evaluation_for_comparison_with_related_work.csv`, `results_quality_split.csv`. Checkpoints are under `checkpoints/`.

## Run on the Leipzig HPC (partition `paula`)

Do everything from the repo root. Large data lives under `WORK_ROOT` on `/work2` (not in git).

### 1. Set paths once

Edit the two defaults in `configs/cluster_env.sh`:

```sh
WORK_ROOT="${WORK_ROOT:-/work2/YOUR_PROJECT_DIR}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/venv/ba/bin/activate}"
```

(Or `export WORK_ROOT=...` / `export VENV_ACTIVATE=...` in the shell.)

### 2. One-time environment setup

```sh
module load Python/3.11.5-GCCcore-13.2.0
python3 -m venv ~/venv/ba
source ~/venv/ba/bin/activate
pip install -r requirements.txt

# AudioSet metadata → $WORK_ROOT/AudioSetData/  (not shipped in this repo)
# Download from Google AudioSet:
#   https://research.google.com/audioset/download.html
#   CSV:      https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv
#   Ontology: https://github.com/audioset/ontology/blob/master/ontology.json
# Place as: unbalanced_train_segments.csv  and  ontology.json

bash configs/check_setup.sh
```

### 3. Build dataset (download + embed)

YouTube download needs a Netscape-format `cookies.txt` in the repo root. Place `cookies.txt` in the repository root (next to `README.md`).

```sh
bash pipelines/run_dataset_pipeline.sh
```

### 4. Train + evaluate

```sh
# uses newest dataset under $WORK_ROOT/datasets/, or set DATASET_RUN_NAME=...
bash pipelines/run_train_and_eval.sh
```

Full stack (dataset then train+eval):

```sh
bash pipelines/run_full_pipeline.sh
```

### 5. Where outputs go

| What | Where |
|------|-------|
| Live run (checkpoints, plots, logs) | `$WORK_ROOT/training_runs/<run>/` |
| Curated package | `$WORK_ROOT/results/<run>/` |
| Slurm logs | `logs/` in the repo |

`bash configs/check_setup.sh` reports missing venv, metadata, or embeddings.

## Experiments

| Exp | Entry point |
|-----|-------------|
| E1 Pair heads | `training/pair_based_training.py` |
| E2 Genre heads | `training/genre_based_training.py` |
| E3a Audio encoder (pair) | `training/audio_encoder_pair_training.py` |
| E3b Audio encoder (genre) | `training/audio_encoder_genre_training.py` |
| E4 Generalization | `generalization_experiment/run_e4_exploration.sh`, `.../run_e4_interpolation.sh` |

### Single job example

```sh
source configs/cluster_env.sh
sbatch --export=WORK_ROOT,VENV_ACTIVATE,DATASET_RUN_NAME=2026-05-26_15-59-27_audioset,TRAINING_RUN_DIR=$WORK_ROOT/training_runs/<run> \
  eval/jobs/evaluation.sh
```

## Notes & limitations

- A full re-download is slow and only partially reproducible: some YouTube segments become unavailable over time, so an exact reconstruction of the dataset is not guaranteed.
- Method, metric definitions, and figures are documented in the thesis, not here.

Author: Theo Hilbrands, Universität Leipzig
