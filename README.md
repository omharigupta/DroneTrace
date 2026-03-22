# DroneTrace — UAV Forensic Anomaly Detection

<p align="center">
  <b>Detect telemetry tampering in drone flight logs using deep learning</b>
</p>

A Transformer-based forensic pipeline for **detecting tampered UAV telemetry**. Built with **LoRA** (Low-Rank Adaptation) for parameter-efficient training, automatic **checkpoint resume**, a **web dashboard** for evidence upload & analysis, and a **forensic evidence checker** that combines model predictions with statistical reference validation.

---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Kaggle Datasets](#kaggle-datasets)
6. [Quick Start](#quick-start)
7. [Dashboard](#dashboard)
8. [Training](#training)
9. [Checkpoint Resume](#checkpoint-resume)
10. [Evidence Checking](#evidence-checking)
11. [CLI Reference](#cli-reference)
12. [Configuration](#configuration)
13. [Architecture](#architecture)

---

## Features

| Capability | Details |
|---|---|
| **Transformer Encoder** | 4-layer multi-head self-attention model for time-series classification |
| **LoRA** | Low-Rank Adaptation (rank 8) on Q/K/V/O attention + FFN layers — trains ~5% of parameters |
| **Checkpoint Resume** | Automatically finds the latest checkpoint and resumes training (no retraining from scratch) |
| **Dual Dataset Support** | Kaggle tampering dataset (training) + supplemental operations log (reference validation) |
| **Forensic Evidence Checker** | SHA-256 hashing ➜ Transformer classification ➜ reference profile deviation ➜ combined verdict |
| **Synthetic Fallback** | Generates synthetic telemetry data if Kaggle datasets are not available |
| **Web Dashboard** | Flask dashboard — drag & drop evidence files, get instant forensic verdicts |
| **Baseline Comparison** | Random Forest baseline with hand-crafted statistical features |

---

## Project Structure

```
DroneTrace/
├── main.py                         # CLI entry point — orchestrates the full pipeline
├── dashboard.py                    # Flask web dashboard for evidence analysis
├── config.py                       # All hyperparameters, paths, thresholds
├── requirements.txt                # Python dependencies
├── .gitignore
│
├── templates/
│   └── index.html                  # Dashboard UI (dark theme, drag & drop upload)
│
├── src/
│   ├── models/
│   │   ├── transformer_model.py    # Transformer encoder + classifier (LoRA-enabled)
│   │   ├── lora.py                 # LoRA layers (LoRADense, LoRAMultiHeadAttention)
│   │   ├── positional_encoding.py  # Sinusoidal positional encoding
│   │   └── anomaly_detector.py     # Dynamic threshold + One-Class SVM ensemble
│   │
│   ├── data/
│   │   ├── preprocessing.py        # Loaders for Kaggle tampering, reference, DJI, NIST formats
│   │   └── dataset.py              # tf.data.Dataset builder with train/val/test splits
│   │
│   ├── training/
│   │   ├── train.py                # Training loop with checkpoint resume, callbacks
│   │   └── baseline.py             # Random Forest baseline
│   │
│   ├── evaluation/
│   │   ├── metrics.py              # Accuracy, F1, confusion matrix, ROC, PR curves
│   │   └── inference.py            # Latency benchmarking (NFR-01)
│   │
│   └── utils/
│       ├── hash_verify.py          # SHA-256 chain-of-custody hashing
│       └── reference_validator.py  # ForensicEvidenceChecker (model + reference validation)
│
├── docs/                           # Design documents (HLD, LLD, specs)
│
├── data/                           # ⬇ NOT tracked in git
│   └── raw/
│       ├── tampering/              # Kaggle tampering dataset goes here
│       ├── reference/              # Kaggle supplemental operations log goes here
│       └── evidence/               # Your evidence files to check go here
│
├── models_saved/                   # ⬇ NOT tracked in git — saved checkpoints
├── logs/                           # ⬇ NOT tracked in git — training logs, hash logs
└── venv/                           # ⬇ NOT tracked in git — Python virtual environment
```

---

## Prerequisites

- **Python 3.10 – 3.11** (3.11 recommended for TensorFlow compatibility)
- **pip**
- **Git**

> **Note:** Python 3.12+ may not have full TensorFlow wheel support yet. Stick with 3.11.

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/omharigupta/DroneTrace.git
cd DroneTrace

# 2. Create a virtual environment (Python 3.11)
python -m venv venv

# 3. Activate it
# Windows:
.\venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## Kaggle Datasets

The system uses two Kaggle datasets. **Both are optional** — if not present, synthetic data is generated automatically.

### Dataset 1 — Tampering Dataset (Training)

- **URL:** https://www.kaggle.com/datasets/rasikaekanayakadevlk/drone-telemetry-tampering-dataset-v2
- **Size:** ~1.48 GB
- **Place in:** `data/raw/tampering/`
- After extraction, the dataset nests inside `drone_temparing_dataset_v2/` with severity sub-folders.
- The system **auto-detects** CSVs recursively — no manual file renaming needed.
- Columns: `case_id, row_idx, label, tamper_type, latitude, longitude, altitude, speed, heading, ...`
- Severity sub-folders: `balanced/`, `strong/`, `subtle/` (each with `rep_00/`, `rep_01/`, etc.)

### Dataset 2 — Supplemental Operations Log (Reference)

- **URL:** https://www.kaggle.com/datasets/samsudeenashad/supplemental-drone-telemetry-data-and-operations-log
- **Size:** ~91 KB
- **Place in:** `data/raw/reference/`
- The main file should be: `data/raw/reference/Supplemental Drone Telemetry Data - Drone Operations Log.csv`

### Download via Kaggle CLI

```bash
pip install kaggle
kaggle datasets download -d rasikaekanayakadevlk/drone-telemetry-tampering-dataset-v2 -p data/raw/tampering/ --unzip
kaggle datasets download -d samsudeenashad/supplemental-drone-telemetry-data-and-operations-log -p data/raw/reference/ --unzip
```

---

## Quick Start

### Train with synthetic data (no downloads needed)

```bash
python main.py
```

This generates synthetic telemetry data, trains the Transformer, evaluates metrics, and saves the model.

### Train with Kaggle datasets

```bash
python main.py --kaggle
```

### Launch the web dashboard

```bash
python dashboard.py
# Open http://127.0.0.1:5000 in your browser
```

### Check a single evidence file (CLI)

```bash
python main.py --mode check --check path/to/evidence_file.csv
```

---

## Training

### Full pipeline (train + evaluate)

```bash
# Synthetic data (default)
python main.py --mode full --epochs 30

# Kaggle tampering dataset
python main.py --kaggle --mode full --epochs 50

# Specific severity level
python main.py --kaggle --severity subtle --epochs 30

# Limit number of flight cases (for quick testing)
python main.py --kaggle --max-cases 100 --epochs 10
```

### Train-only mode

```bash
python main.py --mode train --kaggle --epochs 50
```

### What happens during training

1. **Data loading** — reads Kaggle CSV or generates synthetic flights
2. **Preprocessing** — groups by case_id, applies sliding windows (100 timesteps, 50% overlap)
3. **Dataset split** — 70% train / 15% validation / 15% test
4. **Model build** — Transformer encoder with LoRA adaptation
5. **Training** — with cosine warmup LR schedule and 4 callbacks:
   - `EarlyStopping` (patience=10, monitors val_loss)
   - `ModelCheckpoint` — saves best model to `models_saved/transformer_best.keras`
   - `ModelCheckpoint` — saves every epoch as `models_saved/checkpoint_epoch_XX.keras`
   - `CSVLogger` — logs to `logs/training_log.csv`

---

## Checkpoint Resume

**Training automatically resumes from the last checkpoint.** You never retrain from scratch unless you explicitly ask.

```bash
# Resume training (default behavior)
python main.py --kaggle --epochs 50

# Force training from scratch (ignore checkpoints)
python main.py --kaggle --epochs 50 --no-resume
```

### How it works

1. On startup, the system searches `models_saved/` for:
   - `transformer_best.keras` (best validation model)
   - `checkpoint_epoch_*.keras` (periodic checkpoints)
2. It picks the latest checkpoint and extracts the epoch number.
3. Training resumes from that epoch with `model.fit(initial_epoch=N)`.
4. The CSVLogger appends to the existing log rather than overwriting.

---

## Dashboard

DroneTrace includes a **web-based dashboard** for uploading and analyzing evidence files through your browser.

### Start the dashboard

```bash
python dashboard.py
```

Then open **http://127.0.0.1:5000**.

### Features

- **Drag & drop** file upload (or click to browse)
- **Real-time analysis** — SHA-256 hash → Transformer → Reference → Verdict
- **Visual results** — confidence ring, per-window score breakdown, tampering ratio bar
- **Report history** — view past forensic reports
- **Format guide** — shows accepted formats and sample CSV template right in the UI

### Accepted file formats

| Format | Extension | Key columns |
|---|---|---|
| Tampering Dataset | `.csv` | `case_id`, `row_idx`, `label`, `latitude`, `longitude`, `altitude`, `speed` |
| DJI DatCon Export | `.csv` | `GPS:Lat`, `GPS:Long`, `GPS:heightMSL`, `Motor:Speed:*` |
| NIST Drone Forensics | `.txt` `.dat` | `lat`, `lon`, `altitude`, `groundspeed` |
| Generic CSV | `.csv` | `latitude`, `longitude`, `altitude` (minimum required) |

> **Minimum requirement:** At least `latitude`, `longitude`, and `altitude` columns with ~100+ rows.

### Screenshot workflow

1. Upload a `.csv` / `.txt` / `.dat` telemetry file
2. Dashboard runs the 5-step forensic check automatically
3. View the verdict: **CLEAN** / **TAMPERED** / **SUSPICIOUS** with confidence percentage
4. Inspect per-window Transformer scores and reference validation details

---

## Evidence Checking

After training, you can check new telemetry files for tampering:

### Check a single file

```bash
python main.py --mode check --check data/raw/evidence/suspect_flight.csv
```

### Check an entire directory

```bash
python main.py --mode check --check-dir data/raw/evidence/
```

### How evidence checking works

For each file, the `ForensicEvidenceChecker` runs 5 steps:

1. **SHA-256 hash** — records file integrity for chain-of-custody
2. **Load & preprocess** — parses telemetry into sliding windows
3. **Transformer classification** — model predicts tamper probability for each window
4. **Reference validation** — compares telemetry statistics against the known-good reference profile
5. **Combined verdict** — merges model confidence + reference deviation into final result

**Verdicts:**

| Verdict | Meaning |
|---|---|
| `CLEAN` | No tampering detected |
| `TAMPERED` | High confidence of tampering |
| `SUSPICIOUS` | Some anomalies found — manual review recommended |

---

## CLI Reference

```
python main.py [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--mode {full,train,evaluate,check}` | `full` | Pipeline mode. `check` = evidence-only (no training) |
| `--kaggle` | off | Use Kaggle datasets instead of synthetic data |
| `--severity {balanced,strong,subtle}` | None | Load a specific tampering severity level |
| `--max-cases N` | None | Limit flight cases to load (for quick testing) |
| `--data-dir PATH` | `data/raw` | Directory containing telemetry files |
| `--synthetic / --no-synthetic` | on | Use synthetic data generation |
| `--num-normal N` | 500 | Number of synthetic normal samples |
| `--num-tampered N` | 100 | Number of synthetic tampered samples |
| `--epochs N` | 30 | Maximum training epochs |
| `--resume / --no-resume` | on | Resume from checkpoint or train from scratch |
| `--check FILE` | None | Path to a single evidence file to check |
| `--check-dir DIR` | None | Directory of evidence files to check |

---

## Configuration

All hyperparameters are centralized in `config.py`:

| Parameter | Value | Description |
|---|---|---|
| `WINDOW_SIZE` | 100 | Timesteps per sliding window (10s × 10 Hz) |
| `STRIDE` | 50 | Window stride (50% overlap) |
| `D_MODEL` | 64 | Transformer embedding dimension |
| `NUM_HEADS` | 4 | Multi-head attention heads |
| `NUM_ENCODER_LAYERS` | 4 | Transformer encoder blocks |
| `D_FF` | 128 | Feed-forward inner dimension |
| `DROPOUT_RATE` | 0.1 | Encoder dropout |
| `BATCH_SIZE` | 32 | Training batch size |
| `EPOCHS` | 30 | Default max epochs |
| `LEARNING_RATE` | 1e-4 | Adam optimizer LR |
| `USE_LORA` | True | Enable LoRA adaptation |
| `LORA_RANK` | 8 | LoRA rank (lower = fewer params) |
| `LORA_ALPHA` | 8.0 | LoRA scaling factor |

---

## Architecture

```
Input (batch, 100, 5)
    │
    ▼
Positional Encoding (sinusoidal)
    │
    ▼
┌─────────────────────────────────┐
│  Transformer Encoder Block ×4   │
│  ┌───────────────────────────┐  │
│  │ LoRA Multi-Head Attention │  │
│  │ (4 heads, rank=8)         │  │
│  └───────────────────────────┘  │
│  ┌───────────────────────────┐  │
│  │ LoRA Feed-Forward Network │  │
│  │ (64 → 128 → 64, rank=8)  │  │
│  └───────────────────────────┘  │
│  + LayerNorm + Residual + Drop  │
└─────────────────────────────────┘
    │
    ▼
Global Average Pooling
    │
    ▼
Dense(64) → Dropout(0.3) → Dense(1, sigmoid)
    │
    ▼
Output: tamper probability [0, 1]
```

---

## License

This project is for academic/research purposes.

---

<p align="center">
  <b>DroneTrace</b> — Transformer + LoRA forensic anomaly detection for UAV telemetry<br>
  <a href="https://github.com/omharigupta/DroneTrace">github.com/omharigupta/DroneTrace</a>
</p>
