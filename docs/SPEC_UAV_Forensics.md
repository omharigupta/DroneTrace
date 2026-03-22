# Software Requirements Specification (SRS)
## Transformer-Based UAV Forensic Anomaly Detection System

**Document Version:** 1.0  
**Date:** March 2026  
**Author:** Dissertation Project  
**Standard:** IEEE 830-1998 (adapted)

---

## 1. Introduction

### 1.1 Purpose
This document specifies the functional and non-functional requirements for a Transformer-based anomaly detection system designed for UAV (drone) flight telemetry forensics. The system supports digital forensic investigations by identifying tampered or anomalous flight data in DJI drone logs.

### 1.2 Scope
The system processes raw drone telemetry evidence, applies AI-driven analysis using a Transformer Encoder, and classifies flight segments as Normal or Tampered. It is designed to assist forensic investigators in chain-of-custody verification and flight data integrity assessment.

### 1.3 Definitions & Acronyms

| Term | Definition |
|------|-----------|
| UAV | Unmanned Aerial Vehicle |
| MHA | Multi-Head Attention |
| MIN | Mean Instance Normalization |
| mAP | Mean Average Precision |
| NIST | National Institute of Standards and Technology |
| .DAT | DJI proprietary binary flight log format |
| DatCon | Tool for converting DJI .DAT to CSV |
| PE | Positional Encoding |

---

## 2. Functional Requirements

| ID | Category | Requirement Description | Priority |
|----|----------|------------------------|----------|
| FR-01 | Data Extraction | Support for DJI Mini 2 `.DAT` file parsing via DatCon CSV output | **Must** |
| FR-02 | Data Extraction | Support for NIST drone forensic dataset `.TXT` file parsing | **Must** |
| FR-03 | Data Extraction | Automatic detection of input file format (DJI CSV vs NIST TXT) | Should |
| FR-04 | Preprocessing | Sliding window segmentation with configurable window size (default: 10s) and stride (default: 50% overlap) | **Must** |
| FR-05 | Preprocessing | Two-Stage Normalization: Mean Instance Normalization followed by Min-Max scaling | **Must** |
| FR-06 | Preprocessing | Feature extraction of Latitude, Longitude, Altitude, Speed, and MotorRPM | **Must** |
| FR-07 | ML Model | Implementation of a Transformer Encoder architecture with Multi-Head Self-Attention | **Must** |
| FR-08 | ML Model | Sinusoidal Positional Encoding for temporal ordering | **Must** |
| FR-09 | ML Model | Configurable number of encoder layers (4–8), attention heads, and embedding dimensions | Should |
| FR-10 | ML Model | Binary classification output: Normal (0) vs Tampered (1) via sigmoid activation | **Must** |
| FR-11 | Anomaly Detection | One-Class SVM anomaly detector on Transformer embeddings | Should |
| FR-12 | Anomaly Detection | Dynamic threshold anomaly scoring (μ + kσ method) | **Must** |
| FR-13 | Baseline | Random Forest classifier for comparative benchmarking | **Must** |
| FR-14 | Metrics | Real-time calculation of Accuracy, Precision, Recall, and F1-Score | **Must** |
| FR-15 | Metrics | Calculation of Mean Average Precision (mAP) across threshold values | **Must** |
| FR-16 | Metrics | Generation of Confusion Matrix visualization | **Must** |
| FR-17 | Metrics | Inference speed measurement (ms per sequence) | Should |
| FR-18 | Integrity | SHA-256 hash computation and verification of all input evidence files | **Must** |
| FR-19 | Reporting | Export of anomaly detection results with timestamps and confidence scores | Should |
| FR-20 | Training | Early stopping with configurable patience to prevent overfitting | **Must** |

---

## 3. Non-Functional Requirements

| ID | Category | Requirement Description | Metric |
|----|----------|------------------------|--------|
| NFR-01 | Performance | Must process a 10-minute flight log in < 5 seconds | Latency < 5000ms on Apple M1 |
| NFR-02 | Security | Hash-based verification (SHA-256) of all input log files before processing | 100% pre-processing verification |
| NFR-03 | Accuracy | Minimum F1-Score of 0.85 on NIST test dataset | F1 ≥ 0.85 |
| NFR-04 | Scalability | Support batch processing of up to 100 flight logs sequentially | Batch completion without OOM |
| NFR-05 | Portability | Cross-platform support: macOS (M1/M2), Linux (x86_64), Windows 10+ | Verified on 3 platforms |
| NFR-06 | Maintainability | Modular codebase with < 300 LOC per module | Code review compliance |
| NFR-07 | Usability | CLI interface with clear progress indicators and error messages | No unhandled exceptions |
| NFR-08 | Reproducibility | Fixed random seeds for all stochastic operations | Deterministic results |
| NFR-09 | Model Size | Trained model file < 50 MB for edge deployment | File size verification |
| NFR-10 | Documentation | All public functions include docstrings with parameter descriptions | 100% docstring coverage |

---

## 4. System Interfaces

### 4.1 Input Interfaces
| Interface | Format | Description |
|-----------|--------|-------------|
| DJI Flight Log | `.csv` (from DatCon) | Columnar telemetry with timestamp, GPS, altitude, speed, motor data |
| NIST Evidence | `.txt` | Tab/comma-separated telemetry from NIST drone forensic images |
| Configuration | `config.py` | Python module with hyperparameters, paths, and thresholds |

### 4.2 Output Interfaces
| Interface | Format | Description |
|-----------|--------|-------------|
| Predictions | `.csv` | Per-window anomaly scores and binary classifications |
| Metrics Report | Console + `.json` | Accuracy, F1, mAP, confusion matrix |
| Model Checkpoint | `.keras` / `.h5` | Trained Transformer model weights |
| Hash Log | `.json` | SHA-256 hashes of all processed evidence files |

---

## 5. Constraints

1. **Hardware:** Primary development on Apple MacBook M1 with `tensorflow-metal` acceleration.
2. **Data:** Limited to publicly available NIST forensic datasets and Kaggle UAV telemetry (no classified data).
3. **Framework:** TensorFlow 2.x / Keras (not PyTorch) for Apple Silicon GPU support.
4. **Ethical:** System is designed for forensic assistance only — not autonomous legal decision-making.
5. **Academic:** Must comply with university dissertation guidelines and ethical approval.

---

## 6. Assumptions & Dependencies

| # | Assumption / Dependency |
|---|------------------------|
| 1 | DatCon has already converted `.DAT` files to `.csv` before ingestion |
| 2 | NIST datasets are available in their published format |
| 3 | TensorFlow 2.x with Metal plugin is compatible with the target M1 hardware |
| 4 | Training data includes labeled "Normal" and "Tampered" flight segments |
| 5 | Sensor sampling rate is approximately 10 Hz (standard for DJI consumer drones) |

---

## 7. Acceptance Criteria

| Test ID | Requirement | Test Description | Pass Criteria |
|---------|-------------|------------------|---------------|
| TC-01 | FR-01 | Parse a DJI Mini 2 DatCon CSV | All 5 features extracted correctly |
| TC-02 | FR-07 | Train Transformer on sample data | Loss decreases over 10 epochs |
| TC-03 | FR-14 | Evaluate on test set | F1-Score computed and ≥ 0.85 |
| TC-04 | NFR-01 | Inference on 10-min log | Latency < 5 seconds |
| TC-05 | NFR-02 | Modify 1 byte of input file | SHA-256 mismatch detected |
| TC-06 | FR-13 | Run Random Forest baseline | Metrics computed for comparison |
