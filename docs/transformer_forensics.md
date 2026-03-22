# Transformer-Based Drone Forensics Engine

## Overview
A state-of-the-art (2026) forensic anomaly detection system for UAV flight telemetry, leveraging Multi-Head Self-Attention to capture global temporal dependencies that recurrent models (LSTM) miss.

---

## Pre-processing
- **Sliding Window:** 10-second segments at 10 Hz = 100 timesteps per window. 50% overlap (stride = 50).
- **Normalization:** Two-Stage approach:
  - *Stage 1:* Mean Instance Normalization (MIN) — removes per-flight sensor bias.
  - *Stage 2:* Min-Max scaling of Latitude, Longitude, Altitude, Speed, MotorRPM to [0, 1].
- **Integrity:** SHA-256 hash verification of all input evidence files before processing.

---

## Model Architecture

### Input
- Shape: `(batch_size, seq_len=100, features=5)`
- Features: `[Latitude, Longitude, Altitude, Speed, MotorRPM]`

### Embedding Layer
- Projects 5 raw features into a higher-dimensional space (`d_model=64`)
- Implemented as a Dense linear projection

### Positional Encoding
- Sinusoidal encoding (Vaswani et al., 2017)
- Injects temporal ordering: the model knows that timestep 1 comes before timestep 100
- Formula:
  - $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$
  - $PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$

### Encoder Blocks (×4–8)
Each block contains:
1. **Multi-Head Self-Attention** (4 heads)
   - Computes: $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
   - Captures global flight context: correlates battery drops with GPS shifts across the entire window
2. **Residual Connection + Layer Normalization**
3. **Position-wise Feed-Forward Network** (Dense → ReLU → Dense)
4. **Residual Connection + Layer Normalization**
5. **Dropout** (0.1) for regularization

### Classification Head
- **Global Average Pooling** across the sequence dimension → `(batch, d_model)`
- **Dense(64, ReLU)** → **Dropout(0.3)** → **Dense(1, Sigmoid)**
- Output: Anomaly probability ∈ [0, 1]

---

## Anomaly Detection

### Dynamic Threshold
- Compute anomaly scores on validation set (normal flights only)
- Threshold = $\mu + k\sigma$ (default $k = 2.0$)
- Score > threshold → **Tampered**

### One-Class SVM (Alternative)
- Trained on Transformer penultimate-layer embeddings of normal flights
- RBF kernel, $\nu = 0.05$
- Decision < 0 → **Tampered**

---

## Training

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=1e-4) |
| Schedule | Cosine decay with 500-step warmup |
| Loss | Binary Cross-Entropy |
| Batch Size | 32 |
| Epochs | 50 (early stopping, patience=10) |
| Split | 70% train / 15% val / 15% test |

---

## Evaluation

### Metrics
- **Accuracy** — overall correctness
- **Precision** — false alarm rate (critical for forensics)
- **Recall** — attack detection rate (must not miss tampering)
- **F1-Score** — harmonic mean of precision and recall
- **mAP** — mean average precision across all threshold values

### Confusion Matrix Interpretation
| | Predicted Normal | Predicted Tampered |
|---|---|---|
| **Actual Normal** | True Negative (correct) | False Positive (pilot error flagged) |
| **Actual Tampered** | False Negative (**CRITICAL MISS**) | True Positive (attack detected) |

### Inference Speed
- Measured in **milliseconds per sequence** for real-time forensic use
- Target: < 50ms per 10-second window on Apple M1

---

## Baseline Comparison
- **Random Forest** with hand-crafted statistical features (mean, std, min, max, delta per feature)
- Purpose: Demonstrate the Transformer's advantage in capturing temporal patterns over flat feature vectors

---

## File Structure
```
dron/
├── config.py                          # All hyperparameters
├── main.py                            # Entry point
├── requirements.txt                   # Dependencies
├── docs/                              # This documentation
├── src/
│   ├── data/
│   │   ├── preprocessing.py           # Parse, window, normalize
│   │   └── dataset.py                 # TF Dataset builder
│   ├── models/
│   │   ├── positional_encoding.py     # Sinusoidal PE
│   │   ├── transformer_model.py       # Full encoder model
│   │   └── anomaly_detector.py        # OC-SVM + threshold
│   ├── training/
│   │   ├── train.py                   # Training pipeline
│   │   └── baseline.py               # Random Forest
│   ├── evaluation/
│   │   ├── metrics.py                 # Acc, F1, mAP, CM
│   │   └── inference.py              # Speed benchmarking
│   └── utils/
│       └── hash_verify.py             # SHA-256 integrity
└── data/
    ├── raw/                           # Original evidence
    └── processed/                     # Windowed, normalized
```
