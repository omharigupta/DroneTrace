# Low-Level Design (LLD)
## Transformer-Based UAV Forensic Anomaly Detection System

**Document Version:** 1.0  
**Date:** March 2026  
**Author:** Dissertation Project  

---

## 1. Component Interaction Diagram

```
main.py
  │
  ├── config.py                    (Hyperparameters & paths)
  │
  ├── src/utils/hash_verify.py     (SHA-256 integrity check)
  │
  ├── src/data/preprocessing.py    (Parse, window, normalize)
  │     └── src/data/dataset.py    (TF Dataset / DataLoader)
  │
  ├── src/models/
  │     ├── positional_encoding.py (Sinusoidal PE)
  │     ├── transformer_model.py   (Full Encoder model)
  │     └── anomaly_detector.py    (One-Class SVM threshold)
  │
  ├── src/training/
  │     ├── train.py               (Transformer training loop)
  │     └── baseline.py            (Random Forest benchmark)
  │
  └── src/evaluation/
        ├── metrics.py             (Accuracy, F1, mAP, CM)
        └── inference.py           (Latency benchmarking)
```

---

## 2. Detailed Component Specifications

### 2.1 Positional Encoding

Since Transformers have **no inherent notion of sequence order**, we inject positional information using sinusoidal encoding:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Where:
- $pos$ = position in the flight window (0 to `seq_len - 1`)
- $i$ = dimension index
- $d_{model}$ = embedding dimension (default: 64)

**Implementation:** `src/models/positional_encoding.py`
- Class: `PositionalEncoding(tf.keras.layers.Layer)`
- Input shape: `(batch_size, seq_len, d_model)`
- Output shape: `(batch_size, seq_len, d_model)` (input + PE)

---

### 2.2 Self-Attention Mechanism

The core of the Transformer. For each attention head:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$ (Query) = "What am I looking for?" — the current flight event
- $K$ (Key) = "What do I contain?" — all other flight events
- $V$ (Value) = "What information do I carry?" — the actual sensor values
- $d_k$ = dimension of keys (= $d_{model} / n_{heads}$)

**Multi-Head Attention** runs $h$ parallel attention heads:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

**Implementation:** Uses `tf.keras.layers.MultiHeadAttention`
- `num_heads`: 4 (default)
- `key_dim`: `d_model // num_heads`

---

### 2.3 Transformer Encoder Block

Each encoder block consists of:

```
Input
  │
  ├──► MultiHeadAttention ──► Add & LayerNorm
  │         │                       │
  │         └───────────────────────┘ (Residual)
  │
  ├──► FeedForward (Dense→ReLU→Dense) ──► Add & LayerNorm
  │         │                                    │
  │         └────────────────────────────────────┘ (Residual)
  │
  ▼
Output
```

**Feed-Forward Network (per position):**

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

- Inner dimension (`d_ff`): 128 (default, = 2 × `d_model`)
- Dropout: 0.1 applied after attention and FFN

---

### 2.4 Full Model Architecture

```
Input: (batch, seq_len=100, features=5)
        │
        ▼
  ┌─────────────────┐
  │ Dense Embedding  │  → (batch, 100, d_model=64)
  │ (Linear proj.)   │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │   Positional     │  → (batch, 100, 64)
  │   Encoding       │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Encoder Block ×N │  N = 4 (default)
  │ (MHA + FFN)      │  → (batch, 100, 64)
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Global Average   │  → (batch, 64)
  │ Pooling          │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Dense(64, ReLU)  │  → (batch, 64)
  │ Dropout(0.3)     │
  │ Dense(1, Sigmoid)│  → (batch, 1)
  └─────────────────┘

Output: Anomaly probability ∈ [0, 1]
```

---

### 2.5 Preprocessing Logic

#### Two-Stage Normalization

**Stage 1 — Mean Instance Normalization (MIN):**

For each flight instance $x$ with features $f$:

$$x_{norm}^{(f)} = \frac{x^{(f)} - \mu_{instance}^{(f)}}{\sigma_{instance}^{(f)} + \epsilon}$$

This removes per-flight bias (e.g., different takeoff altitudes).

**Stage 2 — Global Min-Max Scaling:**

$$x_{scaled}^{(f)} = \frac{x_{norm}^{(f)} - \min_{global}^{(f)}}{\max_{global}^{(f)} - \min_{global}^{(f)}}$$

#### Sliding Window Segmentation

```python
window_size = 100     # 10 seconds at 10 Hz sampling
stride = 50           # 50% overlap
# Output shape per window: (100, 5)
```

---

### 2.6 Anomaly Thresholding

Two methods supported:

**Method 1 — Dynamic Threshold:**
- Compute anomaly scores on validation (normal-only) data
- Threshold = $\mu + k\sigma$ where $k$ = 2.0 (configurable)
- Scores above threshold → "Tampered"

**Method 2 — One-Class SVM:**
- Train on embeddings from the Transformer's penultimate layer
- Kernel: RBF, $\nu$ = 0.05
- Decision function < 0 → "Tampered"

---

### 2.7 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | Adam | Adaptive LR, standard for Transformers |
| Learning Rate | 1e-4 | Conservative for small dataset |
| LR Schedule | Cosine decay with warmup (500 steps) | Prevents early divergence |
| Loss | Binary Cross-Entropy | Binary classification task |
| Batch Size | 32 | Fits M1 8GB memory |
| Epochs | 50 (early stopping, patience=10) | Prevents overfitting |
| Validation Split | 20% | Standard hold-out |

---

### 2.8 Evaluation Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall correctness |
| Precision | $\frac{TP}{TP + FP}$ | False alarm rate |
| Recall | $\frac{TP}{TP + FN}$ | Attack detection rate |
| F1-Score | $\frac{2 \cdot P \cdot R}{P + R}$ | Harmonic mean |
| mAP | Mean Average Precision across thresholds | Threshold-independent ranking |
| Inference Latency | ms/sequence | Real-time feasibility |

---

### 2.9 Data Structures

```python
# Raw telemetry record
TelemetryRecord = {
    "timestamp": float,        # Unix epoch or relative seconds
    "latitude": float,         # Degrees
    "longitude": float,        # Degrees
    "altitude": float,         # Meters (AGL)
    "speed": float,            # m/s (ground speed)
    "motor_rpm": float         # Average of 4 motors
}

# Windowed sample
WindowedSample = {
    "data": np.ndarray,        # Shape: (seq_len, n_features)
    "label": int,              # 0 = Normal, 1 = Tampered
    "source_file": str,        # Evidence file path
    "window_start": float,     # Start timestamp
    "window_end": float        # End timestamp
}
```

---

### 2.10 Error Handling

| Scenario | Handling |
|----------|----------|
| SHA-256 mismatch | Reject file, log alert, abort pipeline |
| Missing sensor columns | Fill with NaN → forward-fill → flag |
| Sequence too short for window | Pad with zeros, set `padding_mask` |
| NaN in model input | Replace with 0.0, log warning |
| GPU OOM | Auto-reduce batch size by half, retry |
