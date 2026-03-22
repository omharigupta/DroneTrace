# High-Level Design (HLD)
## Transformer-Based UAV Forensic Anomaly Detection System

**Document Version:** 1.0  
**Date:** March 2026  
**Author:** Dissertation Project  

---

## 1. System Overview

This system performs **forensic anomaly detection** on Unmanned Aerial Vehicle (UAV) flight telemetry data using a Transformer Encoder architecture. It ingests raw flight evidence from DJI drones and NIST forensic datasets, processes them through a multi-stage pipeline, and classifies flight segments as **Normal** or **Tampered/Anomalous**.

The key innovation over traditional CNN-LSTM approaches is the use of **Multi-Head Self-Attention**, which captures **long-range temporal dependencies** across entire flight windows — eliminating the "forgetfulness" inherent in recurrent architectures.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FORENSIC DATA PIPELINE                          │
├─────────────┬─────────────┬──────────────────┬─────────────────────┤
│  INGESTION  │ TRANSFORM   │   AI ENGINE      │   DECISION          │
│  LAYER      │ LAYER       │   (Transformer)  │   LAYER             │
├─────────────┼─────────────┼──────────────────┼─────────────────────┤
│ DJI .DAT    │ Sliding     │ Positional       │ Anomaly Score       │
│ NIST .TXT   │ Window      │ Encoding         │ Generator           │
│ DatCon CSV  │ (10-sec)    │                  │                     │
│ Parsing     │             │ Multi-Head       │ One-Class SVM /     │
│             │ Two-Stage   │ Self-Attention   │ Dynamic Threshold   │
│ SHA-256     │ Norm (MIN)  │ (4-8 layers)     │                     │
│ Integrity   │             │                  │ Binary:             │
│ Check       │ Min-Max     │ Feed-Forward     │ "Normal" /          │
│             │ Scaling     │ Network          │ "Tampered"          │
└─────────────┴─────────────┴──────────────────┴─────────────────────┘
```

---

## 3. Component Description

### 3.1 Ingestion Layer
| Component | Description |
|-----------|-------------|
| DJI .DAT Parser | Reads DJI Mini 2 binary `.DAT` files converted via DatCon to structured CSVs |
| NIST Parser | Reads NIST drone forensic image `.TXT` telemetry files |
| SHA-256 Verifier | Computes and validates cryptographic hashes of input evidence files to ensure chain-of-custody integrity |

### 3.2 Transformation Layer
| Component | Description |
|-----------|-------------|
| Sliding Window | Segments continuous telemetry into 10-second overlapping windows (configurable stride) |
| Two-Stage Normalization | **Stage 1:** Mean Instance Normalization (MIN) to dampen high-frequency sensor fluctuations. **Stage 2:** Min-Max scaling to [0, 1] for model compatibility |
| Feature Selector | Extracts 5 core features: Latitude, Longitude, Altitude, Speed, MotorRPM |

### 3.3 AI Engine — Transformer Encoder
| Component | Description |
|-----------|-------------|
| Feature Embedding | Projects 5-dimensional input into `d_model`-dimensional latent space |
| Positional Encoding | Sinusoidal encoding to inject temporal ordering information |
| Encoder Stack | 4–8 layers of Multi-Head Self-Attention + Feed-Forward blocks with residual connections and LayerNorm |
| Classification Head | Global average pooling → Dense → Sigmoid for binary output |

### 3.4 Decision Layer
| Component | Description |
|-----------|-------------|
| Anomaly Score Generator | Produces a continuous anomaly score ∈ [0, 1] per flight window |
| Threshold Engine | One-Class SVM or dynamic percentile threshold to convert scores into binary decisions |
| Forensic Report | Generates a timestamped report with flagged anomalous windows and confidence scores |

---

## 4. Data Flow Diagram

```
Raw Evidence (.DAT / .TXT)
        │
        ▼
  ┌─────────────┐
  │ SHA-256 Hash │──── Integrity Log
  │ Verification │
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │   DatCon /   │
  │  NIST Parser │
  └──────┬──────┘
         │  Structured CSV
         ▼
  ┌─────────────┐
  │   Sliding    │
  │   Window     │
  │  (10-sec)    │
  └──────┬──────┘
         │  [batch, seq_len, 5]
         ▼
  ┌─────────────┐
  │  Two-Stage   │
  │ Normalization│
  └──────┬──────┘
         │
         ▼
  ┌──────────────────┐
  │  TRANSFORMER     │
  │  ENCODER         │
  │  (Self-Attention)│
  └──────┬───────────┘
         │  Anomaly Score
         ▼
  ┌─────────────┐
  │  Threshold   │
  │  Decision    │
  └──────┬──────┘
         │
         ▼
   "Normal" / "Tampered"
```

---

## 5. Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| ML Framework | TensorFlow 2.x / Keras (with `tensorflow-metal` for M1 acceleration) |
| Data Processing | Pandas, NumPy, SciPy |
| Anomaly Detection | scikit-learn (One-Class SVM) |
| Visualization | Matplotlib, Seaborn |
| Hashing | hashlib (SHA-256) |
| Evaluation | scikit-learn (metrics), custom mAP |

---

## 6. Deployment Constraints

- **Target Hardware:** Apple MacBook M1 (8GB+ RAM)
- **Performance Requirement:** Process 10-minute flight log in < 5 seconds
- **Evidence Integrity:** All input files must pass SHA-256 verification before processing
- **Model Size:** Optimized for edge inference (< 50MB model file)

---

## 7. Non-Functional Requirements (Summary)

| ID | Requirement |
|----|-------------|
| NFR-01 | Processing latency < 5 seconds for 10-minute flight |
| NFR-02 | SHA-256 hash-based chain-of-custody verification |
| NFR-03 | Cross-platform compatibility (macOS M1, Linux, Windows) |
| NFR-04 | Modular architecture for easy model swapping |
