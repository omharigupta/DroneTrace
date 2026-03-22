# Project Plan — Dissertation Timeline
## Transformer-Based UAV Forensic Anomaly Detection

**Duration:** 10 Weeks  
**Start Date:** March 2026  
**Target Submission:** May 2026  

---

## Phase Overview

```
Week  1 ──── 2 ──── 3 ──── 4 ──── 5 ──── 6 ──── 7 ──── 8 ──── 9 ──── 10
      │      │      │      │      │      │      │      │      │      │
      ├──RESEARCH──┤      │      │      │      │      │      │      │
      │            ├SETUP─┤      │      │      │      │      │      │
      │            │      ├────DATA─────┤      │      │      │      │
      │            │      │             ├─────TRAINING──────┤      │
      │            │      │             │                   ├──WRITING──┤
```

---

## Detailed Timeline

| Phase | Week | Task | Deliverable | Status |
|-------|------|------|-------------|--------|
| **Research** | 1 | Literature review: Transformer-based anomaly detection in UAVs (2024–2026 papers) | Annotated bibliography | ☐ |
| **Research** | 2 | Study: Self-Attention mechanisms, Positional Encoding variants, One-Class SVM for anomaly scoring | Literature Review chapter draft | ☐ |
| **Setup** | 3 | Install `tensorflow-metal` on MacBook M1. Set up Python environment, VS Code, Git. Verify GPU acceleration | Environment Configuration Log | ☐ |
| **Data** | 4 | Extract NIST drone forensic telemetry files. Parse DJI Mini 2 DatCon CSVs. Unify schema | Raw data inventory | ☐ |
| **Data** | 5 | Clean data: handle NaNs, remove duplicates. Apply Two-Stage Normalization. Create sliding windows. Train/Val/Test split (70/15/15) | Pre-processed CSVs, `data/processed/` | ☐ |
| **Training** | 6 | Implement Transformer Encoder model. Train on preprocessed data. Monitor loss curves | Initial model checkpoint | ☐ |
| **Training** | 7 | Hyperparameter tuning: vary `n_layers`, `n_heads`, `d_model`, `lr`. Implement LR scheduler with warmup | Tuning results table | ☐ |
| **Training** | 8 | Train Random Forest baseline. Compare Transformer vs RF: Accuracy, F1, mAP. Run One-Class SVM anomaly detector | Comparison metrics table | ☐ |
| **Writing** | 9 | Finalize HLD/LLD diagrams. Write Methodology and Results chapters. Create confusion matrix & ROC plots | Draft chapters (Methodology, Results) | ☐ |
| **Writing** | 10 | Write Introduction, Conclusion, Future Work. Format references (IEEE). Proofread and submit | Final Dissertation PDF | ☐ |

---

## Key Milestones

| Milestone | Target Date | Dependency |
|-----------|------------|------------|
| Literature review complete | End of Week 2 | — |
| Environment validated (TF + Metal) | End of Week 3 | — |
| Data preprocessed and verified | End of Week 5 | Environment ready |
| Transformer model trained | End of Week 7 | Data ready |
| All experiments complete | End of Week 8 | Model trained |
| Final submission | End of Week 10 | All chapters written |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `tensorflow-metal` incompatibility | Medium | High | Fallback to CPU training or Google Colab |
| Insufficient labeled "tampered" data | High | High | Generate synthetic anomalies (GPS spoofing, altitude injection) |
| Transformer overfitting on small dataset | Medium | Medium | Aggressive dropout (0.3), early stopping, data augmentation |
| M1 GPU OOM during training | Low | Medium | Reduce batch size, use gradient accumulation |
| NIST dataset format changes | Low | Low | Flexible parser with column auto-detection |

---

## Resource Requirements

| Resource | Specification |
|----------|--------------|
| Hardware | Apple MacBook M1, 8GB+ RAM |
| GPU | Apple Metal (via `tensorflow-metal`) |
| Cloud (backup) | Google Colab Pro (T4/A100 GPU) |
| Software | Python 3.10, TensorFlow 2.x, scikit-learn, Pandas |
| Data | NIST CFReDS drone images, Kaggle UAV telemetry |
| Version Control | Git + GitHub |

---

## Weekly Checkpoints

Each week should conclude with:
1. **Code commit** — All new code pushed to Git with descriptive messages
2. **Metrics log** — Any training/evaluation metrics saved to `logs/`
3. **Notes update** — Brief summary of progress and blockers
4. **Supervisor review** — If applicable, share progress for feedback
