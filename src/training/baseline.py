"""
Random Forest Baseline for UAV Forensic Anomaly Detection.

Extracts hand-crafted statistical features from windowed telemetry
and trains a Random Forest classifier as a non-temporal baseline.

Purpose: Demonstrate the Transformer's advantage in capturing
temporal patterns over flat feature vectors.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import joblib

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logger = logging.getLogger(__name__)


def extract_statistical_features(X: np.ndarray) -> np.ndarray:
    """
    Extract hand-crafted statistical features from windowed telemetry.

    For each window (seq_len, num_features), computes per-feature:
    - Mean, Std, Min, Max, Median
    - Delta (max - min range)
    - Skewness, Kurtosis (simplified)

    Args:
        X: 3D array of shape (num_samples, seq_len, num_features).

    Returns:
        2D array of shape (num_samples, num_features * 8).
    """
    features_list = []

    for i in range(X.shape[0]):
        window = X[i]  # (seq_len, num_features)
        feats = []

        for f in range(window.shape[1]):
            col = window[:, f]
            feats.extend([
                np.mean(col),
                np.std(col),
                np.min(col),
                np.max(col),
                np.median(col),
                np.max(col) - np.min(col),  # Range (delta)
                float(np.mean(((col - np.mean(col)) / (np.std(col) + 1e-8)) ** 3)),  # Skewness
                float(np.mean(((col - np.mean(col)) / (np.std(col) + 1e-8)) ** 4)),  # Kurtosis
            ])

        features_list.append(feats)

    return np.array(features_list, dtype=np.float32)


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = config.RF_N_ESTIMATORS,
    max_depth: int = config.RF_MAX_DEPTH,
    random_state: int = config.RF_RANDOM_STATE,
    save_path: str = None,
) -> Dict[str, Any]:
    """
    Train a Random Forest baseline and evaluate.

    Args:
        X_train: Training data of shape (n_samples, seq_len, n_features).
        y_train: Training labels.
        X_test: Test data.
        y_test: Test labels.
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        random_state: Random seed.
        save_path: Path to save the trained model.

    Returns:
        Dictionary with model, predictions, and metrics.
    """
    if save_path is None:
        save_path = os.path.join(config.MODEL_SAVE_DIR, "random_forest.joblib")

    print("\n" + "=" * 60)
    print("  TRAINING RANDOM FOREST BASELINE")
    print("=" * 60)

    # Extract statistical features
    print("  Extracting statistical features...")
    X_train_feats = extract_statistical_features(X_train)
    X_test_feats = extract_statistical_features(X_test)
    print(f"  Feature vector size: {X_train_feats.shape[1]} per sample")

    # Train Random Forest
    print(f"  Training RF with {n_estimators} trees, max_depth={max_depth}...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    rf.fit(X_train_feats, y_train)

    # Predict
    y_pred = rf.predict(X_test_feats)
    y_proba = rf.predict_proba(X_test_feats)

    # Compute metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    print("\n  Random Forest Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")

    # Classification report
    report = classification_report(
        y_test, y_pred,
        target_names=["Normal", "Tampered"],
        zero_division=0,
    )
    print(f"\n{report}")

    # Feature importance (top 10)
    importances = rf.feature_importances_
    feature_names = []
    for feat in config.FEATURE_COLUMNS:
        for stat in ["mean", "std", "min", "max", "median", "range", "skew", "kurt"]:
            feature_names.append(f"{feat}_{stat}")

    top_indices = np.argsort(importances)[::-1][:10]
    print("  Top 10 Feature Importances:")
    for idx in top_indices:
        if idx < len(feature_names):
            print(f"    {feature_names[idx]:25s} {importances[idx]:.4f}")

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(rf, save_path)
    print(f"\n  Model saved: {save_path}")

    # Save metrics
    metrics_path = os.path.join(config.LOGS_DIR, "baseline_rf_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 60)

    return {
        "model": rf,
        "predictions": y_pred,
        "probabilities": y_proba,
        "metrics": metrics,
        "feature_importances": dict(zip(feature_names, importances.tolist())),
    }
