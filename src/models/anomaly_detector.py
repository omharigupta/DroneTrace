"""
Anomaly Detection Layer for UAV Forensic Analysis.

Provides two complementary anomaly scoring methods:
1. Dynamic Threshold: Uses statistical thresholding (μ + kσ) on model scores
2. One-Class SVM: Trains on Transformer embeddings for unsupervised detection
"""

import numpy as np
import logging
from typing import Tuple, Optional
from sklearn.svm import OneClassSVM
import joblib
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logger = logging.getLogger(__name__)


class DynamicThresholdDetector:
    """
    Anomaly detector using dynamic statistical thresholding.

    Computes: threshold = μ + k × σ
    on a calibration set of normal flight scores.

    Scores above the threshold → "Tampered"
    Scores below the threshold → "Normal"
    """

    def __init__(self, k: float = config.ANOMALY_THRESHOLD_K):
        """
        Args:
            k: Number of standard deviations above the mean for the threshold.
        """
        self.k = k
        self.mean: Optional[float] = None
        self.std: Optional[float] = None
        self.threshold: Optional[float] = None

    def fit(self, normal_scores: np.ndarray) -> "DynamicThresholdDetector":
        """
        Calibrate the threshold on normal (non-tampered) flight scores.

        Args:
            normal_scores: 1D array of anomaly scores from normal flights.

        Returns:
            self (for chaining).
        """
        self.mean = float(np.mean(normal_scores))
        self.std = float(np.std(normal_scores))
        self.threshold = self.mean + self.k * self.std

        logger.info(f"Dynamic Threshold calibrated:")
        logger.info(f"  Mean: {self.mean:.6f}")
        logger.info(f"  Std:  {self.std:.6f}")
        logger.info(f"  k:    {self.k}")
        logger.info(f"  Threshold: {self.threshold:.6f}")

        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """
        Classify scores as normal (0) or tampered (1).

        Args:
            scores: 1D array of anomaly scores.

        Returns:
            Binary predictions: 0 (Normal) or 1 (Tampered).
        """
        if self.threshold is None:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        return (scores > self.threshold).astype(np.int32)

    def score(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Compute normalized anomaly scores (how many σ above the mean).

        Args:
            raw_scores: 1D array of raw model output scores.

        Returns:
            Normalized deviation scores.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        return (raw_scores - self.mean) / (self.std + 1e-8)


class OneClassSVMDetector:
    """
    Anomaly detector using One-Class SVM on Transformer embeddings.

    Trained on embeddings from normal flights. At inference, embeddings
    that fall outside the learned decision boundary are flagged as tampered.

    Uses RBF kernel with configurable ν (expected outlier fraction).
    """

    def __init__(
        self,
        kernel: str = config.OCSVM_KERNEL,
        nu: float = config.OCSVM_NU,
        gamma: str = config.OCSVM_GAMMA,
    ):
        """
        Args:
            kernel: SVM kernel type ('rbf', 'linear', 'poly').
            nu: Upper bound on the fraction of outliers.
            gamma: Kernel coefficient ('scale', 'auto', or float).
        """
        self.svm = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        self.is_fitted = False

    def fit(self, normal_embeddings: np.ndarray) -> "OneClassSVMDetector":
        """
        Train the One-Class SVM on embeddings from normal flights.

        Args:
            normal_embeddings: 2D array of shape (n_samples, d_model).

        Returns:
            self (for chaining).
        """
        logger.info(f"Training One-Class SVM on {len(normal_embeddings)} normal embeddings...")
        self.svm.fit(normal_embeddings)
        self.is_fitted = True
        logger.info("One-Class SVM training complete.")
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Classify embeddings as normal (0) or tampered (1).

        SVM decision: +1 → Normal (inlier), -1 → Tampered (outlier)
        We convert: -1 → 1 (Tampered), +1 → 0 (Normal)

        Args:
            embeddings: 2D array of shape (n_samples, d_model).

        Returns:
            Binary predictions: 0 (Normal) or 1 (Tampered).
        """
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        svm_predictions = self.svm.predict(embeddings)
        # Convert: +1 (inlier) → 0 (Normal), -1 (outlier) → 1 (Tampered)
        return (svm_predictions == -1).astype(np.int32)

    def decision_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Get raw SVM decision function scores.

        More negative → more anomalous.

        Args:
            embeddings: 2D array of shape (n_samples, d_model).

        Returns:
            Decision function scores.
        """
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        return self.svm.decision_function(embeddings)

    def save(self, filepath: str):
        """Save the trained SVM model to disk."""
        joblib.dump(self.svm, filepath)
        logger.info(f"One-Class SVM saved to {filepath}")

    def load(self, filepath: str):
        """Load a trained SVM model from disk."""
        self.svm = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"One-Class SVM loaded from {filepath}")


class EnsembleAnomalyDetector:
    """
    Ensemble anomaly detector combining Dynamic Threshold and One-Class SVM.

    Flags a sample as tampered if EITHER detector raises an alarm.
    This maximizes recall (attack detection rate) at the cost of some precision.
    """

    def __init__(
        self,
        threshold_k: float = config.ANOMALY_THRESHOLD_K,
        svm_nu: float = config.OCSVM_NU,
    ):
        self.dynamic = DynamicThresholdDetector(k=threshold_k)
        self.ocsvm = OneClassSVMDetector(nu=svm_nu)

    def fit(
        self,
        normal_scores: np.ndarray,
        normal_embeddings: np.ndarray,
    ) -> "EnsembleAnomalyDetector":
        """
        Fit both detectors on normal flight data.

        Args:
            normal_scores: Model output scores for normal flights.
            normal_embeddings: Transformer embeddings for normal flights.

        Returns:
            self.
        """
        self.dynamic.fit(normal_scores)
        self.ocsvm.fit(normal_embeddings)
        return self

    def predict(
        self,
        scores: np.ndarray,
        embeddings: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict using both detectors and combine with OR logic.

        Args:
            scores: Model output scores.
            embeddings: Transformer embeddings.

        Returns:
            Tuple of (ensemble_pred, dynamic_pred, ocsvm_pred).
        """
        dynamic_pred = self.dynamic.predict(scores)
        ocsvm_pred = self.ocsvm.predict(embeddings)
        ensemble_pred = np.maximum(dynamic_pred, ocsvm_pred)  # OR logic
        return ensemble_pred, dynamic_pred, ocsvm_pred
