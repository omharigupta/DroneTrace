"""
Evaluation Metrics for UAV Forensic Anomaly Detection.

Computes:
- Accuracy, Precision, Recall, F1-Score
- Mean Average Precision (mAP)
- Confusion Matrix with visualization
- ROC Curve
- Per-class metrics
"""

import os
import json
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    average_precision_score,
    precision_recall_curve,
)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logger = logging.getLogger(__name__)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    model_name: str = "Transformer",
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics for binary classification.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_scores: Predicted probabilities/scores (for mAP, ROC).
        model_name: Name of the model for logging.

    Returns:
        Dictionary with all computed metrics.
    """
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # Extract TP, TN, FP, FN
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["true_positives"] = int(tp)
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)

    # mAP and ROC AUC (if probability scores available)
    if y_scores is not None:
        metrics["mAP"] = float(average_precision_score(y_true, y_scores))
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        metrics["roc_auc"] = float(auc(fpr, tpr))
    else:
        metrics["mAP"] = None
        metrics["roc_auc"] = None

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"  {model_name} — Evaluation Metrics")
    print(f"{'=' * 50}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    if metrics["mAP"] is not None:
        print(f"  mAP:       {metrics['mAP']:.4f}")
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"{'=' * 50}")

    # Detailed report
    report = classification_report(
        y_true, y_pred,
        target_names=["Normal", "Tampered"],
        zero_division=0,
    )
    print(report)
    metrics["classification_report"] = report

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Transformer",
    save_path: Optional[str] = None,
):
    """
    Plot and save a confusion matrix heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        model_name: Model name for the title.
        save_path: Path to save the figure. If None, saves to logs/.
    """
    if save_path is None:
        save_path = os.path.join(config.LOGS_DIR, f"confusion_matrix_{model_name.lower()}.png")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Tampered"],
        yticklabels=["Normal", "Tampered"],
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved: {save_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    model_name: str = "Transformer",
    save_path: Optional[str] = None,
):
    """
    Plot and save the ROC curve.

    Args:
        y_true: Ground truth labels.
        y_scores: Predicted probabilities.
        model_name: Model name for the title.
        save_path: Path to save the figure.
    """
    if save_path is None:
        save_path = os.path.join(config.LOGS_DIR, f"roc_curve_{model_name.lower()}.png")

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random Baseline")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ROC curve saved: {save_path}")


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    model_name: str = "Transformer",
    save_path: Optional[str] = None,
):
    """
    Plot and save the Precision-Recall curve.

    Args:
        y_true: Ground truth labels.
        y_scores: Predicted probabilities.
        model_name: Model name for the title.
        save_path: Path to save the figure.
    """
    if save_path is None:
        save_path = os.path.join(config.LOGS_DIR, f"pr_curve_{model_name.lower()}.png")

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="green", lw=2, label=f"PR curve (mAP = {ap:.4f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=14)
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  PR curve saved: {save_path}")


def plot_training_history(
    history: dict,
    save_path: Optional[str] = None,
):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: Training history dictionary from model.fit().
        save_path: Path to save the figure.
    """
    if save_path is None:
        save_path = os.path.join(config.LOGS_DIR, "training_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(history["loss"], label="Train Loss", color="blue")
    axes[0].plot(history["val_loss"], label="Val Loss", color="red")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy curve
    axes[1].plot(history["accuracy"], label="Train Accuracy", color="blue")
    axes[1].plot(history["val_accuracy"], label="Val Accuracy", color="red")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Training curves saved: {save_path}")


def compare_models(
    results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
):
    """
    Create a comparison bar chart of different models' metrics.

    Args:
        results: Dict mapping model name → metrics dict.
        save_path: Path to save the figure.
    """
    if save_path is None:
        save_path = os.path.join(config.LOGS_DIR, "model_comparison.png")

    model_names = list(results.keys())
    metric_names = ["accuracy", "precision", "recall", "f1_score"]

    x = np.arange(len(metric_names))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model_name in enumerate(model_names):
        values = [results[model_name].get(m, 0) for m in metric_names]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Forensic Anomaly Detection", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-Score"])
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Model comparison saved: {save_path}")


def save_metrics(metrics: Dict[str, Any], filename: str):
    """
    Save metrics dictionary to a JSON file in the logs directory.

    Args:
        metrics: Metrics dictionary.
        filename: Filename (will be saved in LOGS_DIR).
    """
    filepath = os.path.join(config.LOGS_DIR, filename)
    # Convert numpy types to Python types for JSON serialization
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        elif isinstance(v, (np.float32, np.float64)):
            clean[k] = float(v)
        elif isinstance(v, (np.int32, np.int64)):
            clean[k] = int(v)
        else:
            clean[k] = v

    with open(filepath, "w") as f:
        json.dump(clean, f, indent=2, default=str)
    print(f"  Metrics saved: {filepath}")
