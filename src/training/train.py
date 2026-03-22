"""
Training Pipeline for the Transformer Forensic Model.

Includes:
- Cosine decay learning rate schedule with warmup
- Early stopping
- Model checkpointing
- Training history logging
"""

import os
import json
import logging
import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Any

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.models.transformer_model import build_transformer_model

logger = logging.getLogger(__name__)


def find_latest_checkpoint(checkpoint_dir: str = None) -> Optional[str]:
    """
    Find the most recent checkpoint file in the model save directory.

    Looks for:
      1. transformer_best.keras  (best val_loss model)
      2. checkpoint_epoch_*.keras (periodic per-epoch snapshots)

    Returns the path to the latest checkpoint, or None if none found.
    """
    if checkpoint_dir is None:
        checkpoint_dir = config.MODEL_SAVE_DIR

    if not os.path.isdir(checkpoint_dir):
        return None

    # Priority 1: Best model
    best_path = os.path.join(checkpoint_dir, "transformer_best.keras")
    if os.path.exists(best_path):
        logger.info(f"Found best checkpoint: {best_path}")
        return best_path

    # Priority 2: Latest periodic checkpoint (sort by epoch number)
    import glob
    pattern = os.path.join(checkpoint_dir, "checkpoint_epoch_*.keras")
    checkpoints = sorted(glob.glob(pattern))
    if checkpoints:
        latest = checkpoints[-1]
        logger.info(f"Found periodic checkpoint: {latest}")
        return latest

    return None


def get_initial_epoch_from_checkpoint(checkpoint_path: str) -> int:
    """
    Extract the epoch number from a checkpoint filename.

    Expected formats:
      - checkpoint_epoch_005_vloss_0.1234.keras → returns 5
      - transformer_best.keras → returns 0 (unknown epoch, start fresh)

    Returns:
        Epoch number (0-based for Keras fit's initial_epoch).
    """
    basename = os.path.basename(checkpoint_path)
    if basename.startswith("checkpoint_epoch_"):
        try:
            # Extract epoch number: checkpoint_epoch_005_vloss_0.1234.keras
            parts = basename.replace(".keras", "").split("_")
            epoch_idx = parts.index("epoch") + 1
            return int(parts[epoch_idx])  # This is 1-based from ModelCheckpoint
        except (ValueError, IndexError):
            pass
    return 0


class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Cosine decay learning rate schedule with linear warmup.

    During warmup: LR increases linearly from 0 to max_lr.
    After warmup: LR decays following a cosine curve to 0.

    Args:
        max_lr: Peak learning rate after warmup.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.
    """

    def __init__(self, max_lr: float, warmup_steps: int, total_steps: int):
        super().__init__()
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        # Linear warmup
        warmup_lr = self.max_lr * (step / max(self.warmup_steps, 1))

        # Cosine decay
        decay_steps = max(self.total_steps - self.warmup_steps, 1)
        progress = (step - self.warmup_steps) / decay_steps
        cosine_lr = self.max_lr * 0.5 * (1.0 + tf.cos(np.pi * progress))

        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "max_lr": self.max_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }


def train_transformer(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    num_features: int = config.NUM_FEATURES,
    seq_len: int = config.MAX_SEQ_LEN,
    epochs: int = config.EPOCHS,
    learning_rate: float = config.LEARNING_RATE,
    warmup_steps: int = config.WARMUP_STEPS,
    model_save_path: Optional[str] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Full training pipeline for the Transformer forensic model.

    Supports automatic checkpoint resumption:
      - If resume=True and a checkpoint exists, loads it and continues
        training from the last saved epoch.
      - If no checkpoint exists, trains from scratch.
      - Saves best model + periodic epoch checkpoints for crash recovery.

    Args:
        train_ds: Training tf.data.Dataset.
        val_ds: Validation tf.data.Dataset.
        num_features: Number of input features.
        seq_len: Sequence length.
        epochs: Maximum training epochs.
        learning_rate: Peak learning rate.
        warmup_steps: LR warmup steps.
        model_save_path: Path to save the best model.
        resume: If True, automatically resume from the latest checkpoint.

    Returns:
        Dictionary with model, history, and best metrics.
    """
    if model_save_path is None:
        model_save_path = os.path.join(config.MODEL_SAVE_DIR, "transformer_best.keras")

    print("\n" + "=" * 60)
    print("  TRAINING TRANSFORMER FORENSIC MODEL")
    print("=" * 60)

    # Set random seed
    tf.random.set_seed(config.RANDOM_SEED)

    # Estimate total training steps for LR schedule
    train_size = sum(1 for _ in train_ds)
    total_steps = train_size * epochs

    # Build LR schedule
    lr_schedule = CosineDecayWithWarmup(
        max_lr=learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    # ── CHECK FOR EXISTING CHECKPOINT (resume training) ───────────
    initial_epoch = 0
    checkpoint_path = None

    if resume:
        checkpoint_path = find_latest_checkpoint()

    if checkpoint_path is not None:
        print(f"\n  RESUMING from checkpoint: {os.path.basename(checkpoint_path)}")
        initial_epoch = get_initial_epoch_from_checkpoint(checkpoint_path)
        print(f"  Starting at epoch {initial_epoch + 1} / {epochs}")

        # Load the saved model
        model = tf.keras.models.load_model(checkpoint_path)
        print(f"  Model loaded successfully.")

        # If already completed all epochs, skip training
        if initial_epoch >= epochs:
            print(f"  Model already trained for {initial_epoch} epochs (max={epochs}). Skipping.")
            return {
                "model": model,
                "history": {},
                "best_metrics": {"best_epoch": initial_epoch, "val_loss": 0.0,
                                 "val_accuracy": 0.0, "train_loss": 0.0, "train_accuracy": 0.0},
                "model_path": checkpoint_path,
            }
    else:
        print("\n  No checkpoint found — training from scratch.")
        # Build fresh model
        model = build_transformer_model(
            num_features=num_features,
            seq_len=seq_len,
            learning_rate=learning_rate,
        )

    # Re-compile with LR schedule (needed for both fresh and resumed models)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    # Callbacks
    callbacks = [
        # Early stopping on validation loss
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        # Save best model (overwrites only when val_loss improves)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        # Periodic checkpoint every 5 epochs (for crash recovery / resuming)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                config.MODEL_SAVE_DIR,
                "checkpoint_epoch_{epoch:03d}_vloss_{val_loss:.4f}.keras"
            ),
            monitor="val_loss",
            save_best_only=False,
            save_freq="epoch",
            verbose=0,
        ),
        # Reduce LR on plateau (supplementary)
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        # CSV logger — saves per-epoch metrics to a file for later analysis
        # append=True so resumed training adds to existing log
        tf.keras.callbacks.CSVLogger(
            os.path.join(config.LOGS_DIR, "training_log.csv"),
            append=(initial_epoch > 0),
        ),
    ]

    # Train
    remaining = epochs - initial_epoch
    if initial_epoch > 0:
        print(f"\nResuming training from epoch {initial_epoch + 1} to {epochs} ({remaining} epochs remaining)...")
    else:
        print(f"\nStarting training for up to {epochs} epochs...")
    print(f"  Learning rate: {learning_rate} (with cosine warmup)")
    print(f"  Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    print(f"  Model save path: {model_save_path}")
    if initial_epoch > 0:
        print(f"  Resuming from epoch: {initial_epoch}")
    print()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=1,
    )

    # Extract best metrics
    best_epoch = np.argmin(history.history["val_loss"])
    best_metrics = {
        "best_epoch": int(best_epoch) + 1,
        "train_loss": float(history.history["loss"][best_epoch]),
        "val_loss": float(history.history["val_loss"][best_epoch]),
        "train_accuracy": float(history.history["accuracy"][best_epoch]),
        "val_accuracy": float(history.history["val_accuracy"][best_epoch]),
    }

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best epoch: {best_metrics['best_epoch']}")
    print(f"  Val Loss:   {best_metrics['val_loss']:.4f}")
    print(f"  Val Acc:    {best_metrics['val_accuracy']:.4f}")
    print("=" * 60)

    # Save training history
    history_path = os.path.join(config.LOGS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        h = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        json.dump(h, f, indent=2)
    print(f"  History saved: {history_path}")

    return {
        "model": model,
        "history": history.history,
        "best_metrics": best_metrics,
        "model_path": model_save_path,
    }
