"""
TensorFlow Dataset Builder for UAV Forensic Anomaly Detection.

Creates tf.data.Dataset pipelines for efficient training and evaluation.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    val_split: float = config.VALIDATION_SPLIT,
    test_split: float = config.TEST_SPLIT,
    seed: int = config.RANDOM_SEED,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Split data into train, validation, and test sets.

    Args:
        X: Feature array of shape (num_samples, seq_len, num_features).
        y: Label array of shape (num_samples,).
        val_split: Fraction for validation.
        test_split: Fraction for testing.
        seed: Random seed for reproducibility.

    Returns:
        ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    n = len(X)
    n_test = int(n * test_split)
    n_val = int(n * val_split)

    X_test, y_test = X[:n_test], y[:n_test]
    X_val, y_val = X[n_test : n_test + n_val], y[n_test : n_test + n_val]
    X_train, y_train = X[n_test + n_val :], y[n_test + n_val :]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_tf_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = True,
    seed: int = config.RANDOM_SEED,
) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset from numpy arrays.

    Args:
        X: Feature array of shape (num_samples, seq_len, num_features).
        y: Label array of shape (num_samples,).
        batch_size: Batch size.
        shuffle: Whether to shuffle the data.
        seed: Random seed.

    Returns:
        Batched and prefetched tf.data.Dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((
        tf.cast(X, tf.float32),
        tf.cast(y, tf.float32),
    ))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X), seed=seed)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_datasets(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = config.BATCH_SIZE,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Full pipeline: split data and create TF datasets for train/val/test.

    Args:
        X: Feature array of shape (num_samples, seq_len, num_features).
        y: Label array of shape (num_samples,).
        batch_size: Batch size.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)

    train_ds = create_tf_dataset(X_train, y_train, batch_size, shuffle=True)
    val_ds = create_tf_dataset(X_val, y_val, batch_size, shuffle=False)
    test_ds = create_tf_dataset(X_test, y_test, batch_size, shuffle=False)

    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")

    return train_ds, val_ds, test_ds
