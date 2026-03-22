"""
Positional Encoding Layer for the Transformer model.

Since Transformers have no inherent notion of sequential order,
we inject positional information using sinusoidal encoding
(Vaswani et al., "Attention Is All You Need", 2017).

PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
"""

import numpy as np
import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Sinusoidal Positional Encoding layer.

    Adds position-dependent signals to the input embeddings so that the
    Transformer can distinguish between timestep 1 and timestep 100.

    Args:
        max_seq_len: Maximum sequence length to pre-compute encodings for.
        d_model: Dimensionality of the embeddings.
    """

    def __init__(self, max_seq_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # Pre-compute the positional encoding matrix
        self.pe = self._build_encoding(max_seq_len, d_model)

    @staticmethod
    def _build_encoding(max_seq_len: int, d_model: int) -> tf.Tensor:
        """
        Build the sinusoidal positional encoding matrix.

        Returns:
            Tensor of shape (1, max_seq_len, d_model).
        """
        positions = np.arange(max_seq_len)[:, np.newaxis]  # (seq_len, 1)
        dims = np.arange(d_model)[np.newaxis, :]           # (1, d_model)

        # Compute the angle rates
        angle_rates = 1 / np.power(10000.0, (2 * (dims // 2)) / d_model)
        angle_rads = positions * angle_rates  # (seq_len, d_model)

        # Apply sin to even indices, cos to odd indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        # Add batch dimension: (1, seq_len, d_model)
        pe = angle_rads[np.newaxis, :, :]
        return tf.cast(pe, dtype=tf.float32)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor of same shape with positional encoding added.
        """
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]

    def get_config(self):
        base = super().get_config()
        base.update({
            "max_seq_len": self.max_seq_len,
            "d_model": self.d_model,
        })
        return base
