"""
LoRA (Low-Rank Adaptation) Layer for Transformer-Based Forensic Model.

LoRA injects small trainable rank-decomposition matrices into existing
Dense layers while keeping the original weights frozen. This:
  1. Dramatically reduces trainable parameters (e.g., 90%+ reduction)
  2. Prevents overfitting on small forensic datasets
  3. Enables efficient fine-tuning from a pre-trained checkpoint
  4. Trains faster with less memory

Theory (Hu et al., 2021):
    For a pre-trained weight matrix W ∈ R^(d×k):
        W' = W + ΔW = W + B·A
    Where:
        A ∈ R^(r×k)  — down-projection (rank r << min(d,k))
        B ∈ R^(d×r)  — up-projection
        r = LoRA rank (default: 8)

    Only A and B are trainable. W is frozen.
    At inference, ΔW = B·A can be merged into W for zero latency cost.

    Scaled output: h = Wx + (α/r) · BAx
    Where α is a scaling factor (default: α = r).
"""

import tensorflow as tf
import numpy as np
from typing import Optional


class LoRADense(tf.keras.layers.Layer):
    """
    LoRA-adapted Dense layer.

    Wraps an existing Dense layer by adding low-rank matrices A and B.
    The original Dense weights are frozen; only A and B are trainable.

    Output = Dense(x) + (alpha/rank) * x @ A^T @ B^T

    Args:
        original_dense: The original Dense layer to adapt.
        rank: LoRA rank (r). Lower = fewer params, higher = more expressive.
        alpha: Scaling factor. Default = rank (so scaling = 1.0).
        dropout_rate: Optional dropout on LoRA path for regularization.
    """

    def __init__(
        self,
        units: int,
        rank: int = 8,
        alpha: float = None,
        dropout_rate: float = 0.0,
        activation: Optional[str] = None,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.scaling = self.alpha / self.rank
        self.dropout_rate = dropout_rate
        self.activation_name = activation
        self.use_bias = use_bias

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        # Original (frozen) Dense weights
        self.original_kernel = self.add_weight(
            name="original_kernel",
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            trainable=False,  # FROZEN
        )
        if self.use_bias:
            self.original_bias = self.add_weight(
                name="original_bias",
                shape=(self.units,),
                initializer="zeros",
                trainable=False,  # FROZEN
            )
        else:
            self.original_bias = None

        # LoRA matrices (TRAINABLE)
        # A: down-project from input_dim to rank (initialized with Kaiming/He)
        self.lora_A = self.add_weight(
            name="lora_A",
            shape=(input_dim, self.rank),
            initializer=tf.keras.initializers.HeNormal(),
            trainable=True,
        )
        # B: up-project from rank to units (initialized to zero for stable start)
        self.lora_B = self.add_weight(
            name="lora_B",
            shape=(self.rank, self.units),
            initializer="zeros",
            trainable=True,
        )

        # Optional dropout on LoRA path
        if self.dropout_rate > 0:
            self.lora_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        else:
            self.lora_dropout = None

        # Activation
        self.activation_fn = tf.keras.activations.get(self.activation_name)

        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass: original Dense + scaled LoRA delta.

        h = Wx + b + (α/r) · x·A·B
        """
        # Original path (frozen)
        original_output = tf.matmul(x, self.original_kernel)
        if self.original_bias is not None:
            original_output = original_output + self.original_bias

        # LoRA path (trainable)
        lora_input = x
        if self.lora_dropout is not None:
            lora_input = self.lora_dropout(lora_input, training=training)

        lora_output = tf.matmul(tf.matmul(lora_input, self.lora_A), self.lora_B)
        lora_output = lora_output * self.scaling

        # Combined
        output = original_output + lora_output

        if self.activation_fn is not None:
            output = self.activation_fn(output)

        return output

    def merge_lora_weights(self):
        """
        Merge LoRA matrices into the original weights for inference.

        After merging: W' = W + (α/r)·A·B
        This eliminates the LoRA overhead at inference time (zero extra latency).
        """
        delta = tf.matmul(self.lora_A, self.lora_B) * self.scaling
        self.original_kernel.assign(self.original_kernel + delta)
        # Reset LoRA to zero after merge
        self.lora_A.assign(tf.zeros_like(self.lora_A))
        self.lora_B.assign(tf.zeros_like(self.lora_B))

    def get_config(self):
        base = super().get_config()
        base.update({
            "units": self.units,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation_name,
            "use_bias": self.use_bias,
        })
        return base


class LoRAMultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-Head Attention with LoRA on Q, K, V, and Output projections.

    Instead of training the full attention weight matrices, we inject
    low-rank adaptations that capture task-specific patterns with far
    fewer parameters.

    Args:
        num_heads: Number of attention heads.
        d_model: Total model dimension.
        rank: LoRA rank for each projection.
        alpha: LoRA scaling factor.
        dropout: Attention dropout rate.
    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        rank: int = 8,
        alpha: float = None,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.key_dim = d_model // num_heads
        self.rank = rank
        self.alpha = alpha
        self.attn_dropout = dropout

        # LoRA-adapted Q, K, V projections
        self.query_dense = LoRADense(d_model, rank=rank, alpha=alpha, name="lora_query")
        self.key_dense = LoRADense(d_model, rank=rank, alpha=alpha, name="lora_key")
        self.value_dense = LoRADense(d_model, rank=rank, alpha=alpha, name="lora_value")
        self.output_dense = LoRADense(d_model, rank=rank, alpha=alpha, name="lora_output")

        self.dropout_layer = tf.keras.layers.Dropout(dropout)

    def call(self, query, key, value, training=False):
        batch_size = tf.shape(query)[0]
        seq_len = tf.shape(query)[1]

        # Project Q, K, V through LoRA-adapted Dense
        Q = self.query_dense(query, training=training)
        K = self.key_dense(key, training=training)
        V = self.value_dense(value, training=training)

        # Reshape for multi-head: (batch, seq, d_model) → (batch, heads, seq, key_dim)
        Q = tf.reshape(Q, (batch_size, -1, self.num_heads, self.key_dim))
        Q = tf.transpose(Q, perm=[0, 2, 1, 3])
        K = tf.reshape(K, (batch_size, -1, self.num_heads, self.key_dim))
        K = tf.transpose(K, perm=[0, 2, 1, 3])
        V = tf.reshape(V, (batch_size, -1, self.num_heads, self.key_dim))
        V = tf.transpose(V, perm=[0, 2, 1, 3])

        # Scaled dot-product attention
        # Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
        scale = tf.math.sqrt(tf.cast(self.key_dim, tf.float32))
        attention_scores = tf.matmul(Q, K, transpose_b=True) / scale
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout_layer(attention_weights, training=training)

        # Apply attention to values
        attention_output = tf.matmul(attention_weights, V)

        # Reshape back: (batch, heads, seq, key_dim) → (batch, seq, d_model)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, -1, self.d_model))

        # Output projection (also LoRA-adapted)
        output = self.output_dense(attention_output, training=training)
        return output

    def merge_all_lora(self):
        """Merge LoRA weights in all projections for inference."""
        self.query_dense.merge_lora_weights()
        self.key_dense.merge_lora_weights()
        self.value_dense.merge_lora_weights()
        self.output_dense.merge_lora_weights()

    def get_config(self):
        base = super().get_config()
        base.update({
            "num_heads": self.num_heads,
            "d_model": self.d_model,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.attn_dropout,
        })
        return base


def count_parameters(model: tf.keras.Model) -> dict:
    """
    Count total vs trainable parameters in a model.

    Returns:
        Dict with total, trainable, frozen counts and LoRA efficiency ratio.
    """
    total = int(np.sum([tf.size(w).numpy() for w in model.weights]))
    trainable = int(np.sum([tf.size(w).numpy() for w in model.trainable_weights]))
    frozen = total - trainable

    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": frozen,
        "trainable_ratio": trainable / max(total, 1),
        "reduction": 1.0 - (trainable / max(total, 1)),
    }
