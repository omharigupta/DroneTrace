"""
Transformer Encoder Model for UAV Forensic Anomaly Detection.

Architecture:
    Input (batch, seq_len, 5 features)
    → Dense Embedding (batch, seq_len, d_model)
    → Positional Encoding
    → N × Transformer Encoder Blocks (MHA + FFN + LayerNorm + Residual)
    → Global Average Pooling
    → Dense → Dropout → Dense(1, Sigmoid)
    → Anomaly probability ∈ [0, 1]

LoRA Support:
    When config.USE_LORA=True, attention Q/K/V/O projections and FFN layers
    use Low-Rank Adaptation. Original weights are frozen; only small LoRA
    matrices (rank=8 by default) are trained. This:
    - Reduces trainable params by ~70-90%
    - Prevents overfitting on small forensic datasets
    - Enables efficient fine-tuning from checkpoints
"""

import tensorflow as tf
import numpy as np
from src.models.positional_encoding import PositionalEncoding
from src.models.lora import LoRADense, LoRAMultiHeadAttention, count_parameters

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


class TransformerEncoderBlock(tf.keras.layers.Layer):
    """
    Single Transformer Encoder block with optional LoRA adaptation.

    Contains:
        - Multi-Head Self-Attention (standard or LoRA-adapted)
        - Residual Connection + LayerNorm
        - Position-wise Feed-Forward Network (standard or LoRA-adapted)
        - Residual Connection + LayerNorm
        - Dropout

    Args:
        d_model: Embedding dimension.
        num_heads: Number of attention heads.
        d_ff: Feed-forward inner dimension.
        dropout_rate: Dropout rate.
        use_lora: Whether to use LoRA adaptation.
        lora_rank: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: LoRA-specific dropout.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        if use_lora:
            # LoRA-adapted Multi-Head Attention (Q, K, V, O all have LoRA)
            self.mha = LoRAMultiHeadAttention(
                num_heads=num_heads,
                d_model=d_model,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=dropout_rate,
            )
            # LoRA-adapted Feed-Forward Network
            self.ffn_1 = LoRADense(
                d_ff, rank=lora_rank, alpha=lora_alpha,
                activation="relu", dropout_rate=lora_dropout,
            )
            self.ffn_2 = LoRADense(
                d_model, rank=lora_rank, alpha=lora_alpha,
                dropout_rate=lora_dropout,
            )
        else:
            # Standard Multi-Head Self-Attention
            self.mha = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=d_model // num_heads,
                dropout=dropout_rate,
            )
            # Standard Feed-Forward Network
            self.ffn_1 = tf.keras.layers.Dense(d_ff, activation="relu")
            self.ffn_2 = tf.keras.layers.Dense(d_model)

        # Layer Normalization (always trainable — tiny param count)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass through the encoder block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            training: Whether in training mode (enables dropout).

        Returns:
            Output tensor of same shape.
        """
        # Multi-Head Self-Attention + Residual + LayerNorm
        if self.use_lora:
            attn_output = self.mha(query=x, key=x, value=x, training=training)
        else:
            attn_output = self.mha(query=x, key=x, value=x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.layernorm1(x + attn_output)

        # Feed-Forward Network + Residual + LayerNorm
        ffn_output = self.ffn_1(x, training=training) if self.use_lora else self.ffn_1(x)
        ffn_output = self.ffn_2(ffn_output, training=training) if self.use_lora else self.ffn_2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = self.layernorm2(x + ffn_output)

        return x

    def merge_lora(self):
        """Merge LoRA weights into original weights for zero-cost inference."""
        if self.use_lora:
            if hasattr(self.mha, 'merge_all_lora'):
                self.mha.merge_all_lora()
            if hasattr(self.ffn_1, 'merge_lora_weights'):
                self.ffn_1.merge_lora_weights()
            if hasattr(self.ffn_2, 'merge_lora_weights'):
                self.ffn_2.merge_lora_weights()

    def get_config(self):
        base = super().get_config()
        base.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "dropout_rate": self.dropout_rate,
            "use_lora": self.use_lora,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
        })
        return base


class TransformerForensicModel(tf.keras.Model):
    """
    Full Transformer Encoder model for binary anomaly classification.

    Architecture:
        Input → Embedding → PE → Encoder×N → GAP → Dense → Sigmoid

    Args:
        num_features: Number of input features (default: 5).
        seq_len: Sequence length (default: 100).
        d_model: Embedding dimension (default: 64).
        num_heads: Number of attention heads (default: 4).
        num_layers: Number of encoder blocks (default: 4).
        d_ff: Feed-forward dimension (default: 128).
        dropout_rate: Dropout in encoder blocks (default: 0.1).
        classifier_dropout: Dropout before final layer (default: 0.3).
    """

    def __init__(
        self,
        num_features: int = config.NUM_FEATURES,
        seq_len: int = config.MAX_SEQ_LEN,
        d_model: int = config.D_MODEL,
        num_heads: int = config.NUM_HEADS,
        num_layers: int = config.NUM_ENCODER_LAYERS,
        d_ff: int = config.D_FF,
        dropout_rate: float = config.DROPOUT_RATE,
        classifier_dropout: float = config.CLASSIFIER_DROPOUT,
        use_lora: bool = config.USE_LORA,
        lora_rank: int = config.LORA_RANK,
        lora_alpha: float = config.LORA_ALPHA,
        lora_dropout: float = config.LORA_DROPOUT,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_features = num_features
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_lora = use_lora
        self.lora_rank = lora_rank

        # Feature embedding: project raw features to d_model dimensions
        self.feature_embedding = tf.keras.layers.Dense(d_model, activation=None)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            max_seq_len=seq_len,
            d_model=d_model,
        )

        # Embedding dropout
        self.embedding_dropout = tf.keras.layers.Dropout(dropout_rate)

        # Stack of Transformer Encoder blocks (with optional LoRA)
        self.encoder_blocks = [
            TransformerEncoderBlock(
                d_model, num_heads, d_ff, dropout_rate,
                use_lora=use_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            for _ in range(num_layers)
        ]

        # Classification head (always fully trainable)
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.classifier_dense = tf.keras.layers.Dense(d_model, activation="relu")
        self.classifier_dropout_layer = tf.keras.layers.Dropout(classifier_dropout)
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass through the full model.

        Args:
            x: Input tensor of shape (batch, seq_len, num_features).
            training: Whether in training mode.

        Returns:
            Anomaly probability of shape (batch, 1).
        """
        # Feature embedding: (batch, seq_len, num_features) → (batch, seq_len, d_model)
        x = self.feature_embedding(x)

        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.embedding_dropout(x, training=training)

        # Pass through encoder stack
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, training=training)

        # Classification: (batch, seq_len, d_model) → (batch, 1)
        x = self.global_avg_pool(x)
        x = self.classifier_dense(x)
        x = self.classifier_dropout_layer(x, training=training)
        x = self.output_layer(x)

        return x

    def get_embeddings(self, x: tf.Tensor) -> tf.Tensor:
        """
        Extract penultimate-layer embeddings for anomaly detection.

        Args:
            x: Input tensor of shape (batch, seq_len, num_features).

        Returns:
            Embedding tensor of shape (batch, d_model).
        """
        x = self.feature_embedding(x)
        x = self.positional_encoding(x)

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, training=False)

        x = self.global_avg_pool(x)
        x = self.classifier_dense(x)
        return x

    def merge_lora_weights(self):
        """
        Merge all LoRA weights into original weights for inference.

        After merging:
        - W' = W + (α/r) · A · B for every adapted layer
        - LoRA matrices reset to zero
        - Zero additional inference cost
        """
        if not self.use_lora:
            print("  LoRA not enabled — nothing to merge.")
            return

        print("  Merging LoRA weights into base model...")
        for block in self.encoder_blocks:
            block.merge_lora()
        print("  LoRA merge complete. Model now runs at full speed with no LoRA overhead.")

    def summary_custom(self):
        """Print a custom model summary with LoRA details."""
        print("=" * 60)
        print("  Transformer Forensic Model Summary")
        print("=" * 60)
        print(f"  Input:          ({self.seq_len}, {self.num_features})")
        print(f"  Embedding dim:  {self.d_model}")
        print(f"  Attention heads: {self.num_heads}")
        print(f"  Encoder layers: {self.num_layers}")
        if self.use_lora:
            print(f"  LoRA:           ENABLED (rank={self.lora_rank})")
        else:
            print(f"  LoRA:           disabled (full fine-tuning)")
        print(f"  Output:         Sigmoid (binary)")

        # Count parameters
        param_info = count_parameters(self)
        print(f"  Total params:    {param_info['total_params']:,}")
        print(f"  Trainable:       {param_info['trainable_params']:,}")
        print(f"  Frozen:          {param_info['frozen_params']:,}")
        if self.use_lora:
            print(f"  Param reduction: {param_info['reduction']:.1%} (via LoRA)")
        print("=" * 60)

    def get_config(self):
        return {
            "num_features": self.num_features,
            "seq_len": self.seq_len,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "use_lora": self.use_lora,
            "lora_rank": self.lora_rank,
        }


def build_transformer_model(
    num_features: int = config.NUM_FEATURES,
    seq_len: int = config.MAX_SEQ_LEN,
    d_model: int = config.D_MODEL,
    num_heads: int = config.NUM_HEADS,
    num_layers: int = config.NUM_ENCODER_LAYERS,
    d_ff: int = config.D_FF,
    learning_rate: float = config.LEARNING_RATE,
    use_lora: bool = config.USE_LORA,
    lora_rank: int = config.LORA_RANK,
    lora_alpha: float = config.LORA_ALPHA,
) -> TransformerForensicModel:
    """
    Build and compile the Transformer forensic model.

    When use_lora=True:
        - Attention Q/K/V/O projections use LoRA (frozen base + trainable low-rank)
        - FFN layers use LoRA
        - Embedding and classifier head remain fully trainable
        - ~70-90% parameter reduction

    Args:
        num_features: Number of input features.
        seq_len: Sequence length.
        d_model: Embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of encoder blocks.
        d_ff: Feed-forward dimension.
        learning_rate: Adam learning rate.
        use_lora: Whether to enable LoRA.
        lora_rank: LoRA rank.
        lora_alpha: LoRA scaling factor.

    Returns:
        Compiled TransformerForensicModel.
    """
    model = TransformerForensicModel(
        num_features=num_features,
        seq_len=seq_len,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )

    # Build the model by passing a dummy input
    dummy_input = tf.zeros((1, seq_len, num_features))
    _ = model(dummy_input)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    model.summary_custom()
    return model
