from __future__ import annotations

from typing import Sequence

import tensorflow as tf
from keras import Input, Model
from keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Embedding,
    LayerNormalization,
    MultiHeadAttention,
)

from .base import Encoder, LastTimeStep, normalize_units, sanitize_column_names


class TransformerBlock(tf.keras.layers.Layer):
    """Single Transformer decoder-style block with causal self-attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_multiplier: float,
        dropout: float,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.ff_multiplier = float(max(ff_multiplier, 1.0))
        self.dropout_rate = float(max(min(dropout, 1.0), 0.0))

        self.norm1 = LayerNormalization(epsilon=1e-6, name=f"{self.name}_pre_attn_norm")
        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate,
            name=f"{self.name}_mha",
        )
        self.drop1 = Dropout(self.dropout_rate, name=f"{self.name}_attn_dropout")
        self.norm2 = LayerNormalization(epsilon=1e-6, name=f"{self.name}_pre_ffn_norm")
        self.ffn_dense1 = tf.keras.layers.Dense(
            int(self.d_model * self.ff_multiplier),
            activation=tf.nn.gelu,
            name=f"{self.name}_ffn_dense_1",
        )
        self.ffn_dense2 = tf.keras.layers.Dense(self.d_model, name=f"{self.name}_ffn_dense_2")
        self.drop2 = Dropout(self.dropout_rate, name=f"{self.name}_ffn_dropout")

    def call(
        self,
        inputs: tf.Tensor,
        training: bool | None = None,
        mask: tf.Tensor | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor | None]:
        attn_mask = None
        if mask is not None:
            seq_len = tf.shape(inputs)[-2]
            padding = tf.cast(mask[:, tf.newaxis, :], dtype=inputs.dtype)
            attn_mask = tf.tile(padding, [1, seq_len, 1])
        x = self.norm1(inputs)
        attn_output = self.attention(
            query=x,
            value=x,
            key=x,
            attention_mask=attn_mask,
            use_causal_mask=True,
            training=training,
        )
        attn_output = self.drop1(attn_output, training=training)
        x = inputs + attn_output
        y = self.norm2(x)
        y = self.ffn_dense1(y)
        y = self.ffn_dense2(y)
        y = self.drop2(y, training=training)
        return x + y, mask


class TransformerEncoder(Encoder):
    """Stacked causal Transformer encoder that returns the last timestep embedding."""

    def __init__(
        self,
        units_per_layer: Sequence[int],
        num_heads: int = 4,
        ff_multiplier: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        self.units_per_layer = normalize_units(units_per_layer)
        if len(set(self.units_per_layer)) != 1:
            raise ValueError("TransformerEncoder requires identical units_per_layer for residual connections")
        self.d_model = self.units_per_layer[0]
        self.num_layers = len(self.units_per_layer)
        self.num_heads = max(1, int(num_heads))
        self.ff_multiplier = float(max(ff_multiplier, 1.0))
        self.dropout_rate = float(max(min(dropout, 1.0), 0.0))

        self.input_projection = tf.keras.layers.Dense(self.d_model, name="transformer_input_projection")
        self.blocks = [
            TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ff_multiplier=self.ff_multiplier,
                dropout=self.dropout_rate,
                name=f"transformer_block_{idx}",
            )
            for idx in range(self.num_layers)
        ]
        self.final_norm = LayerNormalization(epsilon=1e-6, name="transformer_final_norm")
        self.last_step = LastTimeStep(name="transformer_last_state")

    def build(self, inputs: tf.Tensor) -> tf.Tensor:
        mask = self._derive_mask(inputs)
        x = self.input_projection(inputs)
        for block in self.blocks:
            x, mask = block(x, mask=mask)
        x = self.final_norm(x)
        return self.last_step(x)

    @staticmethod
    def _derive_mask(inputs: tf.Tensor) -> tf.Tensor | None:
        # Embedding layers with mask_zero=True emit zeros for padding positions,
        # which we can detect to approximate the original mask.
        if inputs.shape.rank is None:
            return None
        approx_mask = tf.reduce_any(tf.not_equal(inputs, 0.0), axis=-1)
        return tf.cast(approx_mask, tf.bool)


def build_transformer_model(
    *,
    total_words: int,
    max_sequence_len: int,
    embedding_output_dims: int,
    units_per_layer: Sequence[int],
    dropout: float,
    column_list: Sequence[str],
    num_heads: int = 4,
    ff_multiplier: float = 4.0,
) -> tuple[Model, list[str]]:
    """Create a Transformer-based autoregressive model for the synthesizer."""
    if total_words <= 0:
        raise ValueError("total_words must be positive to build the model.")
    if max_sequence_len <= 0:
        raise ValueError("max_sequence_len must be positive to build the model.")
    if not column_list:
        raise ValueError("column_list must have at least one column.")

    inputs = Input(shape=(max_sequence_len,), dtype="int32")
    embedding_layer = Embedding(
        total_words,
        embedding_output_dims,
        input_length=max_sequence_len,
        embeddings_regularizer=tf.keras.regularizers.l2(1e-5),
        mask_zero=True,
    )(inputs)

    encoder = TransformerEncoder(
        units_per_layer=units_per_layer,
        num_heads=num_heads,
        ff_multiplier=ff_multiplier,
        dropout=dropout,
    )
    x = encoder.build(embedding_layer)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    modified_column_list = sanitize_column_names(column_list)
    outputs = [
        Dense(total_words, activation="softmax", name=column_name)(x)
        for column_name in modified_column_list
    ]

    model = Model(inputs=inputs, outputs=outputs)
    return model, modified_column_list


__all__ = ["TransformerBlock", "TransformerEncoder", "build_transformer_model"]
