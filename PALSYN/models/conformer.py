from __future__ import annotations

from typing import Sequence

import tensorflow as tf
from keras.layers import BatchNormalization, Conv1D, DepthwiseConv1D, Dropout, LayerNormalization, MultiHeadAttention

from .base import Encoder, LastTimeStep, normalize_units


class FeedForwardModule(tf.keras.layers.Layer):
    """Position-wise feed-forward used inside Conformer blocks."""

    def __init__(
        self,
        dim: int,
        expansion_factor: float = 4.0,
        dropout: float = 0.1,
        activation: str = "swish",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.dim = dim
        self.expansion_factor = float(expansion_factor)
        self.dropout_rate = float(dropout)
        self.activation_name = activation
        inner_dim = int(dim * expansion_factor)
        self.seq = tf.keras.Sequential(
            [
                LayerNormalization(epsilon=1e-6),
                tf.keras.layers.Dense(inner_dim, activation=activation),
                Dropout(dropout),
                tf.keras.layers.Dense(dim),
                Dropout(dropout),
            ],
            name=f"{self.name}_ffn" if self.name else None,
        )

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        return self.seq(inputs, training=training)

    def get_config(self) -> dict[str, float | str]:
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "expansion_factor": self.expansion_factor,
                "dropout": self.dropout_rate,
                "activation": self.activation_name,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="palsyn")
class ConformerBlock(tf.keras.layers.Layer):
    """Single Conformer block with FFN, attention, and convolutional modules."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_multiplier: float,
        conv_kernel_size: int,
        dropout: float,
        activation: str = "swish",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.dim = dim
        self.num_heads = int(num_heads)
        self.ff_multiplier = float(ff_multiplier)
        self.conv_kernel_size = int(conv_kernel_size)
        self.dropout_rate = float(dropout)
        self.activation_name = activation
        self.ffn1 = FeedForwardModule(dim, ff_multiplier, dropout, activation, name=f"{self.name}_ffn1")
        self.ffn2 = FeedForwardModule(dim, ff_multiplier, dropout, activation, name=f"{self.name}_ffn2")
        self.mha_norm = LayerNormalization(epsilon=1e-6, name=f"{self.name}_mha_norm")
        key_dim = max(1, dim // self.num_heads)
        self.mha = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout_rate,
            name=f"{self.name}_mha",
        )
        self.mha_dropout = Dropout(self.dropout_rate)

        self.conv_norm = LayerNormalization(epsilon=1e-6, name=f"{self.name}_conv_norm")
        self.pointwise_conv1 = Conv1D(filters=dim * 2, kernel_size=1, padding="same", name=f"{self.name}_pw1")
        self.depthwise_conv = DepthwiseConv1D(
            kernel_size=conv_kernel_size,
            padding="same",
            name=f"{self.name}_dw",
        )
        self.batch_norm = BatchNormalization(momentum=0.99, epsilon=1e-3, name=f"{self.name}_bn")
        self.activation = tf.keras.layers.Activation(self.activation_name)
        self.pointwise_conv2 = Conv1D(filters=dim, kernel_size=1, padding="same", name=f"{self.name}_pw2")
        self.conv_dropout = Dropout(self.dropout_rate)

        self.final_norm = LayerNormalization(epsilon=1e-6, name=f"{self.name}_final_norm")

    def _glu(self, inputs: tf.Tensor) -> tf.Tensor:
        a, b = tf.split(inputs, num_or_size_splits=2, axis=-1)
        return a * tf.nn.sigmoid(b)

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        x = inputs + 0.5 * self.ffn1(inputs, training=training)

        y = self.mha_norm(x)
        attn = self.mha(y, y, training=training)
        attn = self.mha_dropout(attn, training=training)
        x = x + attn

        y = self.conv_norm(x)
        y = self.pointwise_conv1(y)
        y = self._glu(y)
        y = self.depthwise_conv(y)
        y = self.batch_norm(y, training=training)
        y = self.activation(y)
        y = self.pointwise_conv2(y)
        y = self.conv_dropout(y, training=training)
        x = x + y

        x = x + 0.5 * self.ffn2(x, training=training)
        return self.final_norm(x)

    def get_config(self) -> dict[str, float | str]:
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "ff_multiplier": self.ff_multiplier,
                "conv_kernel_size": self.conv_kernel_size,
                "dropout": self.dropout_rate,
                "activation": self.activation_name,
            }
        )
        return config


class ConformerEncoder(Encoder):
    """Conformer encoder composed of stacked convolution-attention blocks."""

    def __init__(
        self,
        units_per_layer: Sequence[int],
        num_heads: int = 4,
        ff_multiplier: float = 4.0,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
        activation: str = "swish",
    ) -> None:
        self.units_per_layer = normalize_units(units_per_layer)
        self.num_heads = int(num_heads)
        self.ff_multiplier = float(ff_multiplier)
        self.conv_kernel_size = int(conv_kernel_size)
        self.dropout = float(dropout)
        self.activation = activation
        self.projections: list[tf.keras.layers.Layer] = []
        self.blocks: list[ConformerBlock] = []
        for idx, units in enumerate(self.units_per_layer):
            self.projections.append(tf.keras.layers.Dense(units, name=f"conformer_projection_{idx}"))
            self.blocks.append(
                ConformerBlock(
                    dim=units,
                    num_heads=self.num_heads,
                    ff_multiplier=self.ff_multiplier,
                    conv_kernel_size=self.conv_kernel_size,
                    dropout=self.dropout,
                    activation=self.activation,
                    name=f"conformer_block_{idx}",
                )
            )
        self.final_norm = LayerNormalization(epsilon=1e-6, name="conformer_encoder_norm")
        self.last_step = LastTimeStep(name="conformer_last_state")

    def build(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs
        for projection, block in zip(self.projections, self.blocks):
            x = projection(x)
            x = block(x)
        x = self.final_norm(x)
        return self.last_step(x)


__all__ = ["ConformerEncoder"]
