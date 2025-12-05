from __future__ import annotations

from typing import Sequence

import tensorflow as tf
from keras import Input, Model
from keras.layers import BatchNormalization, Conv1D, Dense, Dropout, Embedding

from .base import Encoder, LastTimeStep, normalize_units, sanitize_column_names


class TCNEncoder(Encoder):
    """Temporal convolutional encoder with exponentially increasing dilation.

    Args:
        filters_per_layer: Number of convolution filters per residual block.
        kernel_size: Width of the causal convolutional kernel.
        activation: Activation function applied after each convolution.
        padding: Padding mode for the convolutions (defaults to ``"causal"``).
        dilation_base: Base used to exponentiate dilation per layer.
    """

    def __init__(
        self,
        filters_per_layer: Sequence[int],
        kernel_size: int = 3,
        activation: str = "relu",
        padding: str = "causal",
        dilation_base: int = 2,
    ) -> None:
        self.filters_per_layer = normalize_units(filters_per_layer)
        self.kernel_size = int(kernel_size)
        self.activation = activation
        self.padding = padding
        self.dilation_base = int(dilation_base)
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if self.dilation_base <= 0:
            raise ValueError("dilation_base must be positive")

    def build(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs
        for idx, filters in enumerate(self.filters_per_layer):
            dilation = self.dilation_base**idx
            x = Conv1D(
                filters=filters,
                kernel_size=self.kernel_size,
                padding=self.padding,
                dilation_rate=dilation,
                activation=self.activation,
            )(x)
        return LastTimeStep(name="tcn_last_state")(x)


def build_tcn_model(
    *,
    total_words: int,
    max_sequence_len: int,
    embedding_output_dims: int,
    filters_per_layer: Sequence[int],
    dropout: float,
    column_list: Sequence[str],
    kernel_size: int = 3,
    activation: str = "relu",
    padding: str = "causal",
    dilation_base: int = 2,
) -> tuple[Model, list[str]]:
    """Create a TCN-based autoregressive prediction model for the synthesizer."""
    if total_words <= 0:
        raise ValueError("total_words must be positive to build the model.")
    if max_sequence_len <= 0:
        raise ValueError("max_sequence_len must be positive to build the model.")
    if not column_list:
        raise ValueError("column_list must have at least one column.")

    inputs = Input(shape=(max_sequence_len,), dtype="int32")
    embedding_layer = Embedding(
        input_dim=total_words,
        output_dim=embedding_output_dims,
        input_length=max_sequence_len,
        embeddings_regularizer=tf.keras.regularizers.l2(1e-5),
        mask_zero=True,
    )(inputs)

    encoder = TCNEncoder(
        filters_per_layer=filters_per_layer,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
        dilation_base=dilation_base,
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


__all__ = ["TCNEncoder", "build_tcn_model"]
