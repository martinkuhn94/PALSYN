from __future__ import annotations

from typing import Sequence

import tensorflow as tf
from keras import Input, Model
from keras.layers import BatchNormalization, Bidirectional, Dense, Dropout, Embedding, GRU

from .base import Encoder, normalize_units, sanitize_column_names, stack_recurrent_layers


class GRUEncoder(Encoder):
    """Stacked GRU encoder that returns the final hidden state.

    Args:
        units_per_layer: Hidden dimension per GRU layer. Each layer except the
            last one outputs sequences so the next layer can consume them.
    """

    def __init__(self, units_per_layer: Sequence[int]) -> None:
        self.units_per_layer = normalize_units(units_per_layer)

    def build(self, inputs: tf.Tensor) -> tf.Tensor:
        return stack_recurrent_layers(
            inputs,
            self.units_per_layer,
            lambda units, return_sequences: GRU(units, return_sequences=return_sequences),
        )


class BidirectionalGRUEncoder(Encoder):
    """Bidirectional GRU stack with one softmax head per column.

    Args:
        units_per_layer: Hidden size for every bidirectional stage. The final
            layer emits the last timestep embedding to drive the output heads.
    """

    def __init__(self, units_per_layer: Sequence[int]) -> None:
        self.units_per_layer = normalize_units(units_per_layer)

    def build(self, inputs: tf.Tensor) -> tf.Tensor:
        return stack_recurrent_layers(
            inputs,
            self.units_per_layer,
            lambda units, return_sequences: Bidirectional(
                GRU(units, return_sequences=return_sequences)
            ),
        )


def build_gru_model(
    *,
    total_words: int,
    max_sequence_len: int,
    embedding_output_dims: int,
    units_per_layer: Sequence[int],
    dropout: float,
    column_list: Sequence[str],
    bidirectional: bool = False,
) -> tuple[Model, list[str]]:
    """Create the GRU-based multi-head prediction model used by the synthesizer."""
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

    encoder: Encoder
    if bidirectional:
        encoder = BidirectionalGRUEncoder(units_per_layer)
    else:
        encoder = GRUEncoder(units_per_layer)
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


__all__ = ["GRUEncoder", "BidirectionalGRUEncoder", "build_gru_model"]
