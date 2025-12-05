from __future__ import annotations

from typing import Sequence

import tensorflow as tf
from keras import Input, Model
from keras.layers import LSTM, BatchNormalization, Bidirectional, Dense, Dropout, Embedding

from .base import normalize_units, sanitize_column_names


def _stack_lstm_layers(
    inputs: tf.Tensor, units_per_layer: Sequence[int], *, bidirectional: bool
) -> tf.Tensor:
    """Apply a stack of LSTM layers, returning only the last timestep embedding."""
    units = normalize_units(units_per_layer)
    x = inputs
    last_index = len(units) - 1
    for idx, hidden_units in enumerate(units):
        return_sequences = idx < last_index
        layer = LSTM(hidden_units, return_sequences=return_sequences)
        if bidirectional:
            layer = Bidirectional(layer)
        x = layer(x)
    return x


def build_lstm_model(
    *,
    total_words: int,
    max_sequence_len: int,
    embedding_output_dims: int,
    units_per_layer: Sequence[int],
    dropout: float,
    column_list: Sequence[str],
    bidirectional: bool = False,
) -> tuple[Model, list[str]]:
    """Create the LSTM-based multi-head prediction model used by the synthesizer."""
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

    x = _stack_lstm_layers(embedding_layer, units_per_layer, bidirectional=bidirectional)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    modified_column_list = sanitize_column_names(column_list)
    outputs = [
        Dense(total_words, activation="softmax", name=column_name)(x)
        for column_name in modified_column_list
    ]

    model = Model(inputs=inputs, outputs=outputs)
    return model, modified_column_list


__all__ = ["build_lstm_model"]
