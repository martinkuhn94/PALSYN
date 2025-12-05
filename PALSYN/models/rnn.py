from __future__ import annotations

from typing import Sequence

import tensorflow as tf
from keras import Input, Model
from keras.layers import BatchNormalization, Bidirectional, Dense, Dropout, Embedding, SimpleRNN

from .base import Encoder, normalize_units, sanitize_column_names, stack_recurrent_layers


class SimpleRNNEncoder(Encoder):
    """Classic Elman RNN stack that outputs the last hidden state.

    Args:
        units_per_layer: Hidden dimension for each SimpleRNN layer in the
            stack. Earlier layers propagate sequences; the last layer pools.
    """

    def __init__(self, units_per_layer: Sequence[int]) -> None:
        self.units_per_layer = normalize_units(units_per_layer)

    def build(self, inputs: tf.Tensor) -> tf.Tensor:
        return stack_recurrent_layers(
            inputs,
            self.units_per_layer,
            lambda units, return_sequences: SimpleRNN(units, return_sequences=return_sequences),
        )


class BidirectionalSimpleRNNEncoder(Encoder):
    """Bidirectional SimpleRNN variant for lightweight baselines.

    Args:
        units_per_layer: Hidden units for every bidirectional stage, ordered
            from input to output. Only the last layer collapses time.
    """

    def __init__(self, units_per_layer: Sequence[int]) -> None:
        self.units_per_layer = normalize_units(units_per_layer)

    def build(self, inputs: tf.Tensor) -> tf.Tensor:
        return stack_recurrent_layers(
            inputs,
            self.units_per_layer,
            lambda units, return_sequences: Bidirectional(
                SimpleRNN(units, return_sequences=return_sequences)
            ),
        )


def build_rnn_model(
    *,
    total_words: int,
    max_sequence_len: int,
    embedding_output_dims: int,
    units_per_layer: Sequence[int],
    dropout: float,
    column_list: Sequence[str],
    bidirectional: bool = False,
) -> tuple[Model, list[str]]:
    """Create a SimpleRNN-based autoregressive model for the synthesizer."""
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
        encoder = BidirectionalSimpleRNNEncoder(units_per_layer)
    else:
        encoder = SimpleRNNEncoder(units_per_layer)
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


__all__ = ["SimpleRNNEncoder", "BidirectionalSimpleRNNEncoder", "build_rnn_model"]
