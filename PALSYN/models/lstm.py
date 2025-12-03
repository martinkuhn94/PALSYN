from __future__ import annotations

from typing import Sequence

import tensorflow as tf
from keras.layers import Bidirectional, LSTM

from .base import Encoder, normalize_units, stack_recurrent_layers


class LSTMEncoder(Encoder):
    """Stacked LSTM encoder that emits the last hidden state.

    Args:
        units_per_layer: Number of units per layer. Each entry creates one
            sequential LSTM layer where all but the final layer return the
            full sequence to feed the next layer.
    """

    def __init__(self, units_per_layer: Sequence[int]) -> None:
        self.units_per_layer = normalize_units(units_per_layer)

    def build(self, inputs: tf.Tensor) -> tf.Tensor:
        return stack_recurrent_layers(
            inputs,
            self.units_per_layer,
            lambda units, return_sequences: LSTM(units, return_sequences=return_sequences),
        )


class BidirectionalLSTMEncoder(Encoder):
    """Stacked bidirectional LSTM encoder with last-state pooling.

    Args:
        units_per_layer: Hidden size for each bidirectional layer. Intermediate
            layers keep the temporal dimension while the final layer collapses
            to the last timestep representation.
    """

    def __init__(self, units_per_layer: Sequence[int]) -> None:
        self.units_per_layer = normalize_units(units_per_layer)

    def build(self, inputs: tf.Tensor) -> tf.Tensor:
        return stack_recurrent_layers(
            inputs,
            self.units_per_layer,
            lambda units, return_sequences: Bidirectional(
                LSTM(units, return_sequences=return_sequences)
            ),
        )


__all__ = ["LSTMEncoder", "BidirectionalLSTMEncoder"]
