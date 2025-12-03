from __future__ import annotations

from typing import Sequence

import tensorflow as tf
from keras.layers import Bidirectional, SimpleRNN

from .base import Encoder, normalize_units, stack_recurrent_layers


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


__all__ = ["SimpleRNNEncoder", "BidirectionalSimpleRNNEncoder"]
