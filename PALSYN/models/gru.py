from __future__ import annotations

from typing import Sequence

import tensorflow as tf
from keras.layers import Bidirectional, GRU

from .base import Encoder, normalize_units, stack_recurrent_layers


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


__all__ = ["GRUEncoder", "BidirectionalGRUEncoder"]
