from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Sequence

import tensorflow as tf
from keras.layers import Layer

LayerFactory = Callable[[int, bool], Layer]


class Encoder(ABC):
    """Abstract encoder interface used by the synthesizer.

    Concrete encoders encapsulate the recurrent/convolutional backbone and
    expose a ``build`` method that receives the embedded input tensor and
    returns the final timestep embedding that drives the prediction heads.
    """

    @abstractmethod
    def build(self, inputs: tf.Tensor) -> tf.Tensor:
        """Create the encoder stack starting from the provided tensor."""


def normalize_units(units: Sequence[int]) -> list[int]:
    """Validate and convert a sequence of units into a positive integer list."""
    normalized = [int(value) for value in units]
    if not normalized:
        raise ValueError("units_per_layer must contain at least one value")
    if not all(value > 0 for value in normalized):
        raise ValueError("units_per_layer values must be positive integers")
    return normalized


def stack_recurrent_layers(
    inputs: tf.Tensor,
    units: Sequence[int],
    layer_factory: LayerFactory,
) -> tf.Tensor:
    """Apply stacked recurrent layers with consistent return_sequences handling."""
    x = inputs
    last_index = len(units) - 1
    for idx, layer_units in enumerate(units):
        return_sequences = idx < last_index
        layer = layer_factory(int(layer_units), return_sequences)
        x = layer(x)
    return x


@tf.keras.utils.register_keras_serializable(package="palsyn")
class LastTimeStep(Layer):
    """Serializable helper that extracts the final timestep embedding."""

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if inputs.shape.rank != 3:
            raise ValueError("LastTimeStep expects rank-3 inputs: [batch, time, features]")
        return inputs[:, -1, :]

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        if len(input_shape) != 3:
            raise ValueError("LastTimeStep expects rank-3 inputs: [batch, time, features]")
        return tf.TensorShape([input_shape[0], input_shape[2]])

    def get_config(self) -> dict[str, int]:
        return super().get_config()


__all__ = ["Encoder", "LayerFactory", "LastTimeStep", "normalize_units", "stack_recurrent_layers"]
