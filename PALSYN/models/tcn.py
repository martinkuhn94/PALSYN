from __future__ import annotations

from typing import Sequence

import tensorflow as tf
from keras.layers import Conv1D

from .base import Encoder, LastTimeStep, normalize_units


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


__all__ = ["TCNEncoder"]
