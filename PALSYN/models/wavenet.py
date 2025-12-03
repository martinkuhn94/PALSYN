from __future__ import annotations

from typing import Any, Sequence

import tensorflow as tf
from keras.layers import Activation, Conv1D, LayerNormalization

from .base import Encoder, LastTimeStep, normalize_units


@tf.keras.utils.register_keras_serializable(package="palsyn")
class WaveNetBlock(tf.keras.layers.Layer):
    """Single dilated causal convolution block with residual and skip outputs."""

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        dilation_rate: int,
        skip_channels: int,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.dilation_rate = int(dilation_rate)
        self.skip_channels = int(skip_channels)
        self.filter_conv = Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding="causal",
            dilation_rate=self.dilation_rate,
            name=f"{self.name}_filter",
        )
        self.gate_conv = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="causal",
            dilation_rate=dilation_rate,
            name=f"{self.name}_gate",
        )
        self.residual_conv = Conv1D(self.filters, kernel_size=1, padding="same", name=f"{self.name}_residual")
        self.skip_conv = Conv1D(self.skip_channels, kernel_size=1, padding="same", name=f"{self.name}_skip")
        self.input_projection = Conv1D(self.filters, kernel_size=1, padding="same", name=f"{self.name}_input_proj")

    def call(self, inputs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        filter_out = tf.nn.tanh(self.filter_conv(inputs))
        gate_out = tf.nn.sigmoid(self.gate_conv(inputs))
        z = filter_out * gate_out
        residual = self.residual_conv(z)
        projected_inputs = self.input_projection(inputs)
        output = projected_inputs + residual
        skip = self.skip_conv(z)
        return output, skip

    def get_config(self) -> dict[str, int]:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "dilation_rate": self.dilation_rate,
                "skip_channels": self.skip_channels,
            }
        )
        return config


class WaveNetEncoder(Encoder):
    """WaveNet-style encoder built from dilated causal convolutions."""

    def __init__(
        self,
        units_per_layer: Sequence[int],
        kernel_size: int = 2,
        dilation_base: int = 2,
        skip_channels: int | None = None,
    ) -> None:
        self.units_per_layer = normalize_units(units_per_layer)
        self.kernel_size = int(kernel_size)
        self.dilation_base = max(1, int(dilation_base))
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        self.skip_channels = int(skip_channels) if skip_channels is not None else self.units_per_layer[-1]

        self.input_projection = tf.keras.layers.Dense(self.units_per_layer[0], name="wavenet_input_projection")
        self.blocks: list[WaveNetBlock] = []
        dilation = 1
        for idx, filters in enumerate(self.units_per_layer):
            self.blocks.append(
                WaveNetBlock(
                    filters=filters,
                    kernel_size=self.kernel_size,
                    dilation_rate=dilation,
                    skip_channels=self.skip_channels,
                    name=f"wavenet_block_{idx}",
                )
            )
            dilation *= self.dilation_base

        self.skip_activation = Activation("relu")
        self.skip_conv = Conv1D(self.skip_channels, kernel_size=1, padding="same", name="wavenet_skip_merge")
        self.output_projection = Conv1D(
            self.skip_channels, kernel_size=1, padding="same", name="wavenet_output_projection"
        )
        self.final_norm = LayerNormalization(epsilon=1e-6, name="wavenet_final_norm")
        self.last_step = LastTimeStep(name="wavenet_last_state")

    def build(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.input_projection(inputs)
        skip_total: tf.Tensor | None = None
        last_residual: tf.Tensor | None = None
        for block in self.blocks:
            x, skip = block(x)
            skip_total = skip if skip_total is None else skip_total + skip
            last_residual = x
        if skip_total is None or last_residual is None:
            raise ValueError("WaveNetEncoder requires at least one block")
        skip_total = self.skip_activation(skip_total)
        skip_total = self.skip_conv(skip_total)
        residual_projection = self.output_projection(last_residual)
        combined = self.final_norm(skip_total + residual_projection)
        return self.last_step(combined)


__all__ = ["WaveNetBlock", "WaveNetEncoder"]
