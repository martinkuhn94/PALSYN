from __future__ import annotations

from typing import Callable, Sequence

import tensorflow as tf
from keras import activations, initializers
from keras.layers import RNN

from .base import Encoder, normalize_units


@tf.keras.utils.register_keras_serializable(package="palsyn")
class EchoStateCell(tf.keras.layers.AbstractRNNCell):
    """Non-trainable reservoir cell used by Echo State Networks."""

    def __init__(
        self,
        units: int,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.1,
        leak_rate: float = 1.0,
        bias_scale: float = 0.0,
        activation: str | Callable[[tf.Tensor], tf.Tensor] = "tanh",
        seed: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if units <= 0:
            raise ValueError("units must be positive for EchoStateCell")
        self.units = int(units)
        self.spectral_radius = float(max(spectral_radius, 1e-6))
        self.input_scaling = float(max(input_scaling, 1e-6))
        self.leak_rate = float(min(max(leak_rate, 0.0), 1.0))
        self.bias_scale = float(max(bias_scale, 0.0))
        self.activation_fn = activations.get(activation)
        self.activation_serialized = activations.serialize(self.activation_fn)
        self.seed = None if seed is None else int(seed)
        self.supports_masking = True
        self.input_kernel: tf.Variable | None = None
        self.recurrent_kernel: tf.Variable | None = None
        self.bias: tf.Variable | None = None

    @property
    def state_size(self) -> int:
        return self.units

    @property
    def output_size(self) -> int:
        return self.units

    def build(self, input_shape: tf.TensorShape) -> None:
        input_dim = int(input_shape[-1])
        input_init = initializers.RandomUniform(
            minval=-self.input_scaling,
            maxval=self.input_scaling,
            seed=self.seed,
        )
        self.input_kernel = self.add_weight(
            name="input_kernel",
            shape=(input_dim, self.units),
            initializer=input_init,
            trainable=False,
        )
        recurrent_init = initializers.RandomUniform(
            minval=-1.0,
            maxval=1.0,
            seed=None if self.seed is None else self.seed + 1,
        )
        self.recurrent_kernel = self.add_weight(
            name="recurrent_kernel",
            shape=(self.units, self.units),
            initializer=recurrent_init,
            trainable=False,
        )
        scaled = self._scale_to_spectral_radius(self.recurrent_kernel)
        self.recurrent_kernel.assign(scaled)
        if self.bias_scale > 0.0:
            bias_init = initializers.RandomUniform(
                minval=-self.bias_scale,
                maxval=self.bias_scale,
                seed=None if self.seed is None else self.seed + 2,
            )
        else:
            bias_init = initializers.Zeros()
        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer=bias_init,
            trainable=False,
        )
        self.built = True

    def _scale_to_spectral_radius(self, matrix: tf.Variable) -> tf.Tensor:
        eigvals = tf.linalg.eigvals(tf.cast(matrix, tf.complex64))
        radius = tf.reduce_max(tf.abs(eigvals))
        radius = tf.maximum(radius, tf.constant(1e-6, dtype=radius.dtype))
        scale = tf.cast(self.spectral_radius, matrix.dtype) / tf.cast(radius, matrix.dtype)
        return tf.cast(scale, matrix.dtype) * matrix

    def call(self, inputs: tf.Tensor, states: list[tf.Tensor]) -> tuple[tf.Tensor, list[tf.Tensor]]:
        prev_state = states[0]
        assert self.input_kernel is not None
        assert self.recurrent_kernel is not None
        assert self.bias is not None
        reservoir_input = tf.matmul(inputs, self.input_kernel)
        reservoir_input += tf.matmul(prev_state, self.recurrent_kernel)
        reservoir_input = tf.nn.bias_add(reservoir_input, self.bias)
        activated = self.activation_fn(reservoir_input)
        if self.leak_rate < 1.0:
            new_state = prev_state + self.leak_rate * (activated - prev_state)
        else:
            new_state = activated
        return new_state, [new_state]

    def get_config(self) -> dict[str, float | int | str | None]:
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "spectral_radius": self.spectral_radius,
                "input_scaling": self.input_scaling,
                "leak_rate": self.leak_rate,
                "bias_scale": self.bias_scale,
                "activation": self.activation_serialized,
                "seed": self.seed,
            }
        )
        return config


class EchoStateNetworkEncoder(Encoder):
    """Reservoir-computing encoder built from stacked Echo State layers."""

    def __init__(
        self,
        units_per_layer: Sequence[int],
        spectral_radius: float = 0.9,
        input_scaling: float = 0.1,
        leak_rate: float = 1.0,
        bias_scale: float = 0.0,
        activation: str = "tanh",
        seed: int | None = None,
    ) -> None:
        self.units_per_layer = normalize_units(units_per_layer)
        self.spectral_radius = float(max(spectral_radius, 1e-6))
        self.input_scaling = float(max(input_scaling, 1e-6))
        self.leak_rate = float(min(max(leak_rate, 0.0), 1.0))
        self.bias_scale = float(max(bias_scale, 0.0))
        self.activation = activation
        self.seed = None if seed is None else int(seed)

    def build(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs
        last_idx = len(self.units_per_layer) - 1
        for idx, units in enumerate(self.units_per_layer):
            cell_seed = None if self.seed is None else self.seed + idx
            cell = EchoStateCell(
                units=units,
                spectral_radius=self.spectral_radius,
                input_scaling=self.input_scaling,
                leak_rate=self.leak_rate,
                bias_scale=self.bias_scale,
                activation=self.activation,
                seed=cell_seed,
                name=f"echo_state_cell_{idx}",
            )
            rnn = RNN(
                cell,
                return_sequences=idx < last_idx,
                name=f"echo_state_layer_{idx}",
            )
            x = rnn(x)
        return x


__all__ = ["EchoStateCell", "EchoStateNetworkEncoder"]
