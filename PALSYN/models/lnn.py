from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Sequence

import numpy as np
import tensorflow as tf
from keras import Model, Input
from keras.layers import BatchNormalization, Dense, Dropout, Embedding, RNN

from .base import Encoder, normalize_units, sanitize_column_names


@tf.keras.utils.register_keras_serializable(package="palsyn")
class LiquidTimeConstantCell(tf.keras.layers.AbstractRNNCell):
    """Liquid neural network cell with learnable time constants and sparsity.

    Args:
        units: Number of neurons in the recurrent state.
        tau_min: Lower bound for the learnable time constant prior.
        tau_max: Upper bound for the time constant prior.
        connectivity: Fraction of recurrent edges retained (dropout mask).
        activation: Optional activation applied before the liquid dynamics.
    """

    def __init__(
        self,
        units: int,
        tau_min: float = 0.1,
        tau_max: float = 2.0,
        connectivity: float = 0.3,
        activation: str = "tanh",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.units = int(units)
        self.tau_min = max(float(tau_min), 1e-3)
        self.tau_max = max(float(tau_max), self.tau_min + 1e-3)
        self.connectivity = float(np.clip(connectivity, 0.0, 1.0))
        self.activation_fn = tf.keras.activations.get(activation)
        self._recurrent_mask: tf.Tensor | None = None

    @property
    def state_size(self) -> int:
        return self.units

    @property
    def output_size(self) -> int:
        return self.units

    def build(self, input_shape: Sequence[int]) -> None:
        input_dim = int(input_shape[-1])
        kernel_init = tf.keras.initializers.GlorotUniform()
        recurrent_init = tf.keras.initializers.Orthogonal()
        tau_init = tf.keras.initializers.RandomUniform(self.tau_min, self.tau_max)

        self.input_kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=kernel_init,
            name="input_kernel",
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer=recurrent_init,
            name="recurrent_kernel",
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            name="bias",
        )
        self.time_constants = self.add_weight(
            shape=(self.units,),
            initializer=tau_init,
            name="time_constants",
        )

        if self.connectivity < 1.0:
            mask = (np.random.rand(self.units, self.units) < self.connectivity).astype("float32")
            self._recurrent_mask = tf.constant(mask)
        else:
            self._recurrent_mask = None

        super().build(input_shape)

    def call(
        self, inputs: tf.Tensor, states: Sequence[tf.Tensor]
    ) -> tuple[tf.Tensor, list[tf.Tensor]]:
        prev_state = states[0]
        effective_kernel = self.recurrent_kernel
        if self._recurrent_mask is not None:
            effective_kernel = effective_kernel * self._recurrent_mask

        input_term = tf.matmul(inputs, self.input_kernel)
        recurrent_term = tf.matmul(prev_state, effective_kernel)
        pre_activation = input_term + recurrent_term + self.bias
        if self.activation_fn is not None:
            pre_activation = self.activation_fn(pre_activation)

        tau = tf.nn.softplus(self.time_constants) + 1e-3
        delta = (pre_activation - prev_state) / tau
        next_state = prev_state + delta
        return next_state, [next_state]


class LiquidNeuralNetworkEncoder(Encoder):
    """Encoder composed of stacked liquid neural network cells.

    Args:
        units_per_layer: Number of liquid neurons per recurrent block.
        tau_min: Minimum learnable time constant shared across layers.
        tau_max: Maximum learnable time constant shared across layers.
        connectivity: Sparsity applied to the recurrent kernels.
    """

    def __init__(
        self,
        units_per_layer: Sequence[int],
        tau_min: float = 0.1,
        tau_max: float = 2.0,
        connectivity: float = 0.3,
    ) -> None:
        self.units_per_layer = normalize_units(units_per_layer)
        self.tau_min = max(float(tau_min), 1e-3)
        self.tau_max = max(float(tau_max), self.tau_min + 1e-3)
        self.connectivity = float(np.clip(connectivity, 0.0, 1.0))

    def build(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs
        last_index = len(self.units_per_layer) - 1
        for idx, units in enumerate(self.units_per_layer):
            return_sequences = idx < last_index
            cell = LiquidTimeConstantCell(
                units=units,
                tau_min=self.tau_min,
                tau_max=self.tau_max,
                connectivity=self.connectivity,
                name=f"liquid_cell_{idx}",
            )
            x = RNN(cell, return_sequences=return_sequences, name=f"liquid_layer_{idx}")(x)
        return x


def build_lnn_model(
    *,
    total_words: int,
    max_sequence_len: int,
    embedding_output_dims: int,
    units_per_layer: Sequence[int],
    dropout: float,
    column_list: Sequence[str],
    tau_min: float = 0.1,
    tau_max: float = 2.0,
    connectivity: float = 0.3,
) -> tuple[Model, list[str]]:
    """Create a liquid neural network-based autoregressive model."""
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

    encoder = LiquidNeuralNetworkEncoder(
        units_per_layer=units_per_layer,
        tau_min=tau_min,
        tau_max=tau_max,
        connectivity=connectivity,
    )
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


__all__ = ["LiquidTimeConstantCell", "LiquidNeuralNetworkEncoder", "build_lnn_model"]
