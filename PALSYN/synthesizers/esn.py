from __future__ import annotations

from typing import Any, Sequence

import tensorflow as tf

from PALSYN.models.esn import build_esn_model

from .base import BaseSynthesizer


class ESNSynthesizer(BaseSynthesizer):
    MODEL_TYPE = "ESN"

    def __init__(
        self,
        *,
        max_clusters: int = 10,
        trace_quantile: float = 0.95,
        epsilon: float | None = None,
        seed: int | None = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        validation_split: float = 0.1,
        checkpoint_path: str | None = None,
        l2_norm_clip: float = 1.5,
        embedding_output_dims: int = 16,
        units_per_layer: Sequence[int] | None = None,
        dropout: float = 0.0,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.1,
        leak_rate: float = 1.0,
        bias_scale: float = 0.0,
        activation: str = "tanh",
        pre_processing: dict[str, Any] | None = None,
        model: dict[str, Any] | None = None,
        dp_optimizer: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a differential privacy-aware Echo State Network synthesizer."""
        preprocessing_cfg: dict[str, Any] = {
            "max_clusters": max_clusters,
            "trace_quantile": trace_quantile,
            "seed": seed,
        }
        if pre_processing:
            preprocessing_cfg.update(pre_processing)

        model_cfg: dict[str, Any] = {
            "epochs": epochs,
            "batch_size": batch_size,
            "validation_split": validation_split,
            "checkpoint_path": checkpoint_path,
            "embedding_output_dims": embedding_output_dims,
            "units_per_layer": units_per_layer,
            "dropout": dropout,
            "spectral_radius": spectral_radius,
            "input_scaling": input_scaling,
            "leak_rate": leak_rate,
            "bias_scale": bias_scale,
            "activation": activation,
        }
        if model:
            model_cfg.update(model)

        dp_optimizer_cfg: dict[str, Any] = {
            "epsilon": epsilon,
            "learning_rate": learning_rate,
            "l2_norm_clip": l2_norm_clip,
        }
        if dp_optimizer:
            dp_optimizer_cfg.update(dp_optimizer)

        super().__init__(
            max_clusters=int(preprocessing_cfg.get("max_clusters", max_clusters)),
            trace_quantile=float(preprocessing_cfg.get("trace_quantile", trace_quantile)),
            epsilon=dp_optimizer_cfg.get("epsilon"),
            seed=preprocessing_cfg.get("seed"),
        )
        self._configure_training(
            epochs=int(model_cfg.get("epochs", epochs)),
            batch_size=int(model_cfg.get("batch_size", batch_size)),
            validation_split=float(model_cfg.get("validation_split", validation_split)),
            checkpoint_path=model_cfg.get("checkpoint_path"),
        )
        self._configure_optimizer(
            learning_rate=float(dp_optimizer_cfg.get("learning_rate", learning_rate)),
            l2_norm_clip=float(dp_optimizer_cfg.get("l2_norm_clip", l2_norm_clip)),
        )

        units_setting = model_cfg.get("units_per_layer")
        units = list(units_setting) if units_setting else [128, 64]
        if not units or not all(isinstance(u, int) and u > 0 for u in units):
            raise ValueError("units_per_layer must be a non-empty sequence of positive integers.")
        self.units_per_layer = units
        self.embedding_output_dims = int(
            model_cfg.get("embedding_output_dims", embedding_output_dims)
        )
        self.dropout = float(model_cfg.get("dropout", dropout))
        self.spectral_radius = float(model_cfg.get("spectral_radius", spectral_radius))
        self.input_scaling = float(model_cfg.get("input_scaling", input_scaling))
        self.leak_rate = float(model_cfg.get("leak_rate", leak_rate))
        self.bias_scale = float(model_cfg.get("bias_scale", bias_scale))
        self.activation = str(model_cfg.get("activation", activation))

    def _get_model_specific_init_args(self) -> dict[str, Any]:
        return {
            "embedding_output_dims": self.embedding_output_dims,
            "units_per_layer": self.units_per_layer,
            "dropout": self.dropout,
            "spectral_radius": self.spectral_radius,
            "input_scaling": self.input_scaling,
            "leak_rate": self.leak_rate,
            "bias_scale": self.bias_scale,
            "activation": self.activation,
        }

    def _build_model_impl(self) -> tuple[tf.keras.Model, list[str]]:
        model, modified_columns = build_esn_model(
            total_words=self.total_words,
            max_sequence_len=self.max_sequence_len,
            embedding_output_dims=self.embedding_output_dims,
            units_per_layer=self.units_per_layer,
            dropout=self.dropout,
            column_list=self.column_list,
            spectral_radius=self.spectral_radius,
            input_scaling=self.input_scaling,
            leak_rate=self.leak_rate,
            bias_scale=self.bias_scale,
            activation=self.activation,
            seed=self.seed,
        )
        return model, modified_columns


__all__ = ["ESNSynthesizer"]
