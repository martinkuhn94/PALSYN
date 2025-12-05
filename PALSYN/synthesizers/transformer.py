from __future__ import annotations

from typing import Any, Sequence

import tensorflow as tf

from PALSYN.models.transformer import build_transformer_model

from .base import BaseSynthesizer


class TransformerSynthesizer(BaseSynthesizer):
    MODEL_TYPE = "TRANSFORMER"

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
        embedding_output_dims: int = 32,
        units_per_layer: Sequence[int] | None = None,
        dropout: float = 0.1,
        num_heads: int = 4,
        ff_multiplier: float = 4.0,
        pre_processing: dict[str, Any] | None = None,
        model: dict[str, Any] | None = None,
        dp_optimizer: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a Transformer synthesizer with grouped configuration dicts."""
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
            "num_heads": num_heads,
            "ff_multiplier": ff_multiplier,
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
        units = list(units_setting) if units_setting else [128, 128]
        if not units or not all(isinstance(u, int) and u > 0 for u in units):
            raise ValueError("units_per_layer must be a non-empty sequence of positive integers.")
        self.units_per_layer = units
        self.embedding_output_dims = int(
            model_cfg.get("embedding_output_dims", embedding_output_dims)
        )
        self.dropout = float(model_cfg.get("dropout", dropout))
        self.num_heads = int(model_cfg.get("num_heads", num_heads))
        self.ff_multiplier = float(model_cfg.get("ff_multiplier", ff_multiplier))

    def _get_model_specific_init_args(self) -> dict[str, Any]:
        return {
            "embedding_output_dims": self.embedding_output_dims,
            "units_per_layer": self.units_per_layer,
            "dropout": self.dropout,
            "num_heads": self.num_heads,
            "ff_multiplier": self.ff_multiplier,
        }

    def _build_model_impl(self) -> tuple[tf.keras.Model, list[str]]:
        model, modified_columns = build_transformer_model(
            total_words=self.total_words,
            max_sequence_len=self.max_sequence_len,
            embedding_output_dims=self.embedding_output_dims,
            units_per_layer=self.units_per_layer,
            dropout=self.dropout,
            column_list=self.column_list,
            num_heads=self.num_heads,
            ff_multiplier=self.ff_multiplier,
        )
        return model, modified_columns


__all__ = ["TransformerSynthesizer"]
