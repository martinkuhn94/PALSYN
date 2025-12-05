from __future__ import annotations

"""Optimizer helpers for PALSYN synthesizers."""

from typing import Any

import tensorflow as tf

try:
    from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import (
        DPKerasAdamOptimizer,
    )
except Exception:
    try:
        from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras_vectorized import (
            DPKerasAdamOptimizer,
        )
    except Exception:
        try:
            from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import (
                DPKerasAdamGaussianOptimizer as DPKerasAdamOptimizer,
            )
        except Exception as exc:
            raise ImportError(
                "Unable to import a DP Keras Adam optimizer from tensorflow_privacy."
            ) from exc


def build_optimizer(
    *,
    learning_rate: float,
    l2_norm_clip: float,
    noise_multiplier: float | None,
) -> tf.keras.optimizers.Optimizer:
    """Return either a DP or vanilla Adam optimizer depending on noise settings."""
    if noise_multiplier is not None and noise_multiplier > 0:
        return DPKerasAdamOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=1,
            learning_rate=learning_rate,
        )
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


__all__ = ["build_optimizer"]
