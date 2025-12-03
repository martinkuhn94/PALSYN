from __future__ import annotations

import os
import pickle
import random
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import tensorflow as tf
import yaml
from keras import Input, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import BatchNormalization, Dense, Dropout, Embedding

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

from PALSYN.metrics_logger import CustomProgressBar, MetricsLogger
from PALSYN.models import Encoder, get_custom_objects, get_encoder_class
from PALSYN.postprocessing.log_postprocessing import generate_df
from PALSYN.preprocessing.log_preprocessing import preprocess_event_log
from PALSYN.preprocessing.log_tokenization import tokenize_log
from PALSYN.sampling.log_sampling import sample_batch


def _load_pickle_file(file_path: str) -> Any:
    """Load a trusted pickle artifact stored alongside the trained model."""
    with open(file_path, "rb") as handle:
        return pickle.load(handle)  # noqa: S301 - local, versioned artifacts only

IntArray = npt.NDArray[np.int_]


class DPEventLogSynthesizer:
    """Differentially private sequence model for event log synthesis.

    Builds and trains a privacy-preserving sequence model, then samples
    synthetic event logs. Provides utilities for preprocessing, tokenization,
    training with callbacks, and saving/loading all artifacts.

    Args:
        embedding_output_dims: Size of the embedding vectors.
        method: Encoder type ("LSTM", "Bi-LSTM", "GRU", "Bi-GRU", "RNN", "Bi-RNN", "TCN", "LNN", "Conformer", "WaveNet", "ESN", "Transformer").
        units_per_layer: Hidden units per recurrent layer.
        epochs: Default number of training epochs.
        batch_size: Training batch size.
        max_clusters: Maximum clusters for categorical variables during preprocessing.
        dropout: Dropout rate applied before the output layers.
        trace_quantile: Quantile used to bound trace length.
        l2_norm_clip: L2 clipping norm for the DP optimizer.
        epsilon: Target privacy budget used by preprocessing to derive noise.
        learning_rate: Optimizer learning rate.
        validation_split: Fraction of training data used for validation.
        checkpoint_path: Optional path for training-time checkpoints.
        seed: Random seed for reproducibility. If None, a seed is generated.
        liquid_tau_min: Minimum learnable time constant for liquid cells.
        liquid_tau_max: Maximum learnable time constant for liquid cells.
        liquid_connectivity: Fraction of recurrent connections kept in liquid cells.
        conformer_num_heads: Attention heads per Conformer block.
        conformer_ff_multiplier: Expansion factor for Conformer feed-forward layers.
        conformer_conv_kernel_size: Kernel size for Conformer convolution modules.
        conformer_dropout: Dropout applied within Conformer blocks.
        wavenet_kernel_size: Kernel size for dilated convolutions in WaveNet blocks.
        wavenet_dilation_base: Multiplicative factor for the dilation schedule.
        wavenet_skip_channels: Number of channels used for the WaveNet skip pathway.
        esn_spectral_radius: Target spectral radius for the ESN reservoir weights.
        esn_input_scaling: Scaling applied to ESN input connections.
        esn_leak_rate: Leak rate controlling ESN state updates.
        esn_bias_scale: Range of random biases applied to ESN units.
        esn_activation: Activation function used within the ESN reservoir.
        transformer_num_heads: Attention heads per Transformer block.
        transformer_ff_multiplier: Expansion factor for Transformer feed-forward layers.
        transformer_dropout: Dropout applied within Transformer blocks.
    """

    def __init__(
        self,
        embedding_output_dims: int = 16,
        method: str = "LSTM",
        units_per_layer: list[int] | None = None,
        epochs: int = 3,
        batch_size: int = 16,
        max_clusters: int = 10,
        dropout: float = 0.0,
        trace_quantile: float = 0.95,
        l2_norm_clip: float = 1.5,
        epsilon: float | None = None,
        learning_rate: float = 0.001,
        validation_split: float = 0.1,
        checkpoint_path: str | None = None,
        seed: int | None = None,
        liquid_tau_min: float = 0.1,
        liquid_tau_max: float = 2.0,
        liquid_connectivity: float = 0.3,
        conformer_num_heads: int = 4,
        conformer_ff_multiplier: float = 4.0,
        conformer_conv_kernel_size: int = 15,
        conformer_dropout: float = 0.1,
        wavenet_kernel_size: int = 2,
        wavenet_dilation_base: int = 2,
        wavenet_skip_channels: int | None = None,
        esn_spectral_radius: float = 0.9,
        esn_input_scaling: float = 0.1,
        esn_leak_rate: float = 1.0,
        esn_bias_scale: float = 0.0,
        esn_activation: str = "tanh",
        transformer_num_heads: int = 4,
        transformer_ff_multiplier: float = 4.0,
        transformer_dropout: float = 0.1,
    ) -> None:
        self.modified_column_list: list[str] = []
        self.metrics_df: pd.DataFrame | None = None
        self.dict_dtypes: dict[str, Any] | None = None
        self.cluster_dict: dict[str, Any] | None = None
        self.event_log_sentences: list[list[str]] = []
        self.max_clusters = max_clusters
        self.trace_quantile = trace_quantile

        self.model: Model | None = None
        self.max_sequence_len: int | None = None
        self.total_words: int = 0
        self.tokenizer: Any = None
        self.ys: IntArray | None = None
        self.xs: IntArray | None = None
        self.start_epoch: list[float] = []
        self.num_cols: int = 0
        self.column_list: list[str] = []

        self.units_per_layer = list(units_per_layer) if units_per_layer else [64, 64]
        if not isinstance(self.units_per_layer, list) or not all(
            isinstance(u, int) and u > 0 for u in self.units_per_layer
        ):
            raise ValueError("units_per_layer must be a list of positive ints")

        allowed_methods = {
            "LSTM",
            "Bi-LSTM",
            "GRU",
            "Bi-GRU",
            "RNN",
            "Bi-RNN",
            "TCN",
            "LNN",
            "Conformer",
            "WaveNet",
            "ESN",
            "Transformer",
        }
        if method not in allowed_methods:
            raise ValueError(f"method must be one of {sorted(allowed_methods)}")
        self.method = method
        self.embedding_output_dims = embedding_output_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.checkpoint_path = checkpoint_path
        self.conformer_num_heads = max(1, int(conformer_num_heads))
        self.conformer_ff_multiplier = max(float(conformer_ff_multiplier), 1.0)
        self.conformer_conv_kernel_size = max(1, int(conformer_conv_kernel_size))
        self.conformer_dropout = float(np.clip(conformer_dropout, 0.0, 1.0))
        self.wavenet_kernel_size = max(1, int(wavenet_kernel_size))
        self.wavenet_dilation_base = max(1, int(wavenet_dilation_base))
        self.wavenet_skip_channels = (
            int(wavenet_skip_channels) if wavenet_skip_channels is not None else None
        )
        self.esn_spectral_radius = float(esn_spectral_radius)
        self.esn_input_scaling = float(esn_input_scaling)
        self.esn_leak_rate = float(esn_leak_rate)
        self.esn_bias_scale = float(esn_bias_scale)
        self.esn_activation = str(esn_activation)
        self.transformer_num_heads = max(1, int(transformer_num_heads))
        self.transformer_ff_multiplier = float(max(transformer_ff_multiplier, 1.0))
        self.transformer_dropout = float(np.clip(transformer_dropout, 0.0, 1.0))

        self.liquid_tau_min = max(float(liquid_tau_min), 1e-3)
        self.liquid_tau_max = max(float(liquid_tau_max), self.liquid_tau_min + 1e-3)
        self.liquid_connectivity = float(np.clip(liquid_connectivity, 0.0, 1.0))

        if seed is None:
            try:
                seed = random.SystemRandom().randint(0, 2**31 - 1)
            except Exception:
                seed = int.from_bytes(os.urandom(4), "little")
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        self.noise_multiplier: float = 0.0
        self.epsilon = epsilon
        self.l2_norm_clip = l2_norm_clip
        self.num_examples: int = 0

    def initialize_model(self, input_data: pd.DataFrame) -> None:  # noqa: C901 - orchestration heavy
        """Prepare data, build the network, and compile with a DP optimizer.

        Args:
            input_data: Raw event log DataFrame to preprocess and tokenize.
        """
        (
            self.event_log_sentences,
            self.cluster_dict,
            self.dict_dtypes,
            self.start_epoch,
            self.num_examples,
            self.noise_multiplier,
            self.num_cols,
            self.column_list,
        ) = preprocess_event_log(
            input_data,
            self.max_clusters,
            self.trace_quantile,
            self.epsilon,
            self.batch_size,
            self.epochs,
        )

        (self.xs, self.ys, self.total_words, self.max_sequence_len, self.tokenizer) = tokenize_log(
            self.event_log_sentences, steps=self.num_cols
        )

        inputs = Input(shape=(self.max_sequence_len,), dtype="int32")
        embedding_layer = Embedding(
            self.total_words,
            self.embedding_output_dims,
            input_length=self.max_sequence_len,
            embeddings_regularizer=tf.keras.regularizers.l2(1e-5),
            mask_zero=True,
        )(inputs)
        encoder = self._create_encoder()
        x = encoder.build(embedding_layer)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout)(x)

        outputs = []
        self.modified_column_list = []
        for column in self.column_list:
            self.modified_column_list.append(column.replace(":", "_").replace(" ", "_"))

        for step in range(self.num_cols):
            output = Dense(
                self.total_words, activation="softmax", name=f"{self.modified_column_list[step]}"
            )(x)
            outputs.append(output)

        self.model = Model(inputs=inputs, outputs=outputs)
        optimizer = self._build_optimizer()
        self.model.compile(
            loss=["sparse_categorical_crossentropy"] * self.num_cols,
            optimizer=optimizer,
            metrics=["accuracy"],
        )

    def _build_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """Return a DP optimizer only when noise is required."""
        if self.noise_multiplier is not None and self.noise_multiplier > 0:
            return DPKerasAdamOptimizer(
                l2_norm_clip=self.l2_norm_clip,
                noise_multiplier=self.noise_multiplier,
                num_microbatches=1,
                learning_rate=self.learning_rate,
        )
        return tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _create_encoder(self) -> Encoder:
        """Instantiate the configured encoder class with the appropriate hyperparameters."""
        encoder_class = get_encoder_class(self.method)
        if self.method == "TCN":
            kwargs: dict[str, Any] = {"filters_per_layer": self.units_per_layer}
        elif self.method == "Conformer":
            kwargs = {
                "units_per_layer": self.units_per_layer,
                "num_heads": self.conformer_num_heads,
                "ff_multiplier": self.conformer_ff_multiplier,
                "conv_kernel_size": self.conformer_conv_kernel_size,
                "dropout": self.conformer_dropout,
            }
        elif self.method == "WaveNet":
            kwargs = {
                "units_per_layer": self.units_per_layer,
                "kernel_size": self.wavenet_kernel_size,
                "dilation_base": self.wavenet_dilation_base,
                "skip_channels": self.wavenet_skip_channels,
            }
        elif self.method == "ESN":
            kwargs = {
                "units_per_layer": self.units_per_layer,
                "spectral_radius": self.esn_spectral_radius,
                "input_scaling": self.esn_input_scaling,
                "leak_rate": self.esn_leak_rate,
                "bias_scale": self.esn_bias_scale,
                "activation": self.esn_activation,
                "seed": self.seed,
            }
        elif self.method == "Transformer":
            kwargs = {
                "units_per_layer": self.units_per_layer,
                "num_heads": self.transformer_num_heads,
                "ff_multiplier": self.transformer_ff_multiplier,
                "dropout": self.transformer_dropout,
            }
        else:
            kwargs = {"units_per_layer": self.units_per_layer}

        if self.method == "LNN":
            kwargs.update(
                tau_min=self.liquid_tau_min,
                tau_max=self.liquid_tau_max,
                connectivity=self.liquid_connectivity,
            )

        return encoder_class(**kwargs)

    def train(self, epochs: int | None = None) -> None:
        """Train the model with early stopping, metrics logging, and optional checkpoints.

        Args:
            epochs: Number of epochs. Defaults to the value set at initialization.
        """
        if self.model is None or self.xs is None or self.ys is None:
            raise RuntimeError("Model must be initialized before training.")

        y_outputs = [self.ys[:, step] for step in range(self.num_cols)]

        monitor_metric = f"val_{self.modified_column_list[0]}_accuracy"
        early_stopping = EarlyStopping(
            monitor=monitor_metric,
            mode="max",
            verbose=0,
            patience=7,
            restore_best_weights=True,
            min_delta=0.001,
            baseline=None,
            start_from_epoch=5,
        )

        metrics_logger = MetricsLogger(num_cols=self.num_cols, column_list=self.column_list)
        custom_progress_bar = CustomProgressBar()
        callbacks = [early_stopping, metrics_logger, custom_progress_bar]
        if self.checkpoint_path:
            callbacks.append(
                ModelCheckpoint(
                    filepath=self.checkpoint_path,
                    monitor=monitor_metric,
                    mode="max",
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=0,
                )
            )

        self.model.fit(
            self.xs,
            y_outputs,
            epochs=epochs or self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            validation_split=self.validation_split,
            verbose=0,
        )

        self.metrics_df = metrics_logger.get_dataframe()

    def fit(self, input_data: pd.DataFrame) -> None:
        """Initialize the model and run training on the provided data.

        Args:
            input_data: Event log DataFrame.
        """
        self.initialize_model(input_data)
        self.train(self.epochs)

    def sample(self, sample_size: int, batch_size: int | None = None) -> pd.DataFrame:
        """Generate a synthetic event log using the trained model.

        Args:
            sample_size: Number of traces to sample.
            batch_size: Optional batch size for sampling.

        Returns:
            A DataFrame containing the sampled event log.
        """
        if (
            self.model is None
            or self.tokenizer is None
            or self.max_sequence_len is None
            or self.cluster_dict is None
            or self.dict_dtypes is None
            or not self.start_epoch
            or not self.column_list
        ):
            raise RuntimeError("Model must be trained or loaded before sampling.")

        len_synthetic_event_log = 0
        synthetic_df = pd.DataFrame()
        batch = batch_size or self.batch_size

        while len_synthetic_event_log < sample_size:
            print("Sampling Event Log with:", sample_size - len_synthetic_event_log, "traces left")
            sample_size_new = sample_size - len_synthetic_event_log

            synthetic_event_log_sentences = sample_batch(
                sample_size_new,
                self.tokenizer,
                self.max_sequence_len,
                self.model,
                batch,
                self.num_cols,
                self.column_list,
            )

            df = generate_df(
                synthetic_event_log_sentences, self.cluster_dict, self.dict_dtypes, self.start_epoch
            )
            df.reset_index(drop=True, inplace=True)
            synthetic_df = pd.concat([synthetic_df, df], axis=0, ignore_index=True)
            new_cases = df["case:concept:name"].nunique()
            if new_cases == 0:
                print("Sampling produced 0 new cases; stopping to avoid infinite loop.")
                break
            len_synthetic_event_log += new_cases

        return synthetic_df

    def save_model(self, path: str) -> None:
        """Persist the model, checkpoints, metrics, and preprocessing artifacts.

        Args:
            path: Destination directory.
        """
        if self.model is None:
            raise RuntimeError("Train or load a model before saving.")
        if self.tokenizer is None or self.cluster_dict is None or self.dict_dtypes is None:
            raise RuntimeError("Tokenizer and preprocessing artifacts must be available to save.")

        os.makedirs(path, exist_ok=True)

        self.model.save(os.path.join(path, "model.keras"))
        checkpoints_dir = os.path.join(path, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        full_checkpoint_path = os.path.join(checkpoints_dir, "best.keras")
        self.model.save(full_checkpoint_path)
        if self.metrics_df is not None and not self.metrics_df.empty:
            try:
                self.metrics_df.to_excel(os.path.join(path, "training_metrics.xlsx"), index=False)
            except Exception:
                self.metrics_df.to_csv(os.path.join(path, "training_metrics.csv"), index=False)

        config = {
            "embedding_output_dims": self.embedding_output_dims,
            "method": self.method,
            "units_per_layer": self.units_per_layer,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "max_clusters": self.max_clusters,
            "dropout": self.dropout,
            "trace_quantile": self.trace_quantile,
            "l2_norm_clip": self.l2_norm_clip,
            "epsilon": self.epsilon,
            "noise_multiplier": self.noise_multiplier,
            "num_examples": self.num_examples,
            "learning_rate": self.learning_rate,
            "validation_split": self.validation_split,
            "checkpoint_path": os.path.join("checkpoints", "best.keras"),
            "seed": self.seed,
            "conformer_num_heads": self.conformer_num_heads,
            "conformer_ff_multiplier": self.conformer_ff_multiplier,
            "conformer_conv_kernel_size": self.conformer_conv_kernel_size,
            "conformer_dropout": self.conformer_dropout,
            "wavenet_kernel_size": self.wavenet_kernel_size,
            "wavenet_dilation_base": self.wavenet_dilation_base,
            "wavenet_skip_channels": self.wavenet_skip_channels,
            "esn_spectral_radius": self.esn_spectral_radius,
            "esn_input_scaling": self.esn_input_scaling,
            "esn_leak_rate": self.esn_leak_rate,
            "esn_bias_scale": self.esn_bias_scale,
            "esn_activation": self.esn_activation,
            "transformer_num_heads": self.transformer_num_heads,
            "transformer_ff_multiplier": self.transformer_ff_multiplier,
            "transformer_dropout": self.transformer_dropout,
        }

        with open(os.path.join(path, "model_config.yaml"), "w", encoding="utf-8") as handle:
            yaml.dump(config, handle, default_flow_style=False)

        with open(os.path.join(path, "tokenizer.pkl"), "wb") as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, "cluster_dict.pkl"), "wb") as handle:
            pickle.dump(self.cluster_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, "dict_dtypes.yaml"), "w", encoding="utf-8") as handle:
            yaml.dump(self.dict_dtypes, handle, default_flow_style=False)

        with open(os.path.join(path, "max_sequence_len.pkl"), "wb") as handle:
            pickle.dump(self.max_sequence_len, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, "start_epoch.pkl"), "wb") as handle:
            pickle.dump(self.start_epoch, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, "num_cols.pkl"), "wb") as handle:
            pickle.dump(self.num_cols, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, "column_list.pkl"), "wb") as handle:
            pickle.dump(self.column_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> None:
        """Load a saved model and all required artifacts from a directory.

        Args:
            path: Directory containing the saved model and artifacts.
        """
        self.model = tf.keras.models.load_model(
            os.path.join(path, "model.keras"),
            compile=False,
            safe_mode=False,  # trusted, locally produced models may include Lambda layers
            custom_objects=get_custom_objects(),
        )

        self.tokenizer = _load_pickle_file(os.path.join(path, "tokenizer.pkl"))
        self.cluster_dict = _load_pickle_file(os.path.join(path, "cluster_dict.pkl"))

        with open(os.path.join(path, "dict_dtypes.yaml"), encoding="utf-8") as handle:
            self.dict_dtypes = yaml.safe_load(handle)

        self.max_sequence_len = _load_pickle_file(os.path.join(path, "max_sequence_len.pkl"))
        self.start_epoch = _load_pickle_file(os.path.join(path, "start_epoch.pkl"))
        self.num_cols = _load_pickle_file(os.path.join(path, "num_cols.pkl"))
        self.column_list = _load_pickle_file(os.path.join(path, "column_list.pkl"))

        config_path = os.path.join(path, "model_config.yaml")
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as handle:
                cfg = yaml.safe_load(handle) or {}
            self.embedding_output_dims = cfg.get(
                "embedding_output_dims", self.embedding_output_dims
            )
            self.method = cfg.get("method", self.method)
            self.units_per_layer = cfg.get("units_per_layer", self.units_per_layer)
            self.epochs = cfg.get("epochs", self.epochs)
            self.batch_size = cfg.get("batch_size", self.batch_size)
            self.max_clusters = cfg.get("max_clusters", self.max_clusters)
            self.dropout = cfg.get("dropout", self.dropout)
            self.trace_quantile = cfg.get("trace_quantile", self.trace_quantile)
            self.l2_norm_clip = cfg.get("l2_norm_clip", self.l2_norm_clip)
            self.epsilon = cfg.get("epsilon", self.epsilon)
            self.noise_multiplier = cfg.get("noise_multiplier", self.noise_multiplier)
            self.num_examples = cfg.get("num_examples", self.num_examples)
            self.learning_rate = cfg.get("learning_rate", self.learning_rate)
            self.validation_split = cfg.get("validation_split", self.validation_split)
            self.checkpoint_path = cfg.get("checkpoint_path", self.checkpoint_path)
            self.seed = cfg.get("seed", self.seed)
            self.conformer_num_heads = cfg.get("conformer_num_heads", self.conformer_num_heads)
            self.conformer_ff_multiplier = cfg.get("conformer_ff_multiplier", self.conformer_ff_multiplier)
            self.conformer_conv_kernel_size = cfg.get(
                "conformer_conv_kernel_size", self.conformer_conv_kernel_size
            )
            self.conformer_dropout = cfg.get("conformer_dropout", self.conformer_dropout)
            self.wavenet_kernel_size = cfg.get("wavenet_kernel_size", self.wavenet_kernel_size)
            self.wavenet_dilation_base = cfg.get(
                "wavenet_dilation_base", self.wavenet_dilation_base
            )
            self.wavenet_skip_channels = cfg.get(
                "wavenet_skip_channels", self.wavenet_skip_channels
            )
            self.esn_spectral_radius = cfg.get("esn_spectral_radius", self.esn_spectral_radius)
            self.esn_input_scaling = cfg.get("esn_input_scaling", self.esn_input_scaling)
            self.esn_leak_rate = cfg.get("esn_leak_rate", self.esn_leak_rate)
            self.esn_bias_scale = cfg.get("esn_bias_scale", self.esn_bias_scale)
            self.esn_activation = cfg.get("esn_activation", self.esn_activation)
            self.transformer_num_heads = cfg.get("transformer_num_heads", self.transformer_num_heads)
            self.transformer_ff_multiplier = cfg.get("transformer_ff_multiplier", self.transformer_ff_multiplier)
            self.transformer_dropout = cfg.get("transformer_dropout", self.transformer_dropout)

        self.modified_column_list = [
            c.replace(":", "_").replace(" ", "_") for c in (self.column_list or [])
        ]
