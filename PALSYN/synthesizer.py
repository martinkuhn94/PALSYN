import os
import pickle
import yaml
import random
from typing import List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    GRU,
    LSTM,
    Lambda,
    RNN,
    SimpleRNN,
)

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

from PALSYN.metrics_logger import MetricsLogger, CustomProgressBar
from PALSYN.preprocessing.log_preprocessing import preprocess_event_log
from PALSYN.preprocessing.log_tokenization import tokenize_log
from PALSYN.sampling.log_sampling import sample_batch
from PALSYN.postprocessing.log_postprocessing import generate_df


@tf.keras.utils.register_keras_serializable(package="palsyn")
class LiquidTimeConstantCell(tf.keras.layers.AbstractRNNCell):
    """Simple liquid neural network cell with learnable time constants."""

    def __init__(
            self,
            units: int,
            tau_min: float = 0.1,
            tau_max: float = 2.0,
            connectivity: float = 0.3,
            activation: str = "tanh",
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.units = int(units)
        self.tau_min = max(float(tau_min), 1e-3)
        self.tau_max = max(float(tau_max), self.tau_min + 1e-3)
        self.connectivity = float(np.clip(connectivity, 0.0, 1.0))
        self.activation_fn = tf.keras.activations.get(activation)
        self._recurrent_mask = None

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    def build(self, input_shape) -> None:
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

    def call(self, inputs, states):
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


class DPEventLogSynthesizer:
    """Differentially private sequence model for event log synthesis.

    Builds and trains a privacy-preserving sequence model, then samples
    synthetic event logs. Provides utilities for preprocessing, tokenization,
    training with callbacks, and saving/loading all artifacts.

    Args:
        embedding_output_dims: Size of the embedding vectors.
        method: Encoder type ("LSTM", "Bi-LSTM", "GRU", "Bi-GRU", "RNN", "Bi-RNN", "TCN", "LNN").
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
    """

    def __init__(
            self,
            embedding_output_dims: int = 16,
            method: str = "LSTM",
            units_per_layer: Optional[List[int]] = None,
            epochs: int = 3,
            batch_size: int = 16,
            max_clusters: int = 10,
            dropout: float = 0.0,
            trace_quantile: float = 0.95,
            l2_norm_clip: float = 1.5,
            epsilon: Optional[float] = None,
            learning_rate: float = 0.001,
            validation_split: float = 0.1,
            checkpoint_path: Optional[str] = None,
            seed: Optional[int] = None,
            liquid_tau_min: float = 0.1,
            liquid_tau_max: float = 2.0,
            liquid_connectivity: float = 0.3,
    ) -> None:

        self.modified_column_list = None
        self.metrics_df = None
        self.dict_dtypes = None
        self.cluster_dict = None
        self.event_log_sentences = None
        self.max_clusters = max_clusters
        self.trace_quantile = trace_quantile

        self.model = None
        self.max_sequence_len = None
        self.total_words = None
        self.tokenizer = None
        self.ys = None
        self.xs = None
        self.start_epoch = None
        self.num_cols = None
        self.column_list = None

        self.units_per_layer = units_per_layer or [64, 64]
        if not isinstance(self.units_per_layer, list) or not all(
                isinstance(u, int) and u > 0 for u in self.units_per_layer
        ):
            raise ValueError("units_per_layer must be a list of positive ints")

        allowed_methods = {"LSTM", "Bi-LSTM", "GRU", "Bi-GRU", "RNN", "Bi-RNN", "TCN", "LNN"}
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

        self.liquid_tau_min = max(float(liquid_tau_min), 1e-3)
        self.liquid_tau_max = max(float(liquid_tau_max), self.liquid_tau_min + 1e-3)
        self.liquid_connectivity = float(np.clip(liquid_connectivity, 0.0, 1.0))

        if seed is None:
            try:
                seed = random.SystemRandom().randint(0, 2**31 - 1)
            except Exception:
                seed = int.from_bytes(os.urandom(4), 'little')
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        self.noise_multiplier = None
        self.epsilon = epsilon
        self.l2_norm_clip = l2_norm_clip
        self.num_examples = None

    def initialize_model(self, input_data: pd.DataFrame) -> None:
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
            self.column_list
        ) = preprocess_event_log(
            input_data, self.max_clusters, self.trace_quantile, self.epsilon, self.batch_size, self.epochs
        )

        (self.xs, self.ys, self.total_words, self.max_sequence_len, self.tokenizer) = tokenize_log(
            self.event_log_sentences, steps=self.num_cols
        )

        inputs = Input(shape=(self.max_sequence_len,), dtype='int32')
        embedding_layer = Embedding(
            self.total_words,
            self.embedding_output_dims,
            input_length=self.max_sequence_len,
            embeddings_regularizer=tf.keras.regularizers.l2(1e-5),
            mask_zero=True,
        )(inputs)
        x = embedding_layer

        # Use last-state pooling: final encoder returns a single timestep embedding
        if self.method == "TCN":
            x = self._build_tcn_encoder(x)
        elif self.method == "LNN":
            x = self._build_liquid_encoder(x)
        else:
            for idx, units in enumerate(self.units_per_layer):
                is_last = idx == (len(self.units_per_layer) - 1)
                return_seq = not is_last
                if self.method == "LSTM":
                    x = LSTM(units, return_sequences=return_seq)(x)
                elif self.method == "Bi-LSTM":
                    x = Bidirectional(LSTM(units, return_sequences=return_seq))(x)
                elif self.method == "GRU":
                    x = GRU(units, return_sequences=return_seq)(x)
                elif self.method == "Bi-GRU":
                    x = Bidirectional(GRU(units, return_sequences=return_seq))(x)
                elif self.method == "RNN":
                    x = SimpleRNN(units, return_sequences=return_seq)(x)
                elif self.method == "Bi-RNN":
                    x = Bidirectional(SimpleRNN(units, return_sequences=return_seq))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout)(x)

        outputs = []
        self.modified_column_list = []
        for column in self.column_list:
            self.modified_column_list.append(column.replace(":", "_").replace(" ", "_"))

        for step in range(self.num_cols):
            output = Dense(self.total_words, activation="softmax", name=f"{self.modified_column_list[step]}")(x)
            outputs.append(output)

        self.model = Model(inputs=inputs, outputs=outputs)
        optimizer = self._build_optimizer()
        self.model.compile(
            loss=["sparse_categorical_crossentropy"] * self.num_cols,
            optimizer=optimizer,
            metrics=["accuracy"],
        )

    def _build_optimizer(self):
        """Return a DP optimizer only when noise is required."""
        if self.noise_multiplier is not None and self.noise_multiplier > 0:
            return DPKerasAdamOptimizer(
                l2_norm_clip=self.l2_norm_clip,
                noise_multiplier=self.noise_multiplier,
                num_microbatches=1,
                learning_rate=self.learning_rate,
            )
        return tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _build_liquid_encoder(self, inputs: tf.Tensor) -> tf.Tensor:
        """Stacked liquid neural network encoder."""
        x = inputs
        for idx, units in enumerate(self.units_per_layer):
            return_seq = idx < (len(self.units_per_layer) - 1)
            cell = LiquidTimeConstantCell(
                units=units,
                tau_min=self.liquid_tau_min,
                tau_max=self.liquid_tau_max,
                connectivity=self.liquid_connectivity,
                name=f"liquid_cell_{idx}",
            )
            x = RNN(cell, return_sequences=return_seq, name=f"liquid_layer_{idx}")(x)
        return x

    def _build_tcn_encoder(self, inputs: tf.Tensor) -> tf.Tensor:
        """Temporal convolutional encoder that returns the final timestep embedding."""
        x = inputs
        for idx, filters in enumerate(self.units_per_layer):
            dilation = 2 ** idx
            x = Conv1D(
                filters=filters,
                kernel_size=3,
                padding="causal",
                dilation_rate=dilation,
                activation="relu",
            )(x)
        return Lambda(lambda t: t[:, -1, :], name="tcn_last_state")(x)

    def train(self, epochs: Optional[int] = None) -> None:
        """Train the model with early stopping, metrics logging, and optional checkpoints.

        Args:
            epochs: Number of epochs. Defaults to the value set at initialization.
        """
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

    def sample(self, sample_size: int, batch_size: Optional[int] = None) -> pd.DataFrame:
        """Generate a synthetic event log using the trained model.

        Args:
            sample_size: Number of traces to sample.
            batch_size: Optional batch size for sampling.

        Returns:
            A DataFrame containing the sampled event log.
        """
        if self.model is None or self.tokenizer is None or self.max_sequence_len is None:
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
                self.column_list
            )

            df = generate_df(synthetic_event_log_sentences, self.cluster_dict, self.dict_dtypes, self.start_epoch)
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
            'embedding_output_dims': self.embedding_output_dims,
            'method': self.method,
            'units_per_layer': self.units_per_layer,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'max_clusters': self.max_clusters,
            'dropout': self.dropout,
            'trace_quantile': self.trace_quantile,
            'l2_norm_clip': self.l2_norm_clip,
            'epsilon': self.epsilon,
            'noise_multiplier': self.noise_multiplier,
            'num_examples': self.num_examples,
            'learning_rate': self.learning_rate,
            'validation_split': self.validation_split,
            'checkpoint_path': os.path.join('checkpoints', 'best.keras'),
            'seed': self.seed,
        }

        with open(os.path.join(path, "model_config.yaml"), "w", encoding='utf-8') as handle:
            yaml.dump(config, handle, default_flow_style=False)

        with open(os.path.join(path, "tokenizer.pkl"), "wb") as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, "cluster_dict.pkl"), "wb") as handle:
            pickle.dump(self.cluster_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, "dict_dtypes.yaml"), "w", encoding='utf-8') as handle:
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
        self.model = tf.keras.models.load_model(os.path.join(path, "model.keras"), compile=False)

        with open(os.path.join(path, "tokenizer.pkl"), "rb") as handle:
            self.tokenizer = pickle.load(handle)

        with open(os.path.join(path, "cluster_dict.pkl"), "rb") as handle:
            self.cluster_dict = pickle.load(handle)

        with open(os.path.join(path, "dict_dtypes.yaml"), "r", encoding='utf-8') as handle:
            self.dict_dtypes = yaml.safe_load(handle)

        with open(os.path.join(path, "max_sequence_len.pkl"), "rb") as handle:
            self.max_sequence_len = pickle.load(handle)

        with open(os.path.join(path, "start_epoch.pkl"), "rb") as handle:
            self.start_epoch = pickle.load(handle)

        with open(os.path.join(path, "num_cols.pkl"), "rb") as handle:
            self.num_cols = pickle.load(handle)

        with open(os.path.join(path, "column_list.pkl"), "rb") as handle:
            self.column_list = pickle.load(handle)

        config_path = os.path.join(path, "model_config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding='utf-8') as handle:
                cfg = yaml.safe_load(handle) or {}
            self.embedding_output_dims = cfg.get('embedding_output_dims', self.embedding_output_dims)
            self.method = cfg.get('method', self.method)
            self.units_per_layer = cfg.get('units_per_layer', self.units_per_layer)
            self.epochs = cfg.get('epochs', self.epochs)
            self.batch_size = cfg.get('batch_size', self.batch_size)
            self.max_clusters = cfg.get('max_clusters', self.max_clusters)
            self.dropout = cfg.get('dropout', self.dropout)
            self.trace_quantile = cfg.get('trace_quantile', self.trace_quantile)
            self.l2_norm_clip = cfg.get('l2_norm_clip', self.l2_norm_clip)
            self.epsilon = cfg.get('epsilon', self.epsilon)
            self.noise_multiplier = cfg.get('noise_multiplier', self.noise_multiplier)
            self.num_examples = cfg.get('num_examples', self.num_examples)
            self.learning_rate = cfg.get('learning_rate', self.learning_rate)
            self.validation_split = cfg.get('validation_split', self.validation_split)
            self.checkpoint_path = cfg.get('checkpoint_path', self.checkpoint_path)
            self.seed = cfg.get('seed', self.seed)

        self.modified_column_list = [c.replace(":", "_").replace(" ", "_") for c in (self.column_list or [])]
