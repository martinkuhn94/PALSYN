from __future__ import annotations

import os
import pickle
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
import tensorflow as tf
import yaml
from keras.callbacks import EarlyStopping, ModelCheckpoint

from PALSYN.metrics_logger import CustomProgressBar, MetricsLogger
from PALSYN.models import get_custom_objects
from PALSYN.models.base import sanitize_column_names
from PALSYN.optimizers import build_optimizer
from PALSYN.postprocessing.log_postprocessing import generate_df
from PALSYN.preprocessing.data_preparation_pipeline import DataPreparationPipeline
from PALSYN.sampling.log_sampling import sample_batch


def _initialize_seed(seed: int | None) -> int:
    """Derive a reproducible seed and push it to Python/NumPy/TensorFlow RNGs."""
    if seed is None:
        try:
            seed = random.SystemRandom().randint(0, 2**31 - 1)
        except Exception:
            seed = int.from_bytes(os.urandom(4), "little")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return seed


@dataclass
class TrainingConfig:
    epochs: int = 0
    batch_size: int = 0
    validation_split: float = 0.0
    checkpoint_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "validation_split": self.validation_split,
            "checkpoint_path": self.checkpoint_path,
        }


@dataclass
class OptimizerConfig:
    learning_rate: float = 0.0
    l2_norm_clip: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "l2_norm_clip": self.l2_norm_clip,
        }


@dataclass
class PrivacyState:
    epsilon: float | None = None
    noise_multiplier: float = 0.0
    num_examples: int = 0

    def to_runtime_dict(self) -> dict[str, Any]:
        return {
            "noise_multiplier": self.noise_multiplier,
            "num_examples": self.num_examples,
        }

    def apply_runtime(self, metadata: dict[str, Any]) -> None:
        if "noise_multiplier" in metadata and metadata["noise_multiplier"] is not None:
            self.noise_multiplier = float(metadata["noise_multiplier"])
        if "num_examples" in metadata and metadata["num_examples"] is not None:
            self.num_examples = int(metadata["num_examples"])


IntArray = npt.NDArray[np.int_]
SynthT = TypeVar("SynthT", bound="BaseSynthesizer")


class BaseSynthesizer(ABC):
    MODEL_TYPE: ClassVar[str] = "BASE"

    def __init__(
        self,
        *,
        max_clusters: int = 10,
        trace_quantile: float = 0.95,
        epsilon: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.modified_column_list: list[str] = []
        self.metrics_df: pd.DataFrame | None = None
        self.dict_dtypes: dict[str, Any] | None = None
        self.cluster_dict: dict[str, Any] | None = None
        self.event_log_sentences: list[list[str]] = []

        self.max_clusters = max_clusters
        self.trace_quantile = trace_quantile

        self.model: tf.keras.Model | None = None
        self.max_sequence_len: int | None = None
        self.total_words: int = 0
        self.tokenizer: Any = None
        self.ys: IntArray | None = None
        self.xs: IntArray | None = None
        self.start_epoch: list[float] = []
        self.num_cols: int = 0
        self.column_list: list[str] = []

        self.privacy_state = PrivacyState(
            epsilon=epsilon,
            noise_multiplier=0.0,
            num_examples=0,
        )

        self.seed = _initialize_seed(seed)

        self.training_config = TrainingConfig()
        self.optimizer_config = OptimizerConfig()
        self._training_params_set = False
        self._optimizer_params_set = False
        self._training_configured = False

    # ------------------------------------------------------------------ #
    # Abstract/model-specific helpers
    # ------------------------------------------------------------------ #
    def _configure_training(
        self,
        *,
        epochs: int,
        batch_size: int,
        validation_split: float,
        checkpoint_path: str | None,
    ) -> None:
        self.training_config = TrainingConfig(
            epochs=int(epochs),
            batch_size=int(batch_size),
            validation_split=float(validation_split),
            checkpoint_path=checkpoint_path,
        )
        self._training_params_set = True
        self._refresh_training_ready()

    def _configure_optimizer(self, *, learning_rate: float, l2_norm_clip: float) -> None:
        self.optimizer_config = OptimizerConfig(
            learning_rate=float(learning_rate),
            l2_norm_clip=float(l2_norm_clip),
        )
        self._optimizer_params_set = True
        self._refresh_training_ready()

    def _refresh_training_ready(self) -> None:
        self._training_configured = self._training_params_set and self._optimizer_params_set

    def _ensure_training_ready(self, action: str) -> None:
        if not self._training_configured:
            raise RuntimeError(f"Configure training hyperparameters before {action}.")

    def _store_preprocessing_outputs(self, prepared: dict[str, Any]) -> None:
        self.event_log_sentences = prepared["event_log_sentences"]
        self.cluster_dict = prepared["cluster_dict"]
        self.dict_dtypes = prepared["attribute_dtypes"]
        self.start_epoch = prepared["start_epoch_stats"]
        self.privacy_state.num_examples = prepared["num_examples"]
        self.privacy_state.noise_multiplier = prepared.get("noise_multiplier", 0.0)
        self.num_cols = prepared["num_cols"]
        self.column_list = prepared["column_list"]
        self.xs = prepared["xs"]
        self.ys = prepared["ys"]
        self.total_words = prepared["total_words"]
        self.max_sequence_len = prepared["max_sequence_len"]
        self.tokenizer = prepared["tokenizer"]
        self.model = None
        self.modified_column_list = []

    def _validate_preprocessed_inputs(self) -> None:
        if self.xs is None or self.ys is None:
            raise RuntimeError("Preprocess data before building the model.")
        if self.max_sequence_len is None or self.total_words <= 0:
            raise RuntimeError("Preprocessing did not produce valid sequence data.")
        if not self.column_list or self.num_cols <= 0:
            raise RuntimeError("Preprocessing did not provide column metadata.")

    def _get_base_init_args(self) -> dict[str, Any]:
        return {
            "max_clusters": self.max_clusters,
            "trace_quantile": self.trace_quantile,
            "epsilon": self.privacy_state.epsilon,
            "seed": self.seed,
        }

    def _get_training_init_args(self) -> dict[str, Any]:
        return self.training_config.to_dict() if self._training_params_set else {}

    def _get_optimizer_init_args(self) -> dict[str, Any]:
        return self.optimizer_config.to_dict() if self._optimizer_params_set else {}

    def _get_model_specific_init_args(self) -> dict[str, Any]:
        return {}

    def _get_runtime_state(self) -> dict[str, Any]:
        return self.privacy_state.to_runtime_dict()

    def _apply_runtime_state(self, metadata: dict[str, Any]) -> None:
        self.privacy_state.apply_runtime(metadata)

    def _get_init_args(self) -> dict[str, Any]:
        args = self._get_base_init_args()
        args.update(self._get_training_init_args())
        args.update(self._get_optimizer_init_args())
        args.update(self._get_model_specific_init_args())
        return args

    @abstractmethod
    def _build_model_impl(self) -> tuple[tf.keras.Model, list[str]]:
        ...

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def preprocess_data(self, input_data: pd.DataFrame) -> None:
        self._ensure_training_ready("preprocessing")

        pipeline = DataPreparationPipeline(
            max_clusters=self.max_clusters,
            trace_quantile=self.trace_quantile,
        )
        prepared = pipeline.run(
            input_data,
            epsilon=self.privacy_state.epsilon,
            batch_size=self.training_config.batch_size,
            epochs=self.training_config.epochs,
        )

        self._store_preprocessing_outputs(prepared)

    def build_model(self) -> None:
        self._ensure_training_ready("building the model")
        self._validate_preprocessed_inputs()

        self.model, self.modified_column_list = self._build_model_impl()
        optimizer = build_optimizer(
            learning_rate=self.optimizer_config.learning_rate,
            l2_norm_clip=self.optimizer_config.l2_norm_clip,
            noise_multiplier=self.privacy_state.noise_multiplier,
        )
        self.model.compile(
            loss=["sparse_categorical_crossentropy"] * self.num_cols,
            optimizer=optimizer,
            metrics=["accuracy"],
        )

    def train(self) -> None:
        if self.model is None or self.xs is None or self.ys is None:
            raise RuntimeError("Model must be initialized before training.")
        if not self.modified_column_list:
            raise RuntimeError("Build the model before training to set output heads.")

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
        if self.training_config.checkpoint_path:
            callbacks.append(
                ModelCheckpoint(
                    filepath=self.training_config.checkpoint_path,
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
            epochs=self.training_config.epochs,
            batch_size=self.training_config.batch_size,
            callbacks=callbacks,
            validation_split=self.training_config.validation_split,
            verbose=0,
        )

        self.metrics_df = metrics_logger.get_dataframe()

    def fit(self, input_data: pd.DataFrame) -> None:
        self.preprocess_data(input_data)
        self.build_model()
        self.train()

    def sample(self, sample_size: int, batch_size: int | None = None) -> pd.DataFrame:
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
        batch = batch_size or self.training_config.batch_size

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

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #
    def save_model(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Train or load a model before saving.")
        if self.tokenizer is None or self.cluster_dict is None or self.dict_dtypes is None:
            raise RuntimeError("Tokenizer and preprocessing artifacts must be available to save.")

        os.makedirs(path, exist_ok=True)

        self.model.save(os.path.join(path, "model.keras"))
        checkpoints_dir = os.path.join(path, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        self.model.save(os.path.join(checkpoints_dir, "best.keras"))

        if self.metrics_df is not None and not self.metrics_df.empty:
            metrics_path = os.path.join(path, "training_metrics.xlsx")
            try:
                self.metrics_df.to_excel(metrics_path, index=False)
            except Exception:
                metrics_path = os.path.join(path, "training_metrics.csv")
                self.metrics_df.to_csv(metrics_path, index=False)

        config = {
            "model_type": self.MODEL_TYPE,
            "init_args": self._get_init_args(),
            "runtime_state": self._get_runtime_state(),
        }
        with open(os.path.join(path, "model_config.yaml"), "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle)

        with open(os.path.join(path, "tokenizer.pkl"), "wb") as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(path, "cluster_dict.pkl"), "wb") as handle:
            pickle.dump(self.cluster_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(path, "dict_dtypes.yaml"), "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.dict_dtypes, handle)
        with open(os.path.join(path, "max_sequence_len.pkl"), "wb") as handle:
            pickle.dump(self.max_sequence_len, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(path, "start_epoch.pkl"), "wb") as handle:
            pickle.dump(self.start_epoch, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(path, "num_cols.pkl"), "wb") as handle:
            pickle.dump(self.num_cols, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(path, "column_list.pkl"), "wb") as handle:
            pickle.dump(self.column_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls: type[SynthT], path: str) -> SynthT:
        config_path = os.path.join(path, "model_config.yaml")
        with open(config_path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        model_type = data.get("model_type")
        if model_type != cls.MODEL_TYPE:
            raise ValueError(
                f"Cannot load model type '{model_type}' with class '{cls.__name__}'."
            )

        init_args = data.get("init_args", {})
        model = cls(**init_args)
        model._apply_runtime_state(data.get("runtime_state", {}))
        model._load_artifacts(path)
        return model

    def _load_artifacts(self, path: str) -> None:
        self.model = tf.keras.models.load_model(
            os.path.join(path, "model.keras"),
            compile=False,
            safe_mode=False,
            custom_objects=get_custom_objects(),
        )
        with open(os.path.join(path, "tokenizer.pkl"), "rb") as handle:
            self.tokenizer = pickle.load(handle)  # noqa: S301
        with open(os.path.join(path, "cluster_dict.pkl"), "rb") as handle:
            self.cluster_dict = pickle.load(handle)  # noqa: S301
        with open(os.path.join(path, "dict_dtypes.yaml"), encoding="utf-8") as handle:
            self.dict_dtypes = yaml.safe_load(handle)
        with open(os.path.join(path, "max_sequence_len.pkl"), "rb") as handle:
            self.max_sequence_len = pickle.load(handle)  # noqa: S301
        with open(os.path.join(path, "start_epoch.pkl"), "rb") as handle:
            self.start_epoch = pickle.load(handle)  # noqa: S301
        with open(os.path.join(path, "num_cols.pkl"), "rb") as handle:
            self.num_cols = pickle.load(handle)  # noqa: S301
        with open(os.path.join(path, "column_list.pkl"), "rb") as handle:
            self.column_list = pickle.load(handle)  # noqa: S301
        self.modified_column_list = sanitize_column_names(self.column_list or [])


__all__ = ["BaseSynthesizer"]
