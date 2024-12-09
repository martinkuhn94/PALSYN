import os
import pickle
import yaml

import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import (
    BatchNormalization,
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    LSTM,
    Masking,
    GRU,
    GlobalAveragePooling1D,
    SimpleRNN,
)

from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import (
    DPKerasAdamOptimizer,
)

from PALSYN.metrics_logger import MetricsLogger, CustomProgressBar
from PALSYN.preprocessing.log_preprocessing import preprocess_event_log
from PALSYN.preprocessing.log_tokenization import tokenize_log
from PALSYN.sampling.log_sampling import sample_batch
from PALSYN.postprocessing.log_postprocessing import generate_df


class DPEventLogSynthesizer:
    """
    A class for implementing a Differentially Private Sequence model for event log synthetization. This class handles
    the initialization, training and management of a privacy-preserving sequence models.

    Parameters:
    embedding_output_dims (int): Dimension of the embedding layer output. Default is 16.
    method (str): Type of recurrent layer to use, typically "LSTM". Default is "LSTM".
    units_per_layer (list): Number of units in each LSTM layer. Default is None.
    epochs (int): Number of training epochs. Default is 3.
    batch_size (int): Size of batches for training. Default is 16.
    max_clusters (int): Maximum number of clusters for categorical variables. Default is 10.
    dropout (float): Dropout rate for regularization. Default is 0.0.
    trace_quantile (float): Quantile value for trace length calculation. Default is 0.95.
    l2_norm_clip (float): Clipping norm for differential privacy. Default is 1.5.
    epsilon (float): Privacy budget for differential privacy. Default is None.

    Returns:
    None
    """

    def __init__(
            self,
            embedding_output_dims: int = 16,
            method: str = "LSTM",
            units_per_layer: list = None,
            epochs: int = 3,
            batch_size: int = 16,
            max_clusters: int = 10,
            dropout: float = 0.0,
            trace_quantile: float = 0.95,
            l2_norm_clip: float = 1.5,
            epsilon: float = None,
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

        self.units_per_layer = units_per_layer
        self.method = method
        self.embedding_output_dims = embedding_output_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout

        self.noise_multiplier = None
        self.epsilon = epsilon
        self.l2_norm_clip = l2_norm_clip
        self.num_examples = None

    def initialize_model(self, input_data: pd.DataFrame) -> None:
        """
        Initializes and compiles the differentially private sequence model. This includes preprocessing the input data,
        tokenizing the event log, building the model architecture with the specified sequence layer type,
        and configuring the differentially private optimizer.

        Parameters:
        input_data (pd.DataFrame): Input event log data to be processed.

        Returns:
        None
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
            embeddings_regularizer=tf.keras.regularizers.l2(1e-5)  # Add regularization
        )(inputs)
        x = Masking(mask_value=0)(embedding_layer)

        for i, units in enumerate(self.units_per_layer):
            if self.method == "LSTM":
                x = LSTM(units, return_sequences=True)(x)
            elif self.method == "Bi-LSTM":
                x = Bidirectional(LSTM(units, return_sequences=True))(x)
            elif self.method == "GRU":
                x = GRU(units, return_sequences=True)(x)
            elif self.method == "Bi-GRU":
                x = Bidirectional(GRU(units, return_sequences=True))(x)
            elif self.method == "RNN":
                x = SimpleRNN(units, return_sequences=True)(x)
            elif self.method == "Bi-RNN":
                x = Bidirectional(SimpleRNN(units, return_sequences=True))(x)

        x = GlobalAveragePooling1D()(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout)(x)

        outputs = []
        self.modified_column_list = []
        for column in self.column_list:
            self.modified_column_list.append(column.replace(":", "_").replace(" ", "_"))

        for step in range(self.num_cols):
            output = Dense(self.total_words, activation="softmax", name=f"{self.modified_column_list[step]}")(x)
            outputs.append(output)

        dp_optimizer = DPKerasAdamOptimizer(
            l2_norm_clip=self.l2_norm_clip,
            noise_multiplier=self.noise_multiplier,
            num_microbatches=1,
            learning_rate=0.001,
        )

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            loss=["sparse_categorical_crossentropy"] * self.num_cols,
            optimizer=dp_optimizer,
            metrics=["accuracy"],
        )

    def train(self, epochs: int) -> None:
        """
        Trains the differentially private sequence model using the preprocessed data. Implements early stopping
        based on accuracy and custom callbacks for metrics logging and progress tracking.

        Parameters:
        epochs (int): Number of training epochs to run.

        Returns:
        None
        """
        y_outputs = [self.ys[:, step] for step in range(self.num_cols)]

        early_stopping = EarlyStopping(
            monitor=f"{self.modified_column_list[0]}_accuracy",
            mode="max",
            verbose=0,
            patience=7,
            restore_best_weights=True,
            min_delta=0.001,
            baseline=None,
            start_from_epoch=5
        )

        metrics_logger = MetricsLogger(num_cols=self.num_cols, column_list=self.column_list)
        custom_progress_bar = CustomProgressBar()

        self.model.fit(
            self.xs,
            y_outputs,
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping, metrics_logger, custom_progress_bar],
            verbose=0
        )

        self.metrics_df = metrics_logger.get_dataframe()

    def fit(self, input_data: pd.DataFrame) -> None:
        """
        Fits the differentially private sequence model by initializing the model architecture and training it
        on the provided event log data.

        Parameters:
        input_data (pd.DataFrame): Input event log data to train the model on.

        Returns:
        None
        """
        self.initialize_model(input_data)
        self.train(self.epochs)

    def sample(self, sample_size: int, batch_size: int) -> pd.DataFrame:
        """
        Sample an event log from a trained DP-Bi-LSTM Model. The model must be trained before sampling. The sampling
        process can be controlled by the temperature parameter, which controls the randomness of sampling process.
        A higher temperature results in more randomness.

        Parameters:
        sample_size (int): Number of traces to sample.
        batch_size (int): Number of traces to sample in a batch.

        Returns:
        pd.DataFrame: DataFrame containing the sampled event log.
        """
        len_synthetic_event_log = 0
        synthetic_df = pd.DataFrame()

        while len_synthetic_event_log < sample_size:
            print("Sampling Event Log with:", sample_size - len_synthetic_event_log, "traces left")
            sample_size_new = sample_size - len_synthetic_event_log

            synthetic_event_log_sentences = sample_batch(
                sample_size_new,
                self.tokenizer,
                self.max_sequence_len,
                self.model,
                batch_size,
                self.num_cols,
                self.column_list
            )

            df = generate_df(synthetic_event_log_sentences, self.cluster_dict, self.dict_dtypes, self.start_epoch)
            df.reset_index(drop=True, inplace=True)
            synthetic_df = pd.concat([synthetic_df, df], axis=0, ignore_index=True)
            len_synthetic_event_log += df["case:concept:name"].nunique()

        return synthetic_df

    def save_model(self, path: str) -> None:
        """
        Save a trained PBLES Model to a given path.

        Parameters:
        path (str): Path to save the trained PBLES Model.

        Returns:
        None
        """
        os.makedirs(path, exist_ok=True)

        self.model.save(os.path.join(path, "model.keras"))
        self.metrics_df.to_excel(os.path.join(path, "training_metrics.xlsx"), index=False)

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
            'num_examples': self.num_examples
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
        """
        Load a trained PBLES Model from a given path.

        Parameters:
        path (str): Path to the trained PBLES Model.

        Returns:
        None
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
