import os
import pickle

import pandas as pd
import tensorflow as tf
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
)
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import (
    DPKerasAdamOptimizer,
)

from PBLES.preprocessing.log_preprocessing import preprocess_event_log
from PBLES.preprocessing.log_tokenization import tokenize_log
from PBLES.sampling.log_sampling import sample_batch
from PBLES.postprocessing.log_postprocessing import generate_df


class EventLogDpLstm:
    def __init__(
            self,
            lstm_units=64,
            embedding_output_dims=16,
            epochs=3,
            batch_size=16,
            max_clusters=10,
            dropout=0.0,
            trace_quantile=0.95,
            l2_norm_clip=1.5,
            noise_multiplier=1.0,
    ):
        """
        Class of the Private Bi-LSTM Event Log Synthesizer (PBLES) approach for synthesizing event logs. The class is
        based on a DP-Bi-LSTM model and thus implements differential privacy. This synthesizer is multi-perspective.

        Parameters:
        lstm_units (int): The number of LSTM units in the hidden layer of the LSTM.
        embedding_output_dims (int): Output dimension of the embedding layer.
        epochs (int): Training epochs.
        batch_size (int): Batch size.
        max_clusters (int): Maximum number of clusters to consider.
        dropout (float): Dropout rate.
        trace_quantile (float): Quantile used to truncate trace length.
        l2_norm_clip (float): Clipping value for the L2 norm.
        noise_multiplier (float): Multiplier for the noise added for differential privacy.
        """
        # Event Log Preprocessing Information
        self.dict_dtypes = None
        self.cluster_dict = None
        self.event_log_sentences = None
        self.max_clusters = max_clusters
        self.trace_quantile = trace_quantile
        self.event_attribute_dict = None

        # Attribute Preprocessing Information
        self.model = None
        self.max_sequence_len = None
        self.total_words = None
        self.tokenizer = None
        self.ys = None
        self.xs = None
        self.start_epoch = None

        # Model Information
        self.lstm_units = lstm_units
        self.embedding_output_dims = embedding_output_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.petri_net = None

        # Privacy Information
        self.epsilon = None
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier
        self.num_examples = None

    def fit(self, input_data: pd.DataFrame) -> None:
        """
        Fit and train a DP-Bi-LSTM Model on an event log. The model can be used to
        generate synthetic event logs.

        Parameters:
        input_data (Any): Input data for training the model, typically an event log.
        """

        print("Initializing LSTM Model")

        # Convert Event Log to sentences and preprocess
        (
            self.event_log_sentences,
            self.cluster_dict,
            self.dict_dtypes,
            self.start_epoch,
            self.event_attribute_dict,
            self.num_examples
        ) = preprocess_event_log(input_data, self.max_clusters, self.trace_quantile)

        # Tokenize Attributes
        (
            self.xs,
            self.ys,
            self.total_words,
            self.max_sequence_len,
            self.tokenizer
        ) = tokenize_log(self.event_log_sentences, variant='attributes')

        print("Creating LSTM Model")

        # Input layer
        inputs = Input(shape=(self.max_sequence_len - 1,))

        # Embedding layer
        embedding_layer = Embedding(
            self.total_words,
            self.embedding_output_dims,
            input_length=self.max_sequence_len - 1
        )(inputs)

        # Masking layer
        masking_layer = Masking(mask_value=0)(embedding_layer)

        # First Bi-directional LSTM layer with return_sequences=True to pass sequences to the next LSTM layer
        lstm_layer1 = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(masking_layer)

        # Second Bi-directional LSTM layer, also with return_sequences=True
        lstm_layer2 = Bidirectional(LSTM(int(self.lstm_units / 2), return_sequences=True))(lstm_layer1)

        # Third Bi-directional LSTM layer with return_sequences=False to get the last output only
        lstm_layer3 = Bidirectional(LSTM(int(self.lstm_units / 4)))(lstm_layer2)

        # Batch Normalization and Dropout layers
        batch_normalization = BatchNormalization()(lstm_layer3)
        dropout_layer = Dropout(self.dropout)(batch_normalization)

        # Output layer
        outputs = Dense(self.total_words, activation='softmax')(dropout_layer)

        # Differentially Private Optimizer
        dp_optimizer = DPKerasAdamOptimizer(
            l2_norm_clip=self.l2_norm_clip,
            noise_multiplier=self.noise_multiplier,
            num_microbatches=1,
            learning_rate=0.001
        )

        # Compile model
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=dp_optimizer,
            metrics=['accuracy']
        )

        # Fit model
        self.model.fit(
            self.xs,
            self.ys,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[EarlyStopping(monitor='loss', verbose=1, patience=5)]
        )

        # Compute Privacy Budget
        epsilon = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy_statement(
            number_of_examples=self.num_examples,
            batch_size=self.batch_size,
            num_epochs=self.epochs,
            noise_multiplier=self.noise_multiplier,
            used_microbatching=False,
            delta=1 / (len(self.event_log_sentences) ** 1.1),
        )
        self.epsilon = epsilon

    def sample(self, sample_size: int, batch_size: int,  temperature: float = 1.0) -> pd.DataFrame:
        """
        Sample an event log from a trained DP-Bi-LSTM Model. The model must be trained before sampling. The sampling
        process can be controlled by the temperature parameter, which controls the randomness of sampling process.
        A higher temperature results in more randomness.

        Parameters:
        sample_size (int): Number of traces to sample.
        temperature (float): Temperature for sampling the attribute information (default is 1.0).

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
                temperature,
                batch_size
            )

            # Generate Event Log DataFrame
            df = generate_df(
                synthetic_event_log_sentences,
                self.cluster_dict,
                self.dict_dtypes,
                self.start_epoch,
                self.event_attribute_dict
            )

            df.reset_index(drop=True, inplace=True)

            synthetic_df = pd.concat([synthetic_df, df], axis=0, ignore_index=True)
            len_synthetic_event_log += df['case:concept:name'].nunique()

        return synthetic_df

    def save_model(self, path: str) -> None:
        """
        Save a trained PBLES Model to a given path.

        Parameters:
        path (str): Path to save the trained PBLES Model.
        """
        os.makedirs(path, exist_ok=True)

        self.model.save(os.path.join(path, 'model.keras'))

        with open(os.path.join(path, 'tokenizer.pkl'), 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, 'cluster_dict.pkl'), 'wb') as handle:
            pickle.dump(self.cluster_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, 'dict_dtypes.pkl'), 'wb') as handle:
            pickle.dump(self.dict_dtypes, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, 'max_sequence_len.pkl'), 'wb') as handle:
            pickle.dump(self.max_sequence_len, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, 'start_epoch.pkl'), 'wb') as handle:
            pickle.dump(self.start_epoch, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, 'event_attribute_dict.pkl'), 'wb') as handle:
            pickle.dump(self.event_attribute_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> None:
        """
        Load a trained PBLES Model from a given path.

        Parameters:
        path (str): Path to the trained PBLES Model.
        """
        self.model = tf.keras.models.load_model(os.path.join(path, 'model.keras'), compile=False)

        with open(os.path.join(path, 'tokenizer.pkl'), 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        with open(os.path.join(path, 'cluster_dict.pkl'), 'rb') as handle:
            self.cluster_dict = pickle.load(handle)

        with open(os.path.join(path, 'dict_dtypes.pkl'), 'rb') as handle:
            self.dict_dtypes = pickle.load(handle)

        with open(os.path.join(path, 'max_sequence_len.pkl'), 'rb') as handle:
            self.max_sequence_len = pickle.load(handle)

        with open(os.path.join(path, 'start_epoch.pkl'), 'rb') as handle:
            self.start_epoch = pickle.load(handle)

        with open(os.path.join(path, 'event_attribute_dict.pkl'), 'rb') as handle:
            self.event_attribute_dict = pickle.load(handle)
