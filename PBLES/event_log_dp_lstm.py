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
    GRU,
)
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import (
    DPKerasAdamOptimizer,
)

from PBLES.preprocessing.log_preprocessing import preprocess_event_log
from PBLES.preprocessing.log_tokenization import tokenize_log
from PBLES.sampling.log_sampling import sample_batch_markov
from PBLES.postprocessing.log_postprocessing import generate_df


class EventLogDpLstm:
    def __init__(
            self,
            embedding_output_dims=16,
            method='LSTM',
            units_per_layer=None,
            epochs=3,
            batch_size=16,
            max_clusters=10,
            dropout=0.0,
            trace_quantile=0.95,
            l2_norm_clip=1.5,
            epsilon=1.0,
    ):
        """
        Class of the Private Bi-LSTM Event Log Synthesizer (CPBLES) approach for synthesizing event logs. The class is
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
        self.event_attribute_model = None

        # Attribute Preprocessing Information
        self.model = None
        self.max_sequence_len = None
        self.total_words = None
        self.tokenizer = None
        self.ys = None
        self.xs = None
        self.start_epoch = None

        # Model Information
        self.units_per_layer = units_per_layer
        self.method = method
        self.embedding_output_dims = embedding_output_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.petri_net = None

        # Privacy Information
        self.noise_multiplier = None
        self.epsilon = epsilon
        self.l2_norm_clip = l2_norm_clip
        self.num_examples = None

    def fit(self, input_data: pd.DataFrame) -> None:
        """
        Fit and train a DP-Bi-LSTM Model on an event log. The model can be used to
        generate synthetic event logs.

        Parameters:
        input_data (Any): Input data for training the model, typically an event log.
        """
        print(f"Initializing {self.method} Model")

        # Convert Event Log to sentences and preprocess
        (
            self.event_log_sentences,
            self.cluster_dict,
            self.dict_dtypes,
            self.start_epoch,
            self.num_examples,
            self.event_attribute_model,
            self.noise_multiplier
        ) = preprocess_event_log(input_data, self.max_clusters, self.trace_quantile, (self.epsilon / 2),
                                 self.batch_size,
                                 self.epochs)

        # Tokenize Attributes
        (
            self.xs,
            self.ys,
            self.total_words,
            self.max_sequence_len,
            self.tokenizer
        ) = tokenize_log(self.event_log_sentences, variant='attributes')

        print(f"Creating {self.method} Model")
        # Input layer
        inputs = Input(shape=(self.max_sequence_len - 1,))

        # Embedding layer
        embedding_layer = Embedding(
            self.total_words,
            self.embedding_output_dims,
            input_length=self.max_sequence_len - 1
        )(inputs)

        # Masking layer (handles variable-length sequences with padding)
        x = Masking(mask_value=0)(embedding_layer)

        # Dynamically add layers based on method and units_per_layer
        for i, units in enumerate(self.units_per_layer):
            if self.method == 'LSTM':
                x = LSTM(units, return_sequences=(i < len(self.units_per_layer) - 1))(x)
            elif self.method == 'Bi-LSTM':
                x = Bidirectional(LSTM(units, return_sequences=(i < len(self.units_per_layer) - 1)))(x)
            elif self.method == 'GRU':
                x = GRU(units, return_sequences=(i < len(self.units_per_layer) - 1))(x)

        # Batch Normalization and Dropout layers
        x = BatchNormalization()(x)
        x = Dropout(self.dropout)(x)

        # Output layer
        outputs = Dense(self.total_words, activation='softmax')(x)

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

    def sample(self, sample_size: int, batch_size: int, temperature: float = 1.0) -> pd.DataFrame:
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

            synthetic_event_log_sentences = sample_batch_markov(
                sample_size_new,
                self.tokenizer,
                self.max_sequence_len,
                self.model,
                self.event_attribute_model,
                batch_size,
                temperature
            )

            # Generate Event Log DataFrame
            df = generate_df(
                synthetic_event_log_sentences,
                self.cluster_dict,
                self.dict_dtypes,
                self.start_epoch
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

        self.event_attribute_model.to_excel(os.path.join(path, 'event_attribute_model.xlsx'), index=False)

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

    def load(self, path: str) -> None:
        """
        Load a trained PBLES Model from a given path.

        Parameters:
        path (str): Path to the trained PBLES Model.
        """
        self.model = tf.keras.models.load_model(os.path.join(path, 'model.keras'), compile=False)

        self.event_attribute_model = pd.read_excel(os.path.join(path, 'event_attribute_model.xlsx'))

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
