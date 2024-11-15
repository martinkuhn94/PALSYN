import random
import sys

import numpy as np
from tensorflow.keras import backend as K
from keras.utils import pad_sequences


def sample_batch(
    sample_size: int, tokenizer, max_sequence_len_attr: int, model, temperature_attr: float, batch_size: int
) -> list[list[str]]:
    """
    Generate synthetic event log sentences using a trained DP-BiLSTM model. The sampling is done in batches.

    Parameters:
    sample_size (int): Total number of samples to generate.
    tokenizer_attr (Any): Tokenizer for converting text to sequences.
    max_sequence_len_attr (int): Maximum sequence length for padding.
    model_attr (Any): Trained LSTM model for generating sequences.
    temperature_attr (float): Temperature for sampling probability adjustment.
    batch_size (int): Size of the batches for sampling.

    Returns:
    list: List of generated synthetic event log sentences.
    """
    synthetic_event_log_sentences = []
    index_word = {index: word for word, index in tokenizer.word_index.items()}

    total_processed = 0
    for offset in range(0, sample_size, batch_size):
        current_batch_size = min(batch_size, sample_size - offset)
        batch_seed_texts = [["START==START"] for _ in range(current_batch_size)]
        batch_active = np.ones(current_batch_size, dtype=bool)

        while np.any(batch_active):
            # Prepare data for model prediction
            token_lists = [tokenizer.texts_to_sequences([seq])[0] for seq in batch_seed_texts]
            padded_token_lists = pad_sequences(token_lists, maxlen=max_sequence_len_attr - 1, padding="pre")

            # Reset model states before each new batch prediction
            model.reset_states()

            # Predict next tokens
            predictions = model.predict(padded_token_lists, verbose=0)

            # Update sequences and check for completion
            for i, (active, seq) in enumerate(zip(batch_active, batch_seed_texts)):
                if not active:
                    continue

                prediction_output = predictions[i]
                prediction_output_sq = np.power(prediction_output, temperature_attr)
                prediction_output_normalized = prediction_output_sq / np.sum(prediction_output_sq)
                predicted_index = np.random.choice(len(prediction_output_normalized), p=prediction_output_normalized)
                next_word = index_word.get(predicted_index, "")

                seq.append(next_word)
                if next_word == "END==END" or len(seq) >= max_sequence_len_attr:
                    batch_active[i] = False

        synthetic_event_log_sentences.extend(batch_seed_texts)
        total_processed += current_batch_size

        # Progress update
        progress = (total_processed / sample_size) * 100
        sys.stdout.write(f"\rSampling {progress:.1f}% Complete")
        sys.stdout.flush()

        K.clear_session()

    print("\n")

    return synthetic_event_log_sentences


def sample_batch_markov(
    sample_size: int,
    tokenizer,
    max_sequence_len_attr: int,
    model,
    event_attribute_model,
    batch_size: int,
    temperature_attr: float,
) -> list[list[str]]:
    """
    Generate synthetic event log sentences using a trained DP-BiLSTM model with conditional sampling. Therefore, the
     (Conditional Private Bi-LSTM Event Log Synthesizer is used. The sampling is done in batches.

    Parameters:
    sample_size (int): Total number of samples to generate.
    tokenizer_attr (Any): Tokenizer for converting text to sequences.
    max_sequence_len_attr (int): Maximum sequence length for padding.
    model_attr (Any): Trained LSTM model for generating sequences.
    temperature_attr (float): Temperature for sampling probability adjustment.
    batch_size (int): Size of the batches for sampling.

    Returns:
    list: List of generated synthetic event log sentences.
    """
    synthetic_event_log_sentences = []
    index_word = {index: word for word, index in tokenizer.word_index.items()}
    reverse_index_word = {v: k for k, v in index_word.items()}

    total_processed = 0
    for offset in range(0, sample_size, batch_size):
        current_batch_size = min(batch_size, sample_size - offset)
        batch_seed_texts = [["START==START"] for _ in range(current_batch_size)]
        batch_active = np.ones(current_batch_size, dtype=bool)

        while np.any(batch_active):
            # Prepare data for model prediction
            token_lists = [tokenizer.texts_to_sequences([seq])[0] for seq in batch_seed_texts]
            padded_token_lists = pad_sequences(token_lists, maxlen=max_sequence_len_attr - 1, padding="pre")

            # Reset model states before each new batch prediction
            model.reset_states()

            # Predict next tokens
            predictions = model.predict(padded_token_lists, verbose=0)

            # Update sequences and check for completion
            for i, (active, seq) in enumerate(zip(batch_active, batch_seed_texts)):
                if not active:
                    continue

                current_word = seq[-1]
                current_word_event_attribute_prefix = "==".join(current_word.split("==")[:2])

                if event_attribute_model["Current State"].str.startswith(current_word_event_attribute_prefix).any():
                    markov_model_filtered = event_attribute_model[
                        event_attribute_model["Current State"].str.startswith(current_word_event_attribute_prefix)
                    ]

                    next_possible_prefixes = markov_model_filtered["Next State"].tolist()

                    # Get all words starting with next possible next words from index_word
                    next_words_possible = []
                    for next_possible_prefix in next_possible_prefixes:
                        next_words_possible.extend(
                            [i for i in index_word.values() if i.startswith(next_possible_prefix)]
                        )
                    next_words_possible = list(set(next_words_possible))
                    next_words_possible = [reverse_index_word.get(i, "") for i in next_words_possible]

                    # Make Conditional Predictions
                    prediction_output = predictions[i]
                    prediction_output = [prediction_output[j] for j in next_words_possible]
                    if sum(prediction_output) == 0:
                        prediction_output = [1.0 / len(next_words_possible)] * len(next_words_possible)
                    else:
                        prediction_output = prediction_output / np.sum(prediction_output)
                    prediction_output = prediction_output / np.sum(prediction_output)
                    prediction_output_sq = np.power(prediction_output, temperature_attr)
                    prediction_output_normalized = prediction_output_sq / np.sum(prediction_output_sq)
                    next_words_possible = [index_word.get(i, "") for i in next_words_possible]
                    next_word = random.choices(next_words_possible, weights=prediction_output_normalized, k=1)[0]
                    seq.append(next_word)
                    if next_word == "END==END" or len(seq) >= (max_sequence_len_attr * 2):
                        batch_active[i] = False

                else:
                    prediction_output = predictions[i]
                    prediction_output_sq = np.power(prediction_output, temperature_attr)
                    prediction_output_normalized = prediction_output_sq / np.sum(prediction_output_sq)
                    predicted_index = np.random.choice(
                        len(prediction_output_normalized), p=prediction_output_normalized
                    )
                    next_word = index_word.get(predicted_index, "")

                    seq.append(next_word)
                    if next_word == "END==END" or len(seq) >= max_sequence_len_attr:
                        batch_active[i] = False

        synthetic_event_log_sentences.extend(batch_seed_texts)
        total_processed += current_batch_size

        # Progress update
        progress = (total_processed / sample_size) * 100
        sys.stdout.write(f"\rSampling {progress:.1f}% Complete")
        sys.stdout.flush()
        K.clear_session()

    # Clean event prefixes and exclude overly long sequences
    clean_synthetic_event_log_sentences = []
    for sentence in synthetic_event_log_sentences:
        # Skip sequences that are too long
        if len(sentence) >= (max_sequence_len_attr * 2):
            continue

        trace = []
        for word in sentence:
            if not word:
                continue
            if word == "START==START" or word == "END==END" or word.startswith("case:"):
                trace.append(word)
            else:
                word_part_one = word.split("==")[1]
                word_part_two = word.split("==")[2]
                trace.append(word_part_one + "==" + word_part_two)

        clean_synthetic_event_log_sentences.append(trace)

    print("\n")

    return clean_synthetic_event_log_sentences
