import sys

import numpy as np
from tensorflow.keras import backend as K
from keras.utils import pad_sequences


def sample_batch(
        sample_size: int,
        tokenizer,
        max_sequence_len_attr: int,
        model,
        temperature_attr: float,
        batch_size: int
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
        batch_seed_texts = [['START==START'] for _ in range(current_batch_size)]
        batch_active = np.ones(current_batch_size, dtype=bool)

        while np.any(batch_active):
            # Prepare data for model prediction
            token_lists = [tokenizer.texts_to_sequences([seq])[0] for seq in batch_seed_texts]
            padded_token_lists = pad_sequences(token_lists, maxlen=max_sequence_len_attr - 1, padding='pre')

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
                next_word = index_word.get(predicted_index, '')

                seq.append(next_word)
                if next_word == 'END==END' or len(seq) >= max_sequence_len_attr:
                    batch_active[i] = False

        synthetic_event_log_sentences.extend(batch_seed_texts)
        total_processed += current_batch_size

        # Progress update
        progress = (total_processed / sample_size) * 100
        sys.stdout.write(f'\rSampling {progress:.1f}% Complete')
        sys.stdout.flush()

        K.clear_session()

    print("\n")

    return synthetic_event_log_sentences
