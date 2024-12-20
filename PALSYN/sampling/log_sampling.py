import time
import random

import numpy as np
from tensorflow.keras import backend as K
from keras.utils import pad_sequences
from PALSYN.preprocessing.log_preprocessing import START_TOKEN, END_TOKEN


def clean_sequence(sequence: list[str], max_length: int) -> list[str]:
    if len(sequence) >= max_length:
        return []
    trace = [START_TOKEN]
    for word in sequence:
        if word and word not in {START_TOKEN, END_TOKEN}:
            if word.startswith("case:"):
                trace.append(word)
            else:
                trace.append("==".join(word.split("==")[1:]))

    return trace


def sample_batch(
        sample_size: int,
        tokenizer,
        max_sequence_len: int,
        model,
        batch_size: int,
        num_cols: int,
        column_list: list[str]
) -> list[list[str]]:
    """
    Generate synthetic event log sentences using a trained DP-BiLSTM model with conditional sampling.
    """
    #start_time = time.time()

    # Use the provided batch size instead of the full sample size
    effective_batch_size = batch_size
    synthetic_event_log_sentences = []
    index_word = {index: word for word, index in tokenizer.word_index.items()}
    batch_seed_texts = [[START_TOKEN] * num_cols for _ in range(effective_batch_size)]
    batch_active = np.ones(effective_batch_size, dtype=bool)

    while len(synthetic_event_log_sentences) < sample_size:
        # Reset batch seed texts and active flags for the current batch
        batch_seed_texts = [[START_TOKEN] * num_cols for _ in range(effective_batch_size)]
        batch_active = np.ones(effective_batch_size, dtype=bool)

        while np.any(batch_active):
            token_lists = [tokenizer.texts_to_sequences([seq])[0] for seq in batch_seed_texts]
            padded_token_lists = pad_sequences(token_lists, maxlen=max_sequence_len, padding="pre")
            model.reset_states()
            predictions = model.predict(padded_token_lists, verbose=0)

            for i, (active, seq) in enumerate(zip(batch_active, batch_seed_texts)):
                if not active:
                    continue

                latest_concept_name = None
                synth_row = []

                for prediction_output, column in zip(predictions, column_list):
                    prediction_output = prediction_output[i]
                    prediction_output = prediction_output / np.sum(prediction_output)

                    if latest_concept_name is None:
                        valid_tokens = [
                            index for index, word in index_word.items()
                            if f"=={column}==" in word
                        ]
                    else:
                        valid_tokens = [
                            index for index, word in index_word.items()
                            if f"{latest_concept_name}=={column}==" in word
                        ]

                    if len(valid_tokens) == 0:
                        batch_active[i] = False
                        break

                    filtered_probabilities = [prediction_output[token] for token in valid_tokens]
                    filtered_probabilities = np.array(filtered_probabilities) / np.sum(filtered_probabilities)

                    next_word_index = np.random.choice(valid_tokens, p=filtered_probabilities)
                    next_word = index_word.get(next_word_index, END_TOKEN)

                    if column == "concept:name":
                        latest_concept_name = next_word.split("==")[0]
                        if latest_concept_name == "END" or next_word == END_TOKEN:
                            batch_active[i] = False
                            break

                    synth_row.append(next_word)

                if batch_active[i]:
                    seq.extend(synth_row)
                    if len(seq) >= (max_sequence_len * 2):
                        batch_active[i] = False

        synthetic_event_log_sentences.extend(batch_seed_texts[:sample_size - len(synthetic_event_log_sentences)])

    K.clear_session()

    # Clean event prefixes and exclude overly long sequences
    clean_synthetic_event_log_sentences = [
        clean_sequence(sentence, round(max_sequence_len * 1.5))
        for sentence in synthetic_event_log_sentences
        if len(sentence) < round(max_sequence_len * 1.5)
    ]

    # Randomly sample the required number of sequences
    if len(clean_synthetic_event_log_sentences) > sample_size:
        clean_synthetic_event_log_sentences = random.sample(clean_synthetic_event_log_sentences, sample_size)

    #print(f"\nGenerated {len(clean_synthetic_event_log_sentences)} sequences")
    #print("Time taken to generate synthetic event log sentences: ", time.time() - start_time)

    return clean_synthetic_event_log_sentences

