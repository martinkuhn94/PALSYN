import sys
import time

import numpy as np
from tensorflow.keras import backend as K
from keras.utils import pad_sequences


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
    # Start tim e
    start_time = time.time()

    synthetic_event_log_sentences = []
    index_word = {index: word for word, index in tokenizer.word_index.items()}
    total_processed = 0

    for offset in range(0, sample_size, batch_size):
        current_batch_size = min(batch_size, sample_size - offset)
        batch_seed_texts = [["START==START"] * num_cols for _ in range(current_batch_size)]
        batch_active = np.ones(current_batch_size, dtype=bool)

        while np.any(batch_active):
            # Prepare data for model prediction
            token_lists = [tokenizer.texts_to_sequences([seq])[0] for seq in batch_seed_texts]
            padded_token_lists = pad_sequences(token_lists, maxlen=max_sequence_len, padding="pre")

            # Reset model states before each new batch prediction
            model.reset_states()

            # Get predictions for all sequences in the batch
            predictions = model.predict(padded_token_lists, verbose=0)

            # Update sequences and check for completion
            for i, (active, seq) in enumerate(zip(batch_active, batch_seed_texts)):
                if not active:
                    continue

                latest_concept_name = None
                synth_row = []

                for prediction_output, column in zip(predictions, column_list):
                    prediction_output = prediction_output[i]  # Get predictions for current sequence
                    prediction_output = prediction_output / np.sum(prediction_output)

                    # Filter valid tokens based on column and latest concept name
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
                    next_word = index_word.get(next_word_index, "END==concept:name==END")

                    if column == "concept:name":
                        latest_concept_name = next_word.split("==")[0]
                        if latest_concept_name == "END" or next_word == "END==concept:name==END":
                            batch_active[i] = False
                            break

                    synth_row.append(next_word)

                if batch_active[i]:
                    seq.extend(synth_row)
                    if len(seq) >= (max_sequence_len * 2):
                        batch_active[i] = False

        synthetic_event_log_sentences.extend(batch_seed_texts)
        total_processed += current_batch_size

        progress = (total_processed / sample_size) * 100
        sys.stdout.write(f"\rSampling {progress:.1f}% Complete")
        sys.stdout.flush()
        K.clear_session()

    # Clean event prefixes and exclude overly long sequences
    clean_synthetic_event_log_sentences = []
    for sentence in synthetic_event_log_sentences:
        if len(sentence) >= (max_sequence_len * 1.5):
            continue

        trace = ["START==START"]
        for word in sentence:
            if not word:
                continue
            if word == "START==START" or word == "END==concept:name==END":
                continue
            elif word.startswith("case:"):
                trace.append(word)
            else:
                word_part_one = word.split("==")[1]
                word_part_two = word.split("==")[2]
                trace.append(word_part_one + "==" + word_part_two)
        trace.append("END==END")
        clean_synthetic_event_log_sentences.append(trace)

    print("\n")

    end_time = time.time()
    print("Time taken to generate synthetic event log sentences: ", end_time - start_time)
    return clean_synthetic_event_log_sentences
