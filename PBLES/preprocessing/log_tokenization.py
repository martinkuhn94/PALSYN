from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np


def tokenize_log(event_log_sentences: list, variant: str) -> tuple:
    """
    Tokenize event log sentences based on the specified variant ('control-flow' or 'attributes').

    Parameters:
    event_log_sentences (list): List of event log sentences.
    variant (str): Variant of the event log sentences ('control-flow' or 'attributes').

    Returns:
    tuple: Tokenized event log sentences, one-hot encoded labels, total number of words,
           maximum sequence length, and tokenizer.

    Raises:
    ValueError: If event_log_sentences is not a list and when the variant is not 'control-flow' or 'attributes'.
    """
    if not isinstance(event_log_sentences, list):
        raise ValueError("event_log_sentences must be a list")

    if variant == 'control-flow':
        event_log_sentences = [
            [word for word in sentence if word.startswith('concept:name') or word in ['START==START', 'END==END']]
            for sentence in event_log_sentences
        ]
    elif variant != 'attributes':
        raise ValueError("Variant not found. Please choose between 'control-flow' and 'attributes'")

    tokenizer = Tokenizer(lower=False)
    tokenizer.fit_on_texts(event_log_sentences)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in event_log_sentences:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    max_sequence_len = max(len(x) for x in input_sequences)
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
    ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

    print("delta", len(xs))

    return xs, ys, total_words, max_sequence_len, tokenizer
