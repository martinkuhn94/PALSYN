from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

IntArray = npt.NDArray[np.int_]


def tokenize_log(
    event_log_sentences: Sequence[Sequence[str]], steps: int = 1
) -> tuple[IntArray, IntArray, int, int, Tokenizer]:
    """
    Tokenize event log sentences and build next-step targets.

    Parameters:
    event_log_sentences (Sequence[Sequence[str]]): Event log sentences represented as token lists.
    steps (int): Number of tokens to predict jointly.

    Returns:
    tuple: ``(xs, ys, vocab_size, max_len, tokenizer)``.

    Raises:
    ValueError: If ``event_log_sentences`` is empty or ``steps`` < 1.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not event_log_sentences:
        raise ValueError("event_log_sentences must contain at least one sentence")

    tokenizer = Tokenizer(lower=False)
    tokenizer.fit_on_texts(event_log_sentences)
    total_words = len(tokenizer.word_index) + 1
    print(f"Number of unique tokens: {total_words - 1}")

    encoded_sentences = tokenizer.texts_to_sequences(event_log_sentences)

    input_sequences: list[list[int]] = []
    targets: list[list[int]] = []

    for token_list in encoded_sentences:
        for i in range(steps, len(token_list), steps):
            input_sequences.append(token_list[:i])

            next_steps = token_list[i : i + steps]
            if len(next_steps) < steps:
                next_steps = next_steps + [0] * (steps - len(next_steps))
            targets.append(next_steps)

    if not input_sequences:
        raise ValueError("event_log_sentences produced no token sequences")

    max_sequence_len = max(len(seq) for seq in input_sequences)
    padded_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")

    xs: IntArray = np.asarray(padded_sequences, dtype=np.int_)
    ys_array: IntArray = np.asarray(targets, dtype=np.int_)

    print(f"Number of input sequences: {len(xs)}")
    print(f"Sequence Length: {max_sequence_len}")

    return xs, ys_array, total_words, max_sequence_len, tokenizer
