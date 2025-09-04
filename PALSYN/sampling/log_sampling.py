import time
import random
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from keras import backend as K
from keras.utils import pad_sequences
from PALSYN.preprocessing.log_preprocessing import START_TOKEN, END_TOKEN


def clean_sequence(sequence: List[str], max_length: int) -> List[str]:
    """Clean a generated token sequence.

    Removes special markers, strips event concept prefixes, and preserves
    case-level tokens. The returned sequence is prepended with `START_TOKEN`.
    If the input sequence length is greater than or equal to `max_length`, an
    empty list is returned to filter out runaway generations.

    Args:
        sequence: Generated tokens that may include case-level entries (prefix
            `"case:"`) and event-level tokens in the form
            `"<concept>==<column>==<value>"`.
        max_length: Maximum allowed raw sequence length; sequences at or above
            this length are discarded.

    Returns:
        A cleaned token sequence beginning with `START_TOKEN`, or an empty list
        if discarded.
    """
    if len(sequence) >= max_length:
        return []

    trace: List[str] = [START_TOKEN]
    for word in sequence:
        if not word or word in {START_TOKEN, END_TOKEN}:
            continue
        if word.startswith("case:"):
            trace.append(word)
        else:
            parts = word.split("==")
            trace.append("==".join(parts[1:]))

    return trace


def _build_token_index_maps(index_word: Dict[int, str], columns: Iterable[str]) -> Tuple[Dict[str, List[int]], Dict[Tuple[str, str], List[int]]]:
    """Build lookup tables of token indices by column and by (concept, column).

    Args:
        index_word: Mapping from token index to the corresponding token string.
        columns: Iterable of column names aligned with the model outputs.

    Returns:
        A tuple with two dictionaries:
        - by_column: Maps `column` to a list of token indices whose token
          contains the substring `"=={column}=="`.
        - by_concept_and_column: Maps `(concept, column)` to a list of token
          indices whose token contains `"{concept}=={column}=="`.
    """
    col_set = set(columns)
    by_column: Dict[str, List[int]] = {c: [] for c in col_set}
    by_concept_and_column: Dict[Tuple[str, str], List[int]] = {}

    for idx, token in index_word.items():
        for col in col_set:
            marker = f"=={col}=="
            if marker in token:
                by_column[col].append(idx)
                concept = token.split("==", 1)[0]
                by_concept_and_column.setdefault((concept, col), []).append(idx)

    return by_column, by_concept_and_column


def _safe_normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize an array to a probability distribution.

    If the input sums to zero (or is empty), returns a uniform distribution to
    avoid NaNs.

    Args:
        arr: Array of non-negative scores.

    Returns:
        Array of probabilities summing to 1.0.
    """
    arr = np.asarray(arr, dtype=float)
    total = float(arr.sum())
    if total > 0.0:
        return arr / total
    n = arr.size if arr.size > 0 else 1
    return np.full(n, 1.0 / n, dtype=float)


def sample_batch(
    sample_size: int,
    tokenizer: Any,
    max_sequence_len: int,
    model: Any,
    batch_size: int,
    num_cols: int,
    column_list: List[str],
) -> List[List[str]]:
    """Generate synthetic sequences using conditional per-column sampling.

    At each step, samples one token per column, constraining event-level
    attributes to match the current `concept:name`. A sequence terminates when
    `concept:name` predicts `END` or no valid candidates remain.

    Args:
        sample_size: Target number of sequences to return.
        tokenizer: Fitted tokenizer exposing `word_index` and
            `texts_to_sequences`.
        max_sequence_len: Maximum sequence length used during training; also
            bounds generation and post-processing.
        model: Trained Keras model with one softmax output per column in
            `column_list`.
        batch_size: Minimum number of sequences generated in one sweep.
        num_cols: Number of output columns per generation step.
        column_list: Ordered list of column names matching model outputs.

    Returns:
        List of cleaned token sequences (each a list of strings).

    Notes:
        - Prints a simple progress bar to stdout during generation.
        - Clears the Keras backend session before returning.
    """
    start_time = time.perf_counter()

    effective_batch_size = max(int(batch_size), int(sample_size))

    seed_sequences: List[List[str]] = [[START_TOKEN] * num_cols for _ in range(effective_batch_size)]
    active_mask = np.ones(effective_batch_size, dtype=bool)

    index_word: Dict[int, str] = {index: word for word, index in tokenizer.word_index.items()}

    tokens_by_column, tokens_by_concept_column = _build_token_index_maps(index_word, column_list)

    total_sequences = effective_batch_size
    completed = 0
    last_percent = 0

    def update_progress() -> None:
        nonlocal completed, last_percent
        completed += 1
        pct = int((completed / total_sequences) * 100)
        if pct > last_percent:
            filled = pct // 2
            bar = "█" * filled + "░" * (50 - filled)
            print(f"\rProgress: |{bar}| {pct}% ", end="", flush=True)
            last_percent = pct

    while np.any(active_mask):
        token_lists = [tokenizer.texts_to_sequences([seq])[0] for seq in seed_sequences]
        padded = pad_sequences(token_lists, maxlen=max_sequence_len, padding="pre")

        try:
            model.reset_states()
        except Exception:
            pass

        predictions = model.predict(padded, verbose=0)

        for i, (active, seq) in enumerate(zip(active_mask, seed_sequences)):
            if not active:
                continue

            current_concept: str | None = None
            step_tokens: List[str] = []

            for output_probs, column in zip(predictions, column_list):
                probs = _safe_normalize(output_probs[i])

                if current_concept is None:
                    candidates = tokens_by_column.get(column, [])
                else:
                    candidates = tokens_by_concept_column.get((current_concept, column), [])

                if not candidates:
                    active_mask[i] = False
                    update_progress()
                    break

                cand_probs = _safe_normalize(np.array([probs[idx] for idx in candidates], dtype=float))

                next_index = np.random.choice(candidates, p=cand_probs)
                next_word = index_word.get(int(next_index), END_TOKEN)

                if column == "concept:name":
                    current_concept = next_word.split("==")[0]
                    if current_concept == "END" or next_word == END_TOKEN:
                        active_mask[i] = False
                        update_progress()
                        break

                step_tokens.append(next_word)

            if active_mask[i]:
                seq.extend(step_tokens)
                if len(seq) >= (max_sequence_len * 2):
                    active_mask[i] = False
                    update_progress()

    K.clear_session()

    max_len_cutoff = int(round(max_sequence_len * 1.5))
    cleaned: List[List[str]] = [
        clean_sequence(sentence, max_len_cutoff)
        for sentence in seed_sequences
        if len(sentence) < max_len_cutoff
    ]

    if len(cleaned) > sample_size:
        cleaned = random.sample(cleaned, sample_size)

    duration = time.perf_counter() - start_time
    print(f"\nGenerated {len(cleaned)} sequences")
    print(f"Time to generate sequences: {duration:.2f}s")

    return cleaned
