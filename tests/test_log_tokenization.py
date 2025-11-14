
import numpy as np
import pytest

from PALSYN.preprocessing.log_tokenization import tokenize_log


def test_tokenize_log_basic_sequence():
    sentences = [
        ["START", "concept:name==A==foo", "concept:name==B==bar", "END"],
        ["START", "concept:name==A==bar", "END"],
    ]

    xs, ys, total_words, max_seq_len, tokenizer = tokenize_log(sentences, steps=1)

    assert xs.shape == (5, 3)
    assert ys.shape == (5, 1)
    assert xs.shape[0] == ys.shape[0]
    assert max_seq_len == 3
    assert total_words == len(tokenizer.word_index) + 1
    assert np.all(xs >= 0)
    assert np.all(ys >= 0)
    assert np.all(xs[:, -1] != 0)
    assert np.all(xs[:, :-1] >= 0)


def test_tokenize_log_requires_positive_steps():
    with pytest.raises(ValueError):
        tokenize_log([["START", "END"]], steps=0)


def test_tokenize_log_requires_sentences():
    with pytest.raises(ValueError):
        tokenize_log([], steps=1)
