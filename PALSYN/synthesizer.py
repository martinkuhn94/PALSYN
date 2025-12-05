"""Compatibility wrapper exposing synthesizer classes."""

from PALSYN.synthesizers.base import BaseSynthesizer
from PALSYN.synthesizers.gru import GRUSynthesizer
from PALSYN.synthesizers.lstm import LSTMSynthesizer
from PALSYN.synthesizers.rnn import RNNSynthesizer

DPEventLogSynthesizer = LSTMSynthesizer

__all__ = [
    "BaseSynthesizer",
    "LSTMSynthesizer",
    "RNNSynthesizer",
    "GRUSynthesizer",
    "DPEventLogSynthesizer",
]
