from __future__ import annotations

from typing import Mapping, Type

from .base import Encoder, LastTimeStep
from .conformer import ConformerBlock, ConformerEncoder
from .esn import EchoStateCell, EchoStateNetworkEncoder
from .gru import BidirectionalGRUEncoder, GRUEncoder
from .lnn import LiquidNeuralNetworkEncoder, LiquidTimeConstantCell
from .lstm import BidirectionalLSTMEncoder, LSTMEncoder
from .rnn import BidirectionalSimpleRNNEncoder, SimpleRNNEncoder
from .tcn import TCNEncoder
from .transformer import TransformerBlock, TransformerEncoder
from .wavenet import WaveNetBlock, WaveNetEncoder

ENCODER_REGISTRY: Mapping[str, Type[Encoder]] = {
    "LSTM": LSTMEncoder,
    "Bi-LSTM": BidirectionalLSTMEncoder,
    "GRU": GRUEncoder,
    "Bi-GRU": BidirectionalGRUEncoder,
    "RNN": SimpleRNNEncoder,
    "Bi-RNN": BidirectionalSimpleRNNEncoder,
    "TCN": TCNEncoder,
    "LNN": LiquidNeuralNetworkEncoder,
    "Conformer": ConformerEncoder,
    "ESN": EchoStateNetworkEncoder,
    "Transformer": TransformerEncoder,
    "WaveNet": WaveNetEncoder,
}

_CUSTOM_LAYER_TYPES = [
    LastTimeStep,
    LiquidTimeConstantCell,
    ConformerBlock,
    EchoStateCell,
    TransformerBlock,
    WaveNetBlock,
]


def get_encoder_class(method: str) -> Type[Encoder]:
    """Return the encoder class registered for the provided method name."""
    try:
        return ENCODER_REGISTRY[method]
    except KeyError as exc:
        available = ", ".join(sorted(ENCODER_REGISTRY))
        raise ValueError(f"Unknown encoder method '{method}'. Supported: {available}") from exc


def get_custom_objects() -> Mapping[str, type]:
    """Return Keras custom objects required for model deserialization."""
    return {cls.__name__: cls for cls in _CUSTOM_LAYER_TYPES}


__all__ = [
    "BidirectionalGRUEncoder",
    "BidirectionalLSTMEncoder",
    "BidirectionalSimpleRNNEncoder",
    "ConformerBlock",
    "ConformerEncoder",
    "EchoStateCell",
    "EchoStateNetworkEncoder",
    "ENCODER_REGISTRY",
    "Encoder",
    "get_custom_objects",
    "GRUEncoder",
    "LSTMEncoder",
    "LastTimeStep",
    "LiquidNeuralNetworkEncoder",
    "LiquidTimeConstantCell",
    "SimpleRNNEncoder",
    "TCNEncoder",
    "TransformerBlock",
    "TransformerEncoder",
    "WaveNetBlock",
    "WaveNetEncoder",
    "get_encoder_class",
]
