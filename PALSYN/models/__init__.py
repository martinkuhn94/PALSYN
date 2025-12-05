from __future__ import annotations

from typing import Mapping

from .base import LastTimeStep
from .esn import EchoStateCell
from .lnn import LiquidTimeConstantCell
from .transformer import TransformerBlock

_CUSTOM_LAYER_TYPES = [
    LastTimeStep,
    LiquidTimeConstantCell,
    EchoStateCell,
    TransformerBlock,
]


def get_custom_objects() -> Mapping[str, type]:
    """Return Keras custom objects required for model deserialization."""
    return {cls.__name__: cls for cls in _CUSTOM_LAYER_TYPES}


__all__ = ["get_custom_objects", "LastTimeStep"]
