from __future__ import annotations

"""Lightweight preprocessing pipeline for PALSYN.

This module demonstrates how to separate data preparation from model handling.
The pipeline purposefully ignores model-specific knobs such as training epochs,
batch sizes, and optimizer privacy budgets so that those decisions can be made
later by whatever synthesizer consumes the prepared tensors. The
``DataPreparationPipeline`` returns plain dictionaries instead of custom base
classes to prioritize readability.
"""

from collections.abc import Mapping
from typing import Any

import pandas as pd

from PALSYN.preprocessing.log_preprocessing import preprocess_event_log
from PALSYN.preprocessing.log_tokenization import tokenize_log


class DataPreparationPipeline:
    """Combine preprocessing and tokenization without touching the model layer."""

    def __init__(self, max_clusters: int = 10, trace_quantile: float = 0.95) -> None:
        self.max_clusters = max_clusters
        self.trace_quantile = trace_quantile

    def run(
        self,
        event_log: Any,
        *,
        epsilon: float | None,
        batch_size: int,
        epochs: int,
    ) -> dict[str, Any]:
        """Execute the preprocessing steps and return tensors plus metadata.

        Args:
            event_log: Anything ``pm4py.convert_to_dataframe`` can handle.
            epsilon: Privacy budget shared with preprocessing; ``None`` disables DP.
            batch_size: Training batch size (used to derive DP-SGD noise).
            epochs: Number of training epochs (used to derive DP-SGD noise).

        Returns:
            Dictionary with tokenized tensors, tokenizer, preprocessing metadata,
            and the derived noise multiplier (0 when ``epsilon`` is ``None``).
        """
        (
            event_log_sentences,
            cluster_dict,
            attribute_dtypes,
            start_epoch_stats,
            num_examples,
            noise_multiplier,
            num_cols,
            column_list,
        ) = preprocess_event_log(
            log=event_log,
            max_clusters=self.max_clusters,
            trace_quantile=self.trace_quantile,
            epsilon=epsilon,
            batch_size=batch_size,
            epochs=epochs,
        )

        xs, ys, total_words, max_sequence_len, tokenizer = tokenize_log(
            event_log_sentences, steps=num_cols
        )

        return {
            "event_log_sentences": event_log_sentences,
            "xs": xs,
            "ys": ys,
            "total_words": total_words,
            "max_sequence_len": max_sequence_len,
            "tokenizer": tokenizer,
            "column_list": column_list,
            "num_cols": num_cols,
            "cluster_dict": cluster_dict,
            "attribute_dtypes": attribute_dtypes,
            "start_epoch_stats": start_epoch_stats,
            "num_examples": num_examples,
            "noise_multiplier": noise_multiplier,
        }


def _build_toy_log() -> pd.DataFrame:
    """Create a tiny event log for demonstration/testing purposes."""
    data = {
        "case:concept:name": ["C1", "C1", "C2", "C2"],
        "concept:name": ["Start", "Approve", "Start", "Reject"],
        "time:timestamp": [
            "2024-01-01T09:00:00",
            "2024-01-01T10:00:00",
            "2024-01-02T08:30:00",
            "2024-01-02T12:45:00",
        ],
        "org:resource": ["Alice", "Bob", "Alice", "Cara"],
        "amount": [100.0, 150.0, 80.0, 120.0],
    }
    df = pd.DataFrame(data)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
    return df


def _pretty_print(metadata: Mapping[str, Any]) -> None:
    """Print a small subset of the returned metadata for illustration."""
    print("Tokens shape (xs):", metadata["xs"].shape)
    print("Targets shape (ys):", metadata["ys"].shape)
    print("Vocabulary size:", metadata["total_words"])
    print("Columns modeled:", metadata["column_list"])
    print("Cluster keys:", list(metadata["cluster_dict"].keys()))


if __name__ == "__main__":
    pipeline = DataPreparationPipeline()
    artifacts = pipeline.run(_build_toy_log())
    _pretty_print(artifacts)
