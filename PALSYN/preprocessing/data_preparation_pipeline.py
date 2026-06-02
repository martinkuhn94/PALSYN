from __future__ import annotations

from typing import Any

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
