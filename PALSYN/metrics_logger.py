from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import pandas as pd
from keras.callbacks import Callback as _RuntimeCallback

if TYPE_CHECKING:
    class Callback(_RuntimeCallback):
        params: dict[str, Any]
        model: Any

        def __init__(self, *args: Any, **kwargs: Any) -> None: ...

else:  # pragma: no cover - runtime import for actual callback behavior
    Callback = _RuntimeCallback


class MetricsLogger(Callback):
    """Collect per-epoch training and validation metrics into a DataFrame.

    Captures loss and accuracy for each model output and total loss at the
    end of every epoch. The history can be exported via `get_dataframe`.

    Args:
        num_cols: Number of output heads in the model.
        column_list: Original output names; sanitized for metric keys.
    """

    def __init__(self, num_cols: int, column_list: list[str]) -> None:
        if not TYPE_CHECKING:
            super().__init__()
        self.num_cols = int(num_cols)
        self.column_list = [str(col).replace(":", "_").replace(" ", "_") for col in column_list]
        self.history: list[dict[str, Any]] = []
        self._supports_tf_logs = False

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        epoch_metrics: dict[str, Any] = {"epoch": epoch + 1}
        logs = logs or {}

        for i in range(self.num_cols):
            base = self.column_list[i]
            acc = f"{base}_accuracy"
            loss = f"{base}_loss"
            val_acc = f"val_{base}_accuracy"
            val_loss = f"val_{base}_loss"

            if acc in logs:
                epoch_metrics[acc] = logs[acc]
            if loss in logs:
                epoch_metrics[loss] = logs[loss]
            if val_acc in logs:
                epoch_metrics[val_acc] = logs[val_acc]
            if val_loss in logs:
                epoch_metrics[val_loss] = logs[val_loss]

        if "loss" in logs:
            epoch_metrics["total_loss"] = logs["loss"]
        if "val_loss" in logs:
            epoch_metrics["val_total_loss"] = logs["val_loss"]

        self.history.append(epoch_metrics)

    def get_dataframe(self) -> pd.DataFrame:
        """Return the collected metrics as a pandas DataFrame."""
        return pd.DataFrame(self.history)


class CustomProgressBar(Callback):
    """Minimal progress bar with ETA for model training.

    Prints a single-line progress bar for each epoch with ETA and per-step
    timing. Degrades gracefully if the number of steps is unknown.
    """

    def __init__(self) -> None:
        if not TYPE_CHECKING:
            super().__init__()
        self.last_update: float = 0.0
        self.start_time: float = 0.0
        self.target: int | None = None
        self.seen: int = 0
        self._last_line_len: int = 0

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        print(f"\nEpoch {epoch + 1}/{self.params.get('epochs', '?')}")
        self.seen = 0
        steps = self.params.get("steps")
        if steps is None:
            samples = self.params.get("samples")
            batch_size = self.params.get("batch_size")
            if isinstance(samples, int) and isinstance(batch_size, int) and batch_size > 0:
                steps = (samples + batch_size - 1) // batch_size
        self.target = int(steps) if steps else None
        now = time.time()
        self.start_time = now
        self.last_update = now
        self._last_line_len = 0

    def on_batch_end(self, batch: int, logs: dict[str, Any] | None = None) -> None:
        self.seen += 1
        if not self.target or self.target <= 0:
            return
        now = time.time()
        elapsed = max(now - self.start_time, 1e-6)
        time_per_step = elapsed / max(self.seen, 1)
        steps_remaining = max(self.target - self.seen, 0)
        eta_seconds = steps_remaining * time_per_step

        if eta_seconds < 60:
            eta_str = f"{int(eta_seconds)}s"
        elif eta_seconds < 3600:
            eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
        else:
            eta_str = f"{int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60)}m"

        width = 30
        progress = int(width * self.seen / max(self.target, 1))
        progress = min(progress, width)
        bar = "=" * progress + ">" + "." * max(width - 1 - progress, 0)
        ms = time_per_step * 1000.0
        line = f"{self.seen}/{self.target} [{bar}] - ETA: {eta_str} - {ms:.0f}ms/step"
        pad = " " * max(self._last_line_len - len(line), 0)
        print("\r" + line + pad, end="", flush=True)
        self._last_line_len = len(line)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        total = max(time.time() - self.start_time, 0.0)
        if total < 60:
            time_str = f"{total:.0f}s"
        elif total < 3600:
            time_str = f"{int(total // 60)}m {int(total % 60)}s"
        else:
            time_str = f"{int(total // 3600)}h {int((total % 3600) // 60)}m"
        if self.target:
            final_line = (
                f"{self.target}/{self.target} [==============================] - {time_str}"
            )
        else:
            final_line = f"- {time_str}"
        pad = " " * max(self._last_line_len - len(final_line), 0)
        print("\r" + final_line + pad)
        self._last_line_len = 0
