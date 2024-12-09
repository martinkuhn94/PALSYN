import time

import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class MetricsLogger(Callback):
    """
    Custom callback for logging training metrics during model training. This callback collects metrics for each epoch
    and stores them in a format that can be easily converted to a pandas DataFrame.

    Parameters:
    num_cols (int): Number of output columns in the model.
    column_list (list): List of column names for the outputs.

    Returns:
    None
    """

    def __init__(self, num_cols: int, column_list: list) -> None:
        super().__init__()
        self.num_cols = num_cols
        self.column_list = [col.replace(":", "_").replace(" ", "_") for col in column_list]
        self.history = []
        self._supports_tf_logs = False

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        """
        Callback called at the end of each epoch. Collects and stores metrics for the epoch including accuracy
        and loss for each output column.

        Parameters:
        epoch (int): Current epoch number (0-based).
        logs (dict, optional): Dictionary of metrics.

        Returns:
        None
        """
        epoch_metrics = {'epoch': epoch + 1}
        logs = logs or {}

        for i in range(self.num_cols):
            output_acc = f'{self.column_list[i]}_accuracy'
            output_loss = f'{self.column_list[i]}_loss'

            if output_acc in logs:
                epoch_metrics[output_acc] = logs[output_acc]
            if output_loss in logs:
                epoch_metrics[output_loss] = logs[output_loss]

        if 'loss' in logs:
            epoch_metrics['total_loss'] = logs['loss']

        self.history.append(epoch_metrics)

    def get_dataframe(self) -> pd.DataFrame:
        """
        Convert the collected metrics history to a pandas DataFrame.

        Parameters:
        None

        Returns:
        pd.DataFrame: DataFrame containing all collected metrics.
        """
        return pd.DataFrame(self.history)


class CustomProgressBar(tf.keras.callbacks.Callback):
    """
    Custom progress bar callback for training visualization. Displays a progress bar with ETA and timing
    information during model training.

    Parameters:
    None

    Returns:
    None
    """

    def __init__(self) -> None:
        super().__init__()
        self.last_update = None
        self.start_time = None
        self.target = None
        self.seen = None

    def on_epoch_begin(self, epoch: int, logs: dict = None) -> None:
        """
        Initialize progress bar at the start of each epoch.

        Parameters:
        epoch (int): Current epoch number (0-based).
        logs (dict, optional): Dictionary of metrics.

        Returns:
        None
        """
        print(f'\nEpoch {epoch + 1}/{self.params["epochs"]}')
        self.seen = 0
        self.target = self.params['steps']
        self.start_time = time.time()
        self.last_update = time.time()

    def on_batch_end(self, batch: int, logs: dict = None) -> None:
        """
        Update progress bar after each batch. Calculates and displays progress, ETA, and timing information.

        Parameters:
        batch (int): Current batch number.
        logs (dict, optional): Dictionary of metrics.

        Returns:
        None
        """
        self.seen += 1
        now = time.time()

        time_elapsed = now - self.start_time
        steps_remaining = self.target - self.seen
        time_per_step = time_elapsed / self.seen
        eta_seconds = steps_remaining * time_per_step

        if eta_seconds < 60:
            eta_str = f"{int(eta_seconds)}s"
        elif eta_seconds < 3600:
            eta_str = f"{int(eta_seconds / 60)}m {int(eta_seconds % 60)}s"
        else:
            eta_str = f"{int(eta_seconds / 3600)}h {int((eta_seconds % 3600) / 60)}m"

        progress = int(30 * self.seen / self.target)
        bar = '=' * progress + '>' + '.' * (29 - progress)

        time_per_step_ms = time_per_step * 1000

        print(f'\r{self.seen}/{self.target} [{bar}] - ETA: {eta_str} - {time_per_step_ms:.0f}ms/step', end='')

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        """
        Finalize progress bar at the end of each epoch. Displays final timing information for the completed epoch.

        Parameters:
        epoch (int): Current epoch number (0-based).
        logs (dict, optional): Dictionary of metrics.

        Returns:
        None
        """
        total_time = time.time() - self.start_time
        if total_time < 60:
            time_str = f"{total_time:.0f}s"
        elif total_time < 3600:
            time_str = f"{int(total_time / 60)}m {int(total_time % 60)}s"
        else:
            time_str = f"{int(total_time / 3600)}h {int((total_time % 3600) / 60)}m"

        print(f'\r{self.target}/{self.target} [==============================] - {time_str}')
