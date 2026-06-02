from __future__ import annotations

import os
import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pm4py
from diffprivlib.mechanisms import Laplace
from diffprivlib.models import KMeans as DP_KMeans
from sklearn.cluster import KMeans
from tensorflow_privacy import compute_dp_sgd_privacy_statement

_CPU_COUNT = os.cpu_count() or 1
os.environ["LOKY_MAX_CPU_COUNT"] = str(max(_CPU_COUNT - 1, 1))

# Epsilon allocation for differential privacy
# Total epsilon is split as follows:
# - DP Bounds: 50% (for numeric column bounds)
# - DP KMeans: 25% (for trace clustering)
# - DP-SGD: 25% (for model training noise)
DP_EPSILON_BOUNDS_RATIO = 0.50
DP_EPSILON_KMEANS_RATIO = 0.25
DP_EPSILON_SGD_RATIO = 0.25

START_TOKEN = "START==START"  # noqa: S105 - sentinel token marker
END_TOKEN = "END==concept:name==END"  # noqa: S105 - sentinel token marker


def extract_epsilon_from_string(text: str) -> float:
    """
    Extracts the epsilon value from a privacy report string, assuming Poisson sampling.

    This function parses the privacy report text to find the epsilon value calculated under
    Poisson sampling assumptions. While Poisson sampling is not commonly used in training pipelines,
    with randomly shuffled data the actual epsilon is likely closer to this value compared to
    assuming arbitrary data ordering.

    Parameters:
    text (str): Privacy report text containing the epsilon value.

    Returns:
    float: Extracted epsilon value assuming Poisson sampling. Returns None if no match is found.
    """
    pattern = re.compile(
        r"Epsilon assuming Poisson sampling \(\*\):\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)"
    )
    match = pattern.search(text)
    if match is None:
        warnings.warn(
            "Could not extract epsilon from privacy statement; defaulting to 0.0.",
            RuntimeWarning,
            stacklevel=2,
        )
    return float(match.group(1)) if match else 0.0


def find_noise_multiplier(
    target_epsilon: float,
    num_examples: int,
    batch_size: int,
    epochs: int,
    tol: float = 1e-4,
    max_iter: int = 100,
    privacy_statement_fn=None,
) -> float:
    """
    Finds optimal noise multiplier for differential privacy using binary search.
    The function searches for a noise multiplier that achieves the target epsilon value
    within the specified tolerance, considering multiple DP techniques.

    Parameters:
    target_epsilon (float): Target privacy budget epsilon value
    num_examples (int): Number of training examples
    batch_size (int): Size of training batches
    epochs (int): Number of training epochs
    tol (float): Tolerance for epsilon convergence. Default is 1e-4
    max_iter (int): Maximum number of binary search iterations. Default is 100

    Returns:
    float: Optimal noise multiplier value that achieves target epsilon

    Note:
    The privacy budget is divided among three DP techniques:
    - DP Bounds: 25% of target epsilon
    - DP-KMeans: 25% of target epsilon
    - DP-SGD: 50% of target epsilon
    """
    if target_epsilon <= 0 or tol <= 0 or max_iter <= 0:
        raise ValueError("target_epsilon, tol, and max_iter must be positive.")
    if num_examples <= 0 or batch_size <= 0 or epochs <= 0:
        raise ValueError("num_examples, batch_size, and epochs must be positive.")

    delta = 1 / (num_examples**1.1)
    low, high = 1e-6, 100.0
    best_noise = None

    if privacy_statement_fn is None:
        privacy_statement_fn = compute_dp_sgd_privacy_statement

    def epsilon_for_noise(noise: float) -> float:
        statement = privacy_statement_fn(
            number_of_examples=num_examples,
            batch_size=batch_size,
            num_epochs=epochs,
            noise_multiplier=noise,
            used_microbatching=False,
            delta=delta,
        )
        return extract_epsilon_from_string(statement)

    for _ in range(max_iter):
        current_noise = (low + high) / 2.0
        current_epsilon = epsilon_for_noise(current_noise)

        if abs(current_epsilon - target_epsilon) <= tol:
            best_noise = current_noise
            break

        if current_epsilon > target_epsilon:
            low = current_noise
        else:
            high = current_noise

    if best_noise is None:
        warnings.warn(
            "Noise multiplier search did not converge; returning upper bound.",
            RuntimeWarning,
            stacklevel=2,
        )
        return high

    return best_noise


def calculate_dp_bounds(
    df: pd.DataFrame, epsilon: float, std_multiplier: float = 2
) -> dict[str, tuple[list[float], list[float]]]:
    """Compute DP bounds for numeric columns using noisy mean/std statistics."""
    dp_bounds: dict[str, tuple[list[float], list[float]]] = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        col_data = df[col].dropna()

        if len(col_data) <= 1:
            dp_bounds[col] = ([float("nan")], [float("nan")])
            continue

        true_mean = float(col_data.mean())
        true_std = float(col_data.std())

        sensitivities = {
            "mean": true_std / np.sqrt(len(col_data)),
            "std": true_std / np.sqrt(2 * (len(col_data) - 1)),
        }

        mechanisms = {
            "mean": Laplace(epsilon=epsilon / 2, sensitivity=sensitivities["mean"]),
            "std": Laplace(epsilon=epsilon / 2, sensitivity=sensitivities["std"]),
        }

        dp_mean = float(mechanisms["mean"].randomise(true_mean))
        dp_std = float(abs(mechanisms["std"].randomise(true_std)))

        if col == "time:timestamp":
            min_bound = 0.0
            max_bound = float(max(1e-5, dp_mean + (std_multiplier * dp_std)))
            bounds = ([min_bound], [max_bound])
        else:
            lower = float(dp_mean - (std_multiplier * dp_std))
            upper = float(dp_mean + (std_multiplier * dp_std))
            bounds = ([lower], [upper])

        dp_bounds[col] = bounds

    return dp_bounds


def calculate_clusters(  # noqa: C901 - clustering has branching logic
    df: pd.DataFrame, max_clusters: int, epsilon: float | None = None
) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    """Cluster numeric columns using KMeans or DP-KMeans and return labels plus metadata."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The input must be a pandas DataFrame")

    if not isinstance(max_clusters, int) or max_clusters <= 0:
        raise ValueError("max_clusters must be a positive integer")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_cluster_list: list[tuple[pd.DataFrame, str]] = []

    dp_bounds: dict[str, tuple[list[float], list[float]]] | None = None
    epsilon_k_means: float | None = None

    if epsilon is not None:
        epsilon_k_means = epsilon * DP_EPSILON_KMEANS_RATIO
        dp_bounds = calculate_dp_bounds(df, epsilon * DP_EPSILON_BOUNDS_RATIO)

    for col in numeric_cols:
        df_clean = df[col].dropna()
        unique_values = len(df_clean.unique())

        if unique_values == 0:
            continue
        n_clusters = min(unique_values, max_clusters)

        X = df_clean.values.reshape(-1, 1)

        # Store original values before clustering
        original_values = df_clean.copy()

        if epsilon is not None:
            if dp_bounds is None or epsilon_k_means is None:
                raise ValueError(
                    "Differential privacy bounds must be computed when epsilon is set."
                )
            bounds = dp_bounds.get(col, ([float(df_clean.min())], [float(df_clean.max())]))
            clustering = DP_KMeans(
                n_clusters=n_clusters, epsilon=epsilon_k_means, bounds=bounds, random_state=0
            )
        else:
            clustering = KMeans(n_clusters=n_clusters, random_state=0)

        clustering.fit(X)

        labels = _cluster_column(df, col, clustering)

        df[col] = labels
        df_cluster_list.append(
            (_create_cluster_dataframe_with_original(df, col, labels, original_values), col)
        )

    cluster_dict = _build_cluster_dict(df_cluster_list)

    return df, cluster_dict


def _cluster_column(
    df: pd.DataFrame, col: str, clustering: Any
) -> list[str]:
    """Cluster a single column and return cluster labels."""
    labels = []

    for _, row in df.iterrows():
        if pd.notna(row[col]):
            label_temp = clustering.predict([[row[col]]])
            labels.append(f"{col}_cluster_{label_temp[0]}")
        else:
            labels.append(np.nan)

    return labels


def _create_cluster_dataframe_with_original(
    df: pd.DataFrame, col: str, labels: list[str], original_values: pd.Series
) -> pd.DataFrame:
    """Create a dataframe with cluster assignments and original values."""
    df_cluster = df.copy()
    df_cluster[f"{col}_cluster_label"] = labels
    # Use reindex to preserve original index and values
    df_cluster[f"{col}_original"] = original_values.reindex(df_cluster.index)
    return df_cluster[[f"{col}_original", f"{col}_cluster_label"]].dropna()


def _build_cluster_dict(df_cluster_list: list[tuple[pd.DataFrame, str]]) -> dict[str, list[float]]:
    """Build cluster statistics dictionary from cluster dataframes."""
    cluster_dict: dict[str, list[float]] = {}
    for dataframe, original_col in df_cluster_list:
        cluster_label_col = f"{original_col}_cluster_label"
        original_value_col = f"{original_col}_original"
        unique_cluster = dataframe[cluster_label_col].unique()
        for cluster in unique_cluster:
            cluster_data = dataframe[dataframe[cluster_label_col] == cluster]
            cluster_values = cluster_data[original_value_col].to_numpy()
            cluster_dict[cluster] = [
                float(np.min(cluster_values)),
                float(np.max(cluster_values)),
                float(np.mean(cluster_values)),
                float(np.std(cluster_values)),
            ]
    return cluster_dict


def calculate_starting_epoch(df: pd.DataFrame, epsilon: float | None = None) -> list[float]:
    """
    Calculate starting epoch statistics with optional differential privacy.

    Parameters:
    df (pd.DataFrame): Event log containing ``case:concept:name`` and ``time:timestamp``.
    epsilon (float, optional): Privacy budget. When ``None`` raw statistics are returned.

    Returns:
    list: ``[mean, std, min, max]`` describing the starting epoch distribution.
    """
    if "case:concept:name" not in df or "time:timestamp" not in df:
        raise ValueError("DataFrame must contain 'case:concept:name' and 'time:timestamp' columns")

    try:
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
        starting_epochs = (
            df.sort_values(by="time:timestamp")
            .groupby("case:concept:name")["time:timestamp"]
            .first()
        )
        starting_epoch_list = starting_epochs.astype(np.int64) // 10**9

        if len(starting_epoch_list) == 0:
            raise ValueError("No valid starting timestamps found in the data.")

        starting_epoch_mean = np.mean(starting_epoch_list)
        starting_epoch_std = np.std(starting_epoch_list)
        starting_epoch_min = 0
        max_timestamp = int(starting_epoch_list.max())

        if epsilon is None:
            return [starting_epoch_mean, starting_epoch_std, starting_epoch_min, max_timestamp]

        n_traces = len(starting_epoch_list)
        range_epochs = max_timestamp - starting_epoch_min

        sensitivities = {
            "mean": range_epochs / n_traces,
            "std": range_epochs / np.sqrt(2 * n_traces),
        }

        mechanisms = {
            "mean": Laplace(epsilon=epsilon / 2, sensitivity=sensitivities["mean"]),
            "std": Laplace(epsilon=epsilon / 2, sensitivity=sensitivities["std"]),
        }

        dp_mean = abs(mechanisms["mean"].randomise(starting_epoch_mean))
        dp_std = abs(mechanisms["std"].randomise(starting_epoch_std))

        return [dp_mean, dp_std, starting_epoch_min, max_timestamp]

    except Exception as e:
        raise ValueError(
            f"Error calculating {'DP' if epsilon else ''} starting epochs: {str(e)}"
        ) from e


def calculate_time_between_events(df: pd.DataFrame) -> list[float]:
    """
    Calculate per-trace event deltas for a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): Event log with ``case:concept:name`` and ``time:timestamp``.

    Returns:
    list: Time between events (seconds since epoch) for every trace.
    """
    if "case:concept:name" not in df or "time:timestamp" not in df:
        raise ValueError("DataFrame must contain 'case:concept:name' and 'time:timestamp' columns")

    try:
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
    except Exception as e:
        raise ValueError("Error converting 'time:timestamp' to datetime") from e

    time_between_events: list[float] = []

    for _, group in df.groupby("case:concept:name"):
        if len(group) < 2:
            time_between_events.append(0)
            continue

        time_diffs = group["time:timestamp"].diff().dt.total_seconds().copy()
        time_diffs.fillna(0, inplace=True)
        time_diffs.iloc[0] = 0
        time_between_events.extend(time_diffs.astype(float).tolist())

    return time_between_events


def get_attribute_dtype_mapping(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """
    Determine the attribute datatype mapping from the event log.

    Parameters:
    df (pd.DataFrame): Event log DataFrame whose columns represent attributes.

    Returns:
    dict: Column-to-datatype mapping used during generation.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    dtype_dict: dict[str, str] = {}

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            if column == "time:timestamp":
                dtype_dict[column] = "float64"
            elif df[column].dropna().apply(lambda x: float(x).is_integer()).all():
                dtype_dict[column] = "int64"
            else:
                dtype_dict[column] = "float64"
        else:
            dtype_dict[column] = df[column].dtype.name

    return {"attribute_datatypes": dtype_dict}


def preprocess_event_log(
    log: Any,
    max_clusters: int,
    trace_quantile: float,
    epsilon: float | None,
    batch_size: int,
    epochs: int,
) -> tuple[
    list[list[str]],
    dict[str, list[float]],
    dict[str, dict[str, str]],
    list[float],
    int,
    float,
    int,
    list[str],
]:
    """
    Preprocesses event log data with optional differential privacy.

    Parameters:
    log: Event log to process
    max_clusters (int): Maximum number of clusters for trace clustering
    trace_quantile (float): Quantile value for trace length filtering
    epsilon (float): Privacy budget (None for no DP)
    batch_size (int): Batch size for DP-SGD
    epochs (int): Number of training epochs

    Returns:
    tuple: Processed event log data and metadata
    """
    df = _convert_log_to_dataframe(log)
    df = _filter_traces_by_quantile(df, trace_quantile)
    num_examples = len(df)

    _print_trace_statistics(df)

    df = df.sort_values(by=["case:concept:name", "time:timestamp"])

    noise_multiplier, starting_epoch_dist, time_between_events = _process_timestamps(
        df, epsilon, batch_size, epochs
    )
    df["time:timestamp"] = time_between_events

    attribute_dtype_mapping = get_attribute_dtype_mapping(df)
    df, cluster_dict = _process_clusters(df, max_clusters, epsilon)

    df = _reorder_columns(df)

    event_log_sentence_list, column_list, num_cols = _build_event_log_sentences(
        df, num_examples
    )

    return (
        event_log_sentence_list,
        cluster_dict,
        attribute_dtype_mapping,
        starting_epoch_dist,
        num_examples,
        noise_multiplier,
        num_cols,
        column_list,
    )


def _convert_log_to_dataframe(log: Any) -> pd.DataFrame:
    """Convert event log to DataFrame with error handling."""
    try:
        return pm4py.convert_to_dataframe(log)
    except Exception as e:
        raise ValueError(f"Error converting log to DataFrame: {e}") from e


def _filter_traces_by_quantile(df: pd.DataFrame, trace_quantile: float) -> pd.DataFrame:
    """Filter traces by quantile to remove overly long traces."""
    trace_length = df.groupby("case:concept:name").size()
    trace_length_q = trace_length.quantile(trace_quantile)
    return df.groupby("case:concept:name").filter(lambda x: len(x) <= trace_length_q)


def _print_trace_statistics(df: pd.DataFrame) -> None:
    """Print trace count statistics."""
    num_traces = df["case:concept:name"].unique().size
    print(f"Number of traces: {num_traces}")


def _process_timestamps(
    df: pd.DataFrame, epsilon: float | None, batch_size: int, epochs: int
) -> tuple[float, list[float], list[float]]:
    """Process timestamps and compute noise multiplier for DP."""
    if epsilon is None:
        print("No Epsilon is specified setting noise multiplier to 0")
        noise_multiplier = 0.0
        starting_epoch_dist = calculate_starting_epoch(df)
        time_between_events = calculate_time_between_events(df)
    else:
        print("Finding Optimal Noise Multiplier")
        epsilon_sgd = epsilon * DP_EPSILON_SGD_RATIO
        noise_multiplier = find_noise_multiplier(
            epsilon_sgd, len(df), batch_size, epochs
        )
        starting_epoch_dist = calculate_starting_epoch(df, epsilon)
        time_between_events = calculate_time_between_events(df)

    return noise_multiplier, starting_epoch_dist, time_between_events


def _process_clusters(
    df: pd.DataFrame, max_clusters: int, epsilon: float | None
) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    """Process numeric columns with clustering."""
    if epsilon is None:
        return calculate_clusters(df, max_clusters)
    else:
        return calculate_clusters(df, max_clusters, epsilon)


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns to put standard columns first."""
    cols = ["concept:name", "time:timestamp"] + [
        col for col in df.columns if col not in ["concept:name", "time:timestamp"]
    ]
    return df[cols]


def _build_event_log_sentences(
    df: pd.DataFrame, num_examples: int
) -> tuple[list[list[str]], list[str], int]:
    """Build event log sentences from DataFrame."""
    column_list = df.columns.tolist()
    if "case:concept:name" in column_list:
        column_list.remove("case:concept:name")

    num_cols = len(column_list)

    global_attributes = [
        col for col in df.columns if col.startswith("case:") and col != "case:concept:name"
    ]

    event_log_sentence_list = _create_sentence_list(
        df, global_attributes, column_list, num_cols
    )

    return event_log_sentence_list, column_list, num_cols


def _create_sentence_list(
    df: pd.DataFrame,
    global_attributes: list[str],
    column_list: list[str],
    num_cols: int,
) -> list[list[str]]:
    """Create list of event log sentences."""
    event_log_sentence_list: list[list[str]] = []
    total_traces = df["case:concept:name"].nunique()

    for i, (_, trace_group) in enumerate(df.groupby("case:concept:name"), 1):
        _update_progress(i, total_traces)

        trace_sentence_list = _create_trace_sentence(
            trace_group, global_attributes, column_list, num_cols
        )
        event_log_sentence_list.append(trace_sentence_list)

    print("\rProcessing traces: 100.0%", end="", flush=True)
    print()  # New line after completion

    return event_log_sentence_list


def _update_progress(current: int, total: int) -> None:
    """Update and print progress."""
    progress = min(99.9, (current / total) * 100)
    if current % 100 == 0:
        print(f"\rProcessing traces: {progress:.1f}%", end="", flush=True)


def _create_trace_sentence(
    trace_group: pd.DataFrame,
    global_attributes: list[str],
    column_list: list[str],
    num_cols: int,
) -> list[str]:
    """Create a single trace sentence."""
    trace_sentence_list = [START_TOKEN] * num_cols

    trace_sentence_list.extend(
        [f"{attr}=={str(trace_group[attr].iloc[0])}" for attr in global_attributes]
    )

    trace_data = trace_group.drop(columns=["case:concept:name"])
    concept_names = trace_data["concept:name"].values

    for idx, row in enumerate(trace_data.values):
        concept_name = concept_names[idx]
        trace_sentence_list.extend(
            [
                f"{concept_name}=={col}=={str(val) if pd.notna(val) else 'nan'}"
                for col, val in zip(trace_data.columns, row)
            ]
        )

    trace_sentence_list.extend([END_TOKEN] * num_cols)
    return trace_sentence_list
