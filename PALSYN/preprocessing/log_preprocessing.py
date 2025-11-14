from __future__ import annotations

import os
import re
from datetime import datetime
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
    epsilon_poisson_match = re.search(r"Epsilon assuming Poisson sampling \(\*\):\s+(\S+)", text)
    if epsilon_poisson_match is None:
        raise ValueError("Could not extract epsilon from privacy statement.")
    epsilon_poisson = epsilon_poisson_match.group(1)
    return float(epsilon_poisson)


def find_noise_multiplier(
    target_epsilon: float,
    num_examples: int,
    batch_size: int,
    epochs: int,
    tol: float = 1e-4,
    max_iter: int = 100,
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
    delta = 1 / (num_examples**1.1)
    search_range = {"low": 1e-6, "high": 100}
    noise_multiplier = None

    for _ in range(max_iter):
        current_noise = (search_range["low"] + search_range["high"]) / 2

        privacy_statement = compute_dp_sgd_privacy_statement(
            number_of_examples=num_examples,
            batch_size=batch_size,
            num_epochs=epochs,
            noise_multiplier=current_noise,
            used_microbatching=False,
            delta=delta,
        )

        current_epsilon = extract_epsilon_from_string(privacy_statement)
        epsilon_difference = abs(current_epsilon - target_epsilon)

        if epsilon_difference <= tol:
            noise_multiplier = current_noise
            break

        search_range["low" if current_epsilon > target_epsilon else "high"] = current_noise

    if noise_multiplier is None:
        noise_multiplier = search_range["high"]
        print(
            f"Warning: Noise multiplier could not be found within {max_iter} iterations.\n"
            f"Using highest noise multiplier: {noise_multiplier}\n"
            "Consider choosing epsilon values better suited to the dataset "
            "and model configurations"
        )
    else:
        print(
            f"Optimal Noise multiplier found: {noise_multiplier}\n"
            f"Privacy budget distribution:\n"
            f"- DP Bounds: {target_epsilon * 0.5}\n"
            f"- DP-KMeans: {target_epsilon * 0.5}\n"
            f"- DP-SGD: {target_epsilon}"
        )

    return noise_multiplier


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
    df_org = df.copy()
    df_cluster_list: list[pd.DataFrame] = []

    dp_bounds: dict[str, tuple[list[float], list[float]]] | None = None
    epsilon_k_means: float | None = None

    if epsilon is not None:
        epsilon_bounds = epsilon * 0.5
        epsilon_k_means = epsilon * 0.5
        dp_bounds = calculate_dp_bounds(df, epsilon_bounds)

    for col in numeric_cols:
        df_clean = df[col].dropna()
        unique_values = len(df_clean.unique())

        if unique_values == 0:
            continue
        n_clusters = min(unique_values, max_clusters)

        X = df_clean.values.reshape(-1, 1)

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

        label = []
        for row in df.iterrows():
            if str(row[1][col]) != "nan":
                label_temp = clustering.predict([[row[1][col]]])
                label.append(col + "_cluster_" + str(label_temp[0]))
            else:
                label.append(np.nan)

        df[col] = label
        df_org[col + "_cluster_label"] = label
        df_cluster_list.append(df_org[[col, col + "_cluster_label"]].dropna())

    cluster_dict: dict[str, list[float]] = {}
    for dataframe in df_cluster_list:
        unique_cluster = dataframe[dataframe.columns[1]].unique()
        for cluster in unique_cluster:
            dataframe_temp_values = dataframe[dataframe[dataframe.columns[1]] == cluster]
            dataframe_temp_cluster_values = dataframe_temp_values[dataframe_temp_values.columns[0]]
            dataframe_temp_cluster_values_np = dataframe_temp_cluster_values.to_numpy()
            cluster_dict[cluster] = [
                min(dataframe_temp_cluster_values_np),
                max(dataframe_temp_cluster_values_np),
                dataframe_temp_cluster_values_np.mean(),
                dataframe_temp_cluster_values_np.std(),
            ]

    return df, cluster_dict


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
        max_timestamp = int(datetime.now().timestamp())

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
    try:
        df = pm4py.convert_to_dataframe(log)
    except Exception as e:
        raise ValueError("Error converting log to DataFrame") from e

    print("Number of traces: " + str(df["case:concept:name"].unique().size))

    trace_length = df.groupby("case:concept:name").size()
    trace_length_q = trace_length.quantile(trace_quantile)
    df = df.groupby("case:concept:name").filter(lambda x: len(x) <= trace_length_q)

    print("Number of traces after truncation: " + str(df["case:concept:name"].unique().size))
    df = df.sort_values(by=["case:concept:name", "time:timestamp"])
    num_examples = len(df)

    if epsilon is None:
        print("No Epsilon is specified setting noise multiplier to 0")
        noise_multiplier = 0.0
        starting_epoch_dist = calculate_starting_epoch(df)
        time_between_events = calculate_time_between_events(df)
        df["time:timestamp"] = time_between_events
        attribute_dtype_mapping = get_attribute_dtype_mapping(df)
        df, cluster_dict = calculate_clusters(df, max_clusters)
    else:
        print("Finding Optimal Noise Multiplier")
        epsilon_noise_multiplier = epsilon / 2
        epsilon_k_means = epsilon / 2
        noise_multiplier = find_noise_multiplier(
            epsilon_noise_multiplier, num_examples, batch_size, epochs
        )
        # Epsilon does not need to be shared here since the first timestamp defines a
        # distinct dataset.
        starting_epoch_dist = calculate_starting_epoch(df, epsilon)
        time_between_events = calculate_time_between_events(df)
        df["time:timestamp"] = time_between_events
        attribute_dtype_mapping = get_attribute_dtype_mapping(df)
        df, cluster_dict = calculate_clusters(df, max_clusters, epsilon_k_means)

    cols = ["concept:name", "time:timestamp"] + [
        col for col in df.columns if col not in ["concept:name", "time:timestamp"]
    ]
    df = df[cols]

    event_log_sentence_list: list[list[str]] = []
    total_traces = df["case:concept:name"].nunique()

    num_cols = len(df.columns) - 1
    column_list = df.columns.tolist()

    if "case:concept:name" in column_list:
        column_list.remove("case:concept:name")

    # Pre-filter global attributes once
    global_attributes = [
        col for col in df.columns if col.startswith("case:") and col != "case:concept:name"
    ]

    # Use groupby instead of filtering for each trace
    for i, (_, trace_group) in enumerate(df.groupby("case:concept:name"), 1):
        progress = min(99.9, (i / total_traces) * 100)
        if i % 100 == 0:  # Update progress less frequently
            print(f"\rProcessing traces: {progress:.1f}%", end="", flush=True)

        # Initialize trace sentence
        trace_sentence_list = [START_TOKEN] * num_cols

        # Handle global attributes (case: attributes)
        trace_sentence_list.extend(
            [f"{attr}=={str(trace_group[attr].iloc[0])}" for attr in global_attributes]
        )

        # Process trace events - drop case:concept:name once
        trace_data = trace_group.drop(columns=["case:concept:name"])
        concept_names = trace_data["concept:name"].values

        # Process each event in the trace
        for idx, row in enumerate(trace_data.values):
            concept_name = concept_names[idx]
            trace_sentence_list.extend(
                [
                    f"{concept_name}=={col}=={str(val) if pd.notna(val) else 'nan'}"
                    for col, val in zip(trace_data.columns, row)
                ]
            )

        trace_sentence_list.extend([END_TOKEN] * num_cols)
        event_log_sentence_list.append(trace_sentence_list)

    # Print 100% at completion with carriage return
    print("\rProcessing traces: 100.0%", end="", flush=True)
    print()  # New line after completion

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
