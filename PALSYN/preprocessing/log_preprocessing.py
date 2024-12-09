import re
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pm4py
from sklearn.cluster import KMeans
from diffprivlib.models import KMeans as DP_KMeans
from diffprivlib.mechanisms import Laplace
from tensorflow_privacy import compute_dp_sgd_privacy_statement

os.environ["LOKY_MAX_CPU_COUNT"] = str(max(os.cpu_count() - 1, 1))

START_TOKEN = 'START==START'
END_TOKEN = 'END==concept:name==END'


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
    epsilon_poisson = epsilon_poisson_match.group(1) if epsilon_poisson_match else None
    return float(epsilon_poisson)


def find_noise_multiplier(
        target_epsilon: float,
        num_examples: int,
        batch_size: int,
        epochs: int,
        tol: float = 1e-4,
        max_iter: int = 100
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
    delta = 1 / (num_examples ** 1.1)
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
            delta=delta
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
            f"Consider choosing epsilon values better suited to the dataset and model configurations"
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


def calculate_dp_bounds(df: pd.DataFrame, epsilon: float, std_multiplier: float = 2) -> dict:
    """
    Calculates differentially private bounds for numerical columns using noisy mean and standard deviation.

    For each numeric column, computes DP bounds using the formula: mean ± (std_multiplier * std).
    Special handling is applied for timestamp columns to ensure non-negative bounds.

    Parameters:
    df (pd.DataFrame): Input dataframe with numeric columns
    epsilon (float): Privacy budget for the bounds calculation, split equally between mean and std
    std_multiplier (float): Multiplier for standard deviation to determine bound width. Default is 2

    Returns:
    dict: Dictionary mapping column names to their DP bounds ([lower], [upper])

    Note:
    - Privacy budget (epsilon) is split equally between mean and standard deviation calculations
    - Timestamp columns are bounded by [0, noisy_max] to ensure validity
    """
    dp_bounds = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        col_data = df[col].dropna()

        if len(col_data) <= 1:
            dp_bounds[col] = ([np.nan], [np.nan])
            continue

        true_mean = float(col_data.mean())
        true_std = float(col_data.std())

        sensitivities = {
            'mean': true_std / np.sqrt(len(col_data)),
            'std': true_std / np.sqrt(2 * (len(col_data) - 1))
        }

        mechanisms = {
            'mean': Laplace(epsilon=epsilon / 2, sensitivity=sensitivities['mean']),
            'std': Laplace(epsilon=epsilon / 2, sensitivity=sensitivities['std'])
        }

        dp_mean = mechanisms['mean'].randomise(true_mean)
        dp_std = abs(mechanisms['std'].randomise(true_std))

        if col == "time:timestamp":
            min_bound = 0
            max_bound = max(1e-5, dp_mean + (std_multiplier * dp_std))
            bounds = ([min_bound], [max_bound])
        else:
            bounds = (
                [dp_mean - (std_multiplier * dp_std)],
                [dp_mean + (std_multiplier * dp_std)]
            )

        dp_bounds[col] = bounds

    return dp_bounds


def calculate_clusters(df, max_clusters, epsilon=None):
    """
    Calculate clusters for each numeric column using either KMeans or DP-KMeans.

    Parameters:
    df: Pandas DataFrame.
    max_clusters: Number of maximum clusters.
    epsilon: Privacy budget for DP-KMeans. If None, uses regular KMeans.

    Returns:
    tuple: A tuple containing a Pandas DataFrame with cluster labels and a dictionary with cluster information.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The input must be a pandas DataFrame")

    if not isinstance(max_clusters, int) or max_clusters <= 0:
        raise ValueError("max_clusters must be a positive integer")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_org = df.copy()
    df_cluster_list = []

    dp_bounds = None
    epsilon_k_means = None

    if epsilon:
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

        if epsilon:
            bounds = dp_bounds[col]
            clustering = DP_KMeans(n_clusters=n_clusters, epsilon=epsilon_k_means,
                                   bounds=bounds, random_state=0)
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

    cluster_dict = {}
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


def calculate_starting_epoch(
        df: pd.DataFrame,
        epsilon: float = None
) -> list:
    """
    Calculate starting epoch statistics for an event log with optional differential privacy.

    Parameters:
    df (pd.DataFrame): Event log DataFrame containing 'case:concept:name' and 'time:timestamp'
    epsilon (float, optional): Privacy budget for differential privacy. If None, returns non-DP statistics

    Returns:
    list: [Mean, Standard Deviation, Min, Max] of starting epochs
    """
    if "case:concept:name" not in df or "time:timestamp" not in df:
        raise ValueError("DataFrame must contain 'case:concept:name' and 'time:timestamp' columns")

    try:
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
        starting_epochs = df.sort_values(by="time:timestamp").groupby("case:concept:name")["time:timestamp"].first()
        starting_epoch_list = starting_epochs.astype(np.int64) // 10 ** 9

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
            'mean': range_epochs / n_traces,
            'std': range_epochs / np.sqrt(2 * n_traces)
        }

        mechanisms = {
            'mean': Laplace(epsilon=epsilon / 2, sensitivity=sensitivities['mean']),
            'std': Laplace(epsilon=epsilon / 2, sensitivity=sensitivities['std'])
        }

        dp_mean = abs(mechanisms['mean'].randomise(starting_epoch_mean))
        dp_std = abs(mechanisms['std'].randomise(starting_epoch_std))

        return [dp_mean, dp_std, starting_epoch_min, max_timestamp]

    except Exception as e:
        raise ValueError(f"Error calculating {'DP' if epsilon else ''} starting epochs: {str(e)}")


def calculate_time_between_events(df: pd.DataFrame) -> list:
    """
    Calculate the time between events for each trace in a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): A DataFrame representing an event log, expected to contain columns
                       'case:concept:name' and 'time:timestamp'.

    Returns:
    list: A list of time between events for each trace in the DataFrame, given in seconds as Unix time.
    """
    if "case:concept:name" not in df or "time:timestamp" not in df:
        raise ValueError("DataFrame must contain 'case:concept:name' and 'time:timestamp' columns")

    try:
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
    except Exception as e:
        raise ValueError(f"Error converting 'time:timestamp' to datetime: {e}")

    time_between_events = []

    for _, group in df.groupby("case:concept:name"):
        if len(group) < 2:
            time_between_events.append(0)
            continue

        time_diffs = group["time:timestamp"].diff().dt.total_seconds().copy()
        time_diffs.fillna(0, inplace=True)
        time_diffs.iloc[0] = 0
        time_between_events.extend(time_diffs)

    return time_between_events


def get_attribute_dtype_mapping(df: pd.DataFrame) -> dict:
    """
    Get the attribute data type mapping from an Event Log (XES) and return it as dictionary.
    This is necessary to generate synthetic data, maintaining the correct datatypes from the original data.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame representing an event log, where columns are attributes.

    Returns:
    dict: Dictionary containing the attribute data type mapping
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    dtype_dict = {}

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            if column == 'time:timestamp':
                dtype_dict[column] = 'float64'
            elif df[column].dropna().apply(lambda x: float(x).is_integer()).all():
                dtype_dict[column] = 'int64'
            else:
                dtype_dict[column] = 'float64'
        else:
            dtype_dict[column] = df[column].dtype.name

    return {'attribute_datatypes': dtype_dict}


def preprocess_event_log(log, max_clusters: int, trace_quantile: float, epsilon: float, batch_size: int, epochs: int):
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
        raise ValueError(f"Error converting log to DataFrame: {e}")

    print("Number of traces: " + str(df["case:concept:name"].unique().size))

    trace_length = df.groupby("case:concept:name").size()
    trace_length_q = trace_length.quantile(trace_quantile)
    df = df.groupby("case:concept:name").filter(lambda x: len(x) <= trace_length_q)

    print("Number of traces after truncation: " + str(df["case:concept:name"].unique().size))
    df = df.sort_values(by=["case:concept:name", "time:timestamp"])
    num_examples = len(df)

    if epsilon is None:
        print("No Epsilon is specified setting noise multiplier to 0")
        noise_multiplier = 0
        starting_epoch_dist = calculate_starting_epoch(df)
        time_between_events = calculate_time_between_events(df)
        df["time:timestamp"] = time_between_events
        attribute_dtype_mapping = get_attribute_dtype_mapping(df)
        df, cluster_dict = calculate_clusters(df, max_clusters)
    else:
        print("Finding Optimal Noise Multiplier")
        epsilon_noise_multiplier = epsilon / 2
        epsilon_k_means = epsilon / 2
        noise_multiplier = find_noise_multiplier(epsilon_noise_multiplier, num_examples, batch_size, epochs)
        # Epsilon does not need to be shared here since the first timestamp defines a distinct dataset.
        starting_epoch_dist = calculate_starting_epoch(df, epsilon)
        time_between_events = calculate_time_between_events(df)
        df["time:timestamp"] = time_between_events
        attribute_dtype_mapping = get_attribute_dtype_mapping(df)
        df, cluster_dict = calculate_clusters(df, max_clusters, epsilon_k_means)

    cols = ["concept:name", "time:timestamp"] + [
        col for col in df.columns if col not in ["concept:name", "time:timestamp"]
    ]
    df = df[cols]

    event_log_sentence_list = []
    total_traces = df["case:concept:name"].nunique()

    num_cols = len(df.columns) - 1
    column_list = df.columns.tolist()

    if 'case:concept:name' in column_list:
        column_list.remove('case:concept:name')

    # Pre-filter global attributes once
    global_attributes = [
        col for col in df.columns
        if col.startswith("case:") and col != "case:concept:name"
    ]

    # Pre-calculate total traces
    total_traces = df['case:concept:name'].nunique()
    event_log_sentence_list = []

    # Use groupby instead of filtering for each trace
    for i, (_, trace_group) in enumerate(df.groupby("case:concept:name"), 1):
        progress = min(99.9, (i / total_traces) * 100)
        if i % 100 == 0:  # Update progress less frequently
            print(f"\rProcessing traces: {progress:.1f}%", end="", flush=True)

        # Initialize trace sentence
        trace_sentence_list = [START_TOKEN] * num_cols

        # Handle global attributes (case: attributes)
        trace_sentence_list.extend([
            f"{attr}=={str(trace_group[attr].iloc[0])}"
            for attr in global_attributes
        ])

        # Process trace events - drop case:concept:name once
        trace_data = trace_group.drop(columns=['case:concept:name'])
        concept_names = trace_data["concept:name"].values

        # Process each event in the trace
        for idx, row in enumerate(trace_data.values):
            concept_name = concept_names[idx]
            trace_sentence_list.extend([
                f"{concept_name}=={col}=={str(val) if pd.notna(val) else 'nan'}"
                for col, val in zip(trace_data.columns, row)
            ])

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
        column_list
    )