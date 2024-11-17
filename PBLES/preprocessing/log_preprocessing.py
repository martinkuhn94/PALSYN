import re

import numpy as np
import pandas as pd
import pm4py
from diffprivlib.models import KMeans
from diffprivlib.mechanisms import Laplace
from tensorflow_privacy import compute_dp_sgd_privacy_statement
from datetime import datetime

from PBLES.event_attribute_model.event_attribute_model import build_attribute_model


def calculate_fixed_bins(df, num_bins):
    """
    Create a specified number of equal-sized bins for each numeric column in a DataFrame.

    Parameters:
    df: Pandas DataFrame.
    num_bins: Number of bins.

    Returns:
    tuple: A tuple containing a DataFrame with binned values and a dictionary with bin information.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The input must be a pandas DataFrame")

    if not isinstance(num_bins, int) or num_bins <= 0:
        raise ValueError("num_bins must be a positive integer")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    bin_dict = {}

    for col in numeric_cols:
        df_clean = df[col].dropna()
        if len(df_clean) == 0:
            continue

        # Calculate min and max values
        min_val = df_clean.min()
        max_val = df_clean.max()

        # Calculate bin edges
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)

        # Assign each value to a bin
        binned_values = np.digitize(df_clean, bins=bin_edges, right=False) - 1
        binned_values = np.clip(binned_values, 0, len(bin_edges) - 2)  # Ensure valid bin indices

        # Replace values in the DataFrame with the bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        df[col] = binned_values.apply(lambda x: bin_centers[x])

        # Create bin information for the column
        for i in range(len(bin_edges) - 1):
            bin_values = df_clean[(df_clean >= bin_edges[i]) & (df_clean < bin_edges[i + 1])]
            bin_dict[f"{col}_bin_{i}"] = [
                bin_edges[i],  # Bin start
                bin_edges[i + 1],  # Bin end
                bin_centers[i],  # Bin center (mean value for replacement)
                bin_values.std() if len(bin_values) > 1 else 0.0,  # Standard deviation
            ]

    return df, bin_dict



def extract_epsilon_from_string(text):
    """
    Poisson sampling is not usually done in training pipelines, but assuming
    that the data was randomly shuffled, it is believed that the actual epsilon
    should be closer to this value than the conservative assumption of an arbitrary
    data order.

    :param text:
    :return:
    """
    # Regex to find the line with epsilon assuming Poisson sampling or similar
    epsilon_poisson_match = re.search(r"Epsilon assuming Poisson sampling \(\*\):\s+([^\s]+)", text)

    if epsilon_poisson_match:
        epsilon_poisson = epsilon_poisson_match.group(1)
    else:
        epsilon_poisson = None

    return float(epsilon_poisson)


def find_noise_multiplier(target_epsilon, num_examples, batch_size, epochs, tol=1e-4, max_iter=100):
    delta = 1 / (num_examples**1.1)
    low, high = 1e-6, 30  # Initial bounds for noise multiplier
    best_noise_multiplier = None

    for _ in range(max_iter):
        mid = (low + high) / 2
        current_epsilon = compute_dp_sgd_privacy_statement(
            number_of_examples=num_examples,
            batch_size=batch_size,
            num_epochs=epochs,
            noise_multiplier=mid,
            used_microbatching=False,
            delta=delta,
        )

        current_epsilon = extract_epsilon_from_string(current_epsilon)

        if abs(current_epsilon - target_epsilon) <= tol:
            best_noise_multiplier = mid
            break

        if current_epsilon > target_epsilon:
            low = mid  # Increase noise
        else:
            high = mid  # Decrease noise

    # if noise_multiplier cannot be found after all iterations choose "high"
    if best_noise_multiplier is None:
        best_noise_multiplier = high
        # print warning message
        print(
            f"Warning: Noise multiplier could not be found within the maximum number of iterations. "
            f"Choosing the highest noise multiplier: {best_noise_multiplier}"
            f"Consider choosing another Epsilon values better suited to the dataset and the model configurations"
        )
    else:
        print(f"Optimal Noise multiplier found: {best_noise_multiplier}")
        print("Since three differential Privacy Techniques are used, the epsilon is divided in the following way: ")
        print(f"DP Bounds: {target_epsilon * 0.75}")
        print(f"DP-KMeans: {target_epsilon * 0.25}")
        print(f"DP-SDG: {target_epsilon}")

    return best_noise_multiplier


def calculate_dp_bounds(df, epsilon, lower_quantile=0.01, upper_quantile=0.99):
    """
    Calculate differentially private bounds for numeric columns using quantile-based estimation.

    Parameters:
    df (pd.DataFrame): Input dataframe.
    epsilon (float): Privacy budget for calculating the bounds.
    lower_quantile (float): Lower quantile (e.g., 0.01 for 1st percentile).
    upper_quantile (float): Upper quantile (e.g., 0.99 for 99th percentile).

    Returns:
    dict: A dictionary with differentially private bounds (min, max) for each numeric column.
    """
    dp_bounds = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        col_data = df[col].dropna()

        # Special handling for "time:timestamp"
        if col == "time:timestamp":
            min_value = 0
            max_value = col_data.quantile(upper_quantile)  # Use upper quantile for timestamp bounds
            sensitivity = max_value  # Sensitivity assumes max_value derived from upper quantile

            # Apply DP to upper quantile only
            laplace_mechanism_max = Laplace(epsilon=epsilon, sensitivity=sensitivity)
            noisy_max = laplace_mechanism_max.randomise(max_value)

            dp_bounds[col] = (min_value, noisy_max)
            print(f"Column: {col}, Min (fixed): {min_value}, DP Max: {noisy_max}")
        else:
            # Compute the true quantiles
            true_lower_quantile = col_data.quantile(lower_quantile)
            true_upper_quantile = col_data.quantile(upper_quantile)

            print(f"Column: {col}, Lower Quantile: {true_lower_quantile}, Upper Quantile: {true_upper_quantile}")

            # Sensitivity is based on the range of the true quantiles
            sensitivity = true_upper_quantile - true_lower_quantile

            # Initialize the Laplace mechanism with epsilon and calculated sensitivity
            laplace_mechanism_lower = Laplace(epsilon=epsilon / 2, sensitivity=sensitivity)
            laplace_mechanism_upper = Laplace(epsilon=epsilon / 2, sensitivity=sensitivity)

            # Apply Laplace noise to the quantiles
            noisy_lower_quantile = laplace_mechanism_lower.randomise(true_lower_quantile)
            noisy_upper_quantile = laplace_mechanism_upper.randomise(true_upper_quantile)

            # Ensure the noisy lower quantile is smaller than the noisy upper quantile
            if noisy_lower_quantile > noisy_upper_quantile:
                noisy_lower_quantile, noisy_upper_quantile = noisy_upper_quantile, noisy_lower_quantile

            # Store bounds as a tuple (noisy_lower_quantile, noisy_upper_quantile)
            dp_bounds[col] = (noisy_lower_quantile, noisy_upper_quantile)
            print(f"DP Lower Quantile: {noisy_lower_quantile}, DP Upper Quantile: {noisy_upper_quantile}")

    return dp_bounds

def calculate_cluster_dp(df, max_clusters, epsilon):
    """
    Calculate clusters for each numeric column in a pandas DataFrame using DP-KMeans and differentially private bounds.

    Parameters:
    df: Pandas DataFrame.
    max_clusters: Number of maximum clusters.
    epsilon: Privacy budget for DP-KMeans.

    Returns:
    tuple: A tuple containing a Pandas DataFrame with cluster labels and a dictionary with cluster information.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The input must be a pandas DataFrame")

    if not isinstance(max_clusters, int) or max_clusters <= 0:
        raise ValueError("max_clusters must be a positive integer")

    epsilon_bounds = epsilon * 0.75
    epsilon_k_means = epsilon * 0.25

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_org = df.copy()
    df_cluster_list = []

    # Step 1: Calculate DP bounds for numeric columns
    dp_bounds = calculate_dp_bounds(df, epsilon_bounds)

    for col in numeric_cols:
        df_clean = df[col].dropna()
        unique_values = len(df_clean.unique())
        if unique_values == 0:
            continue
        elif unique_values < max_clusters:
            n_clusters = unique_values
        else:
            n_clusters = max_clusters

        X = df_clean.values.reshape(-1, 1)

        # Use DP-KMeans with the DP bounds
        bounds = dp_bounds[col]  # Pass noisy bounds as tuple for each column
        dp_kmeans = KMeans(n_clusters=n_clusters, epsilon=epsilon_k_means, bounds=bounds, random_state=0)
        dp_kmeans.fit(X)

        label = []
        for row in df.iterrows():
            if str(row[1][col]) != "nan":
                label_temp = dp_kmeans.predict([[row[1][col]]])
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
            print(f"Cluster: {cluster}, Min: {min(dataframe_temp_cluster_values_np)}, Max: {max(dataframe_temp_cluster_values_np)}, Mean: {dataframe_temp_cluster_values_np.mean()}, Std: {dataframe_temp_cluster_values_np.std()}")
    return df, cluster_dict


def calculate_starting_epoch_dp(df: pd.DataFrame, epsilon: float) -> list:
    """
    Calculate differentially private starting epoch statistics for an event log.

    Parameters:
    df (pd.DataFrame): A DataFrame representing an event log, expected to contain columns
                       'case:concept:name' and 'time:timestamp'.
    epsilon (float): Privacy budget for differential privacy.

    Returns:
    list: A list containing four elements:
          [DP Mean, DP Standard Deviation, DP Min (fixed at 0), Max (current timestamp)].

    Raises:
    ValueError: If required columns are missing or if there are issues in date conversion.
    """
    try:
        if "case:concept:name" not in df or "time:timestamp" not in df:
            raise ValueError("DataFrame must contain 'case:concept:name' and 'time:timestamp' columns")

        # Convert timestamp to datetime and sort by it
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
        starting_epochs = df.sort_values(by="time:timestamp").groupby("case:concept:name")["time:timestamp"].first()

        # Convert timestamps to UNIX time (seconds since epoch)
        starting_epoch_list = starting_epochs.astype(np.int64) // 10**9

        if len(starting_epoch_list) == 0:
            raise ValueError("No valid starting timestamps found in the data.")

        # Calculate true statistics
        starting_epoch_mean = np.mean(starting_epoch_list)
        starting_epoch_std = np.std(starting_epoch_list)

        print(f"Real Mean: {starting_epoch_mean}, Real Std: {starting_epoch_std}")

        # Fixed minimum value
        starting_epoch_min = 0

        # Maximum value is today's timestamp
        max_timestamp = int(datetime.now().timestamp())

        # Sensitivities
        n_traces = len(starting_epoch_list)
        range_epochs = max_timestamp - starting_epoch_min

        sensitivity_mean = range_epochs / n_traces
        sensitivity_std = range_epochs / np.sqrt(2 * n_traces)  # Approximation for std sensitivity

        # Apply Laplace noise for DP
        laplace_mechanism_mean = Laplace(epsilon=epsilon / 2, sensitivity=sensitivity_mean)
        dp_mean = abs(laplace_mechanism_mean.randomise(starting_epoch_mean))

        laplace_mechanism_std = Laplace(epsilon=epsilon / 2, sensitivity=sensitivity_std)
        dp_std = abs(laplace_mechanism_std.randomise(starting_epoch_std))

        print(f"DP Mean: {dp_mean}, DP Std: {dp_std}")

        # Return results
        return [dp_mean, dp_std, starting_epoch_min, max_timestamp]

    except Exception as e:
        raise ValueError(f"An error occurred in calculating differentially private starting epochs: {str(e)}")

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
    Get the attribute data type mapping from an Event Log (XES). This is necessary to generate synthetic data,
    maintaining the correct datatypes from the original data.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame representing an event log, where columns are attributes.

    Returns:
    dict: A dictionary where keys are attribute names (column names) and values are their respective data types.

    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    return df.dtypes.apply(lambda x: x.name).to_dict()


def preprocess_event_log(log, max_clusters: int, trace_quantile: float, epsilon: float, batch_size: int, epochs: int):
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

    # Find noise multiplier
    print("Finding Optimal Noise Multiplier")
    noise_multiplier = find_noise_multiplier(epsilon, num_examples, batch_size, epochs)

    starting_epoch_dist = calculate_starting_epoch_dp(df, epsilon) # Epsilon does not need to be shared here since the first timestamp defines a distinct dataset.
    time_between_events = calculate_time_between_events(df)
    df["time:timestamp"] = time_between_events
    attribute_dtype_mapping = get_attribute_dtype_mapping(df)
    df, cluster_dict = calculate_cluster_dp(df, max_clusters, epsilon)

    cols = ["concept:name", "time:timestamp"] + [
        col for col in df.columns if col not in ["concept:name", "time" ":timestamp"]
    ]
    df = df[cols]
    event_attribute_model = build_attribute_model(df)

    event_log_sentence_list = []
    total_traces = df["case:concept:name"].nunique()

    for i, trace in enumerate(df["case:concept:name"].unique(), start=1):
        print(f"\rProcessing trace {i} of {total_traces}", end="", flush=True)
        df_temp = df[df["case:concept:name"] == trace]
        df_temp = df_temp.drop(columns=['case:concept:name'])
        trace_sentence_list = ["START==START"]

        for global_attribute in df_temp:
            if global_attribute.startswith("case:") and global_attribute != "case:concept:name":
                trace_sentence_list.append(global_attribute + "==" + str(df_temp[global_attribute].iloc[0]))

        for row in df_temp.iterrows():
            concept_name = row[1]["concept:name"]
            for col in df_temp.columns:
                trace_sentence_list.append(concept_name + "==" + col + "==" + str(row[1][col]) if str(
                    row[1][col]) != "nan" else concept_name + "==" + col + "==" + "nan")

        trace_sentence_list.append("END==END")
        event_log_sentence_list.append(trace_sentence_list)

    print()
    return (
        event_log_sentence_list,
        cluster_dict,
        attribute_dtype_mapping,
        starting_epoch_dist,
        num_examples,
        event_attribute_model,
        noise_multiplier,
    )
