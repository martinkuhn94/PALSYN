import re

import numpy as np
import pandas as pd
import pm4py
from diffprivlib.models import KMeans
from diffprivlib.mechanisms import Laplace
from tensorflow_privacy import compute_dp_sgd_privacy_statement

from PBLES.event_attribute_model.event_attribute_model import build_attribute_model


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


def calculate_dp_bounds(df, epsilon):
    """
    Calculate differentially private bounds (min and max) for each numeric column in the dataframe.
    The sensitivity is calculated as the difference between the maximum and minimum values in each column.

    Parameters:
    df (pd.DataFrame): Input dataframe.
    epsilon (float): Privacy budget for calculating the bounds.

    Returns:
    dict: A dictionary with differentially private bounds (min, max) for each numeric column.
    """
    dp_bounds = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        col_data = df[col].dropna()

        # Compute the true min and max
        min_value = col_data.min()
        max_value = col_data.max()
        print(f"Column: {col}, Min: {min_value}, Max: {max_value}")

        # Sensitivity is calculated as the difference between max and min
        sensitivity = max_value - min_value

        # Initialize the Laplace mechanism with epsilon and calculated sensitivity
        laplace_mechanism_min = Laplace(epsilon=epsilon / 2, sensitivity=sensitivity)
        laplace_mechanism_max = Laplace(epsilon=epsilon / 2, sensitivity=sensitivity)

        # Apply Laplace noise to min and max values
        noisy_min = laplace_mechanism_min.randomise(min_value)
        noisy_max = laplace_mechanism_max.randomise(max_value)

        # Ensure min is smaller than max, swap if necessary
        if noisy_min > noisy_max:
            noisy_min, noisy_max = noisy_max, noisy_min

        # Store bounds as a tuple (min, max)
        dp_bounds[col] = (noisy_min, noisy_max)
        print(f"DP Min: {noisy_min}, DP Max: {noisy_max}")

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

    return df, cluster_dict


def calculate_starting_epoch(df: pd.DataFrame) -> list:
    """
    Calculate the starting epoch for an event log. The starting epoch is the average starting time of the first events
    in each trace, represented as unix time. This function calculates the mean and standard deviation of these times.

    Parameters:
    df (pd.DataFrame): A DataFrame representing an event log, expected to contain columns
                       'case:concept:name' and 'time:timestamp'.

    Returns:
    list: A list containing two elements: the mean and standard deviation of the starting epochs.

    Raises:
    ValueError: If required columns are missing or if there are issues in date conversion.
    """
    try:
        if "case:concept:name" not in df or "time:timestamp" not in df:
            raise ValueError("DataFrame must contain 'case:concept:name' and 'time:timestamp' columns")

        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])

        starting_epochs = df.sort_values(by="time:timestamp").groupby("case:concept:name")["time:timestamp"].first()

        starting_epoch_list = starting_epochs.astype(np.int64) // 10**9

        if len(starting_epoch_list) < len(starting_epochs):
            print("Warning: Some traces did not have valid starting timestamps and were excluded from the calculation.")

        if len(starting_epoch_list) > 0:
            starting_epoch_mean = np.mean(starting_epoch_list)
            starting_epoch_std = np.std(starting_epoch_list)
            starting_epoch_dist = [starting_epoch_mean, starting_epoch_std]
        else:
            raise ValueError("No valid starting timestamps found in the data.")

        return starting_epoch_dist

    except Exception as e:
        raise ValueError(f"An error occurred in calculating starting epochs: {str(e)}")


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
    """
    Preprocess an event log. The event log is transformed into a pandas DataFrame. The time between events is calculated
    and added to the DataFrame. The DataFrame is clustered and the cluster information is added to the DataFrame. The
    DataFrame is transformed into a list of event log sentences. Each event log sentence is like a list of words.
    Each word is a string of the form 'attribute_name==attribute_value'.

    Parameters:
    trace_quantile (float): Quantile used to truncate trace length.
    log: Event Log (XES).
    max_clusters (int): Maximum number of clusters. Is lower if the number of unique values in a column is lower.

    Returns:
    event_log_sentence_list (list): List of event log sentences
    cluster_dict (dict): Dictionary with cluster information for each numeric columns.
    attribute_dtype_mapping (dict): Dictionary with attribute data types.
    starting_epoch_dist (list): List with mean and standard deviation of starting epochs.
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

    # Find noise multiplier
    print("Finding Optimal Noise Multiplier")
    noise_multiplier = find_noise_multiplier(epsilon, num_examples, batch_size, epochs)

    starting_epoch_dist = calculate_starting_epoch(df)
    time_between_events = calculate_time_between_events(df)
    df["time:timestamp"] = time_between_events
    attribute_dtype_mapping = get_attribute_dtype_mapping(df)
    # Epsilon is divided by 2 to ensure that the total epsilon is equal to the input epsilon since
    # DP-SDG and DP-Kmeans are performed sequentially
    df, cluster_dict = calculate_cluster_dp(df, max_clusters, epsilon)

    cols = ["concept:name", "time:timestamp"] + [
        col for col in df.columns if col not in ["concept:name", "time" ":timestamp"]
    ]
    df = df[cols]
    event_attribute_model = build_attribute_model(df)

    event_log_sentence_list = []
    for trace in df["case:concept:name"].unique():
        df_temp = df[df["case:concept:name"] == trace]
        trace_sentence_list = ["START==START"]
        for global_attribute in df_temp:
            if global_attribute.startswith("case:") and global_attribute != "case:concept:name":
                trace_sentence_list.append(global_attribute + "==" + str(df_temp[global_attribute].iloc[0]))
        for row in df_temp.iterrows():
            concept_name = row[1]["concept:name"]
            for col in df.columns:
                if str(row[1][col]) != "nan":
                    trace_sentence_list.append(concept_name + "==" + col + "==" + str(row[1][col]))
                else:
                    trace_sentence_list.append(concept_name + "==" + col + "==" + "nan")

        trace_sentence_list.append("END==END")
        event_log_sentence_list.append(trace_sentence_list)

    return (
        event_log_sentence_list,
        cluster_dict,
        attribute_dtype_mapping,
        starting_epoch_dist,
        num_examples,
        event_attribute_model,
        noise_multiplier,
    )
