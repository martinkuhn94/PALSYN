import numpy as np
import pandas as pd
import pm4py
from sklearn.cluster import KMeans


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
        if 'case:concept:name' not in df or 'time:timestamp' not in df:
            raise ValueError("DataFrame must contain 'case:concept:name' and 'time:timestamp' columns")

        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])

        starting_epochs = df.sort_values(by='time:timestamp').groupby('case:concept:name')['time:timestamp'].first()

        starting_epoch_list = starting_epochs.astype(np.int64) // 10 ** 9

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
    if 'case:concept:name' not in df or 'time:timestamp' not in df:
        raise ValueError("DataFrame must contain 'case:concept:name' and 'time:timestamp' columns")

    try:
        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
    except Exception as e:
        raise ValueError(f"Error converting 'time:timestamp' to datetime: {e}")

    time_between_events = []

    for _, group in df.groupby('case:concept:name'):
        if len(group) < 2:
            time_between_events.append(0)
            continue

        time_diffs = group['time:timestamp'].diff().dt.total_seconds().copy()
        time_diffs.fillna(0, inplace=True)
        time_diffs.iloc[0] = 0
        time_between_events.extend(time_diffs)

    return time_between_events


def calculate_cluster(df, max_clusters):
    """
    Calculate clusters for each numeric column in a pandas DataFrame. The number of clusters is determined by the number
    of unique values in a column. If the number of unique values is smaller than the maximum number of clusters, the
    number of unique values is used as the number of clusters.

    Parameters:
    df: Pandas DataFrame.
    max_clusters: Number of maximum clusters.

    Returns:
    tuple: A tuple containing a Pandas DataFrame with cluster labels and a dictionary with cluster information.

    Raises:
    ValueError: If the input DataFrame is not valid or max_clusters is not a positive integer.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The input must be a pandas DataFrame")

    if not isinstance(max_clusters, int) or max_clusters <= 0:
        raise ValueError("max_clusters must be a positive integer")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_org = df.copy()
    df_cluster_list = []

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

        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        kmeans.fit(X)
        label = []
        for row in df.iterrows():
            if str(row[1][col]) != "nan":
                label_temp = kmeans.predict([[row[1][col]]])
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
            cluster_dict[cluster] = [min(dataframe_temp_cluster_values_np), max(dataframe_temp_cluster_values_np),
                                     dataframe_temp_cluster_values_np.mean(), dataframe_temp_cluster_values_np.std()]

    return df, cluster_dict


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


def get_event_attribute_dict(df: pd.DataFrame) -> dict:
    """
    Get the event attribute dictionary from an event log. The event attribute dictionary contains the unique values of
    each attribute in the event log.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame representing an event log, where columns are attributes.

    Returns:
    dict: A dictionary where keys are attribute names (column names) and values are their unique values.
    """
    concept_name_attribute_dict = {}
    concept_name_columns = df["concept:name"].unique()

    for concept_name in concept_name_columns:
        df_temp = df[df["concept:name"] == concept_name]
        df_temp = df_temp.dropna(axis=1, how='all')
        concept_name_attribute_dict[concept_name] = set(df_temp.columns)

    return concept_name_attribute_dict


def preprocess_event_log(log, max_clusters: int, trace_quantile: float):
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

    print("Number of traces: " + str(df['case:concept:name'].unique().size))

    trace_length = df.groupby('case:concept:name').size()
    trace_length_q = trace_length.quantile(trace_quantile)
    df = df.groupby('case:concept:name').filter(lambda x: len(x) <= trace_length_q)

    print("Number of traces after truncation: " + str(df['case:concept:name'].unique().size))
    df = df.sort_values(by=['case:concept:name', 'time:timestamp'])
    num_examples = len(df)

    starting_epoch_dist = calculate_starting_epoch(df)
    time_between_events = calculate_time_between_events(df)
    df['time:timestamp'] = time_between_events
    attribute_dtype_mapping = get_attribute_dtype_mapping(df)
    df, cluster_dict = calculate_cluster(df, max_clusters)

    cols = ['concept:name', 'time:timestamp'] + [col for col in df.columns if col not in ['concept:name', 'time'
                                                                                                          ':timestamp']]
    df = df[cols]

    event_log_sentence_list = []
    for trace in df['case:concept:name'].unique():
        df_temp = df[df['case:concept:name'] == trace]
        trace_sentence_list = ["START==START"]
        for global_attribute in df_temp:
            if global_attribute.startswith('case:') and global_attribute != 'case:concept:name':
                trace_sentence_list.append(global_attribute + "==" + str(df_temp[global_attribute].iloc[0]))
        for row in df_temp.iterrows():
            for col in df.columns:
                if not col.startswith('case:'):
                    if str(row[1][col]) != "nan":
                        trace_sentence_list.append(col + "==" + str(row[1][col]))

        trace_sentence_list.append("END==END")
        event_log_sentence_list.append(trace_sentence_list)

    event_attribute_dict = get_event_attribute_dict(df)
    print(event_attribute_dict)

    return (event_log_sentence_list, cluster_dict, attribute_dtype_mapping, starting_epoch_dist, event_attribute_dict,
            num_examples)
