import datetime
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
import xml.etree.ElementTree as ET

XES_NAMESPACE = 'http://www.xes-standard.org/'
NS = {'': XES_NAMESPACE}
NA_VALUES = {
    '', 'NA', 'nan', 'NaN', 'null', 'NULL', '<NA>', 'NaT',
    '&lt;NA&gt;', '&lt;nan&gt;', '&lt;NaN&gt;', '&lt;null&gt;',
    '&lt;NULL&gt;', '&lt;NA&gt;', '&lt;NaT&gt;'
}


def clean_xes_file(xml_file, output_file):
    """
    Clean XES file by removing empty strings, NA values, and HTML-encoded NA strings.

    Parameters:
    xml_file (str): Path to input XES file
    output_file (str): Path to output cleaned XES file
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    ET.register_namespace('', XES_NAMESPACE)

    for event in root.findall('.//event', NS):
        to_remove = []
        for elem in event:
            value = elem.get('value', '').strip()
            if value.upper() in {x.upper() for x in NA_VALUES}:
                to_remove.append(elem)

        for elem in to_remove:
            event.remove(elem)

    tree.write(output_file, encoding='utf-8', xml_declaration=True)


def generate_df(synthetic_event_log_sentences, cluster_dict, dict_dtypes, start_epoch) -> pd.DataFrame:
    """
    Generate a DataFrame from synthetic event log sentences.

    Parameters:
    synthetic_event_log_sentences: List of synthetic event log sentences.
    cluster_dict: Dictionary of cluster information.
    dict_dtypes: Dictionary of data types.
    start_epoch: List containing start epoch information.
    event_attribute_dict: Dictionary containing event attribute mappings.

    Returns:
    pd.DataFrame: Generated DataFrame.
    """
    transformed_sentences = transform_sentences(synthetic_event_log_sentences, cluster_dict, dict_dtypes, start_epoch)
    df = create_dataframe_from_sentences(transformed_sentences, dict_dtypes)
    df = reorder_and_sort_df(df)

    return df


def reorder_and_sort_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort the DataFrame by 'case:concept:name' and 'time:timestamp' if these columns are present.
    Make 'case:concept:name' the first column, 'concept:name' the second column, and 'time:timestamp' the third.

    Parameters:
    df (pd.DataFrame): Input DataFrame to be reordered and sorted.

    Returns:
    pd.DataFrame: Reordered and sorted DataFrame.
    """
    if "case:concept:name" in df.columns and "time:timestamp" in df.columns:
        df.sort_values(by=["case:concept:name", "time:timestamp"], inplace=True)

    columns_order = []
    if "case:concept:name" in df.columns:
        columns_order.append("case:concept:name")
    if "concept:name" in df.columns:
        columns_order.append("concept:name")
    if "time:timestamp" in df.columns:
        columns_order.append("time:timestamp")

    other_columns = [col for col in df.columns if col not in columns_order]
    df = df[columns_order + other_columns]

    return df


def create_start_epoch(start_epoch: list[float]) -> datetime.datetime:
    """
    Create a start epoch for the synthetic event log generation. The start epoch is generated using a normal
    distribution with the mean and standard deviation specified in the start_epoch list. The generated epoch is
    then checked against the min and max bounds. If the value is out of bounds, it is regenerated.

    Parameters:
    start_epoch (list[float]): List containing: [mean, standard deviation, min bound, max bound]

    Returns:
    datetime.datetime: Start epoch as a datetime object.
    """
    mean, std, min_bound, max_bound = start_epoch
    epoch_dist = norm(loc=mean, scale=std)

    while True:
        epoch_value = epoch_dist.rvs(1)[0]

        if min_bound <= epoch_value <= max_bound:
            break

    epoch = datetime.datetime.fromtimestamp(epoch_value)
    return epoch


def transform_sentences(synthetic_event_log_sentences, cluster_dict, dict_dtypes, start_epoch) -> list[list[str]]:
    """
    Transform synthetic event log sentences by processing each word in the sentence and updating the temporary sentence.

    Parameters:
    synthetic_event_log_sentences: List of synthetic event log sentences.
    cluster_dict: Dictionary of cluster information.
    dict_dtypes: Dictionary of data types.
    start_epoch: List containing start epoch information.

    Returns:
    list: List of transformed synthetic event log sentences.
    """
    transformed_sentences = []
    for sentence, case_id in zip(synthetic_event_log_sentences, range(len(synthetic_event_log_sentences))):
        temp_sentence = ["case:concept:name==" + str(datetime.datetime.now().timestamp()).replace(".", "")]
        epoch = create_start_epoch(start_epoch)
        for word in sentence:
            temp_sentence, epoch = process_word(word, temp_sentence, dict_dtypes, cluster_dict, epoch)

        transformed_sentences.append(temp_sentence)

    return transformed_sentences


def process_word(word, temp_sentence, dict_dtypes, cluster_dict, epoch):
    """
    Process a word in the sentence and update the temporary sentence list.

    Parameters:
    word: The word to process
    temp_sentence: The temporary sentence list to update
    dict_dtypes: Dictionary of data types from YAML
    cluster_dict: Dictionary of cluster information
    epoch: Current epoch time

    Returns:
    tuple: (Updated temporary sentence list, Updated epoch)
    """
    parts = word.split("==")
    if len(parts) == 2:
        key, value = parts
    else:
        key = parts[0]
        value = "0"

    dtype_mapping = dict_dtypes['attribute_datatypes']
    if key in dtype_mapping and key != "time:timestamp":
        if value in cluster_dict:
            generation_input = cluster_dict[value]
            dist = norm(loc=generation_input[2], scale=generation_input[3])
            value = dist.rvs(1)[0]
            value = round(value, 5) if dtype_mapping[key] in ["float", "float64"] else round(value)
            temp_sentence.append(f"{key}=={value}")
        else:
            temp_sentence.append(word)
    elif key == "time:timestamp":
        generation_input = cluster_dict[value]
        dist = norm(loc=generation_input[2], scale=generation_input[3])
        value = abs(dist.rvs(1)[0])
        value = round(value)
        epoch = epoch + datetime.timedelta(seconds=value)
        timestamp_string = epoch.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
        if timestamp_string == "NaT":
            print("NaT was generated using previous Timestamp")
            timestamp_string = temp_sentence[-1].split("==")[1]
            timestamp_string = datetime.datetime.strptime(timestamp_string, "%Y-%m-%dT%H:%M:%S.%f+00:00")
            timestamp_string = timestamp_string + datetime.timedelta(seconds=1)
        temp_sentence.append(f"time:timestamp=={timestamp_string}")

    return temp_sentence, epoch


def create_dataframe_from_sentences(transformed_sentences, dict_dtypes) -> pd.DataFrame:
    """
    Create a DataFrame from the transformed synthetic event log sentences. The DataFrame is created by parsing the
    transformed sentences and converting the data types of the columns based on the dict_dtypes dictionary. The
    DataFrame is then sorted by 'case:concept:name' and 'time:timestamp' and the timestamps are interpolated and
    forward-filled.

    Parameters:
    transformed_sentences: List of transformed synthetic event log sentences.
    dict_dtypes: Dictionary of data types from YAML (nested under 'attribute_datatypes').

    Returns:
    pd.DataFrame: DataFrame created from the transformed synthetic event log sentences.
    """
    parsed_data = []
    removed_traces = 0

    for sentence in transformed_sentences:
        try:
            case_dict = {
                word.split("==")[0]: word.split("==")[1] for word in sentence if word.split("==")[0].startswith("case:")
            }
            event_indices = [i for i, s in enumerate(sentence) if s.startswith("concept:name")]
            event_indices.pop(0)
            events = np.split(sentence, event_indices)
            event_dict_list = []
            for event in events:
                event_dict = {word.split("==")[0]: word.split("==")[1] for word in event}
                event_dict.update(case_dict)
                event_dict_list.append(event_dict)
            parsed_data.append(event_dict_list)
        except Exception:
            removed_traces += 1

    df = pd.DataFrame()
    for case in parsed_data:
        df = pd.concat([df, pd.DataFrame(case)], ignore_index=True)

    dtype_mapping = dict_dtypes['attribute_datatypes']

    for key, value in dtype_mapping.items():
        if key in df.columns:
            df[key] = convert_column_dtype(df[key], value)

    if "time:timestamp" not in df.columns:
        df["time:timestamp"] = [pd.Timestamp("2000-01-01").strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")] * len(df)
    else:
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce")

    df.sort_values(by=["case:concept:name", "time:timestamp"], inplace=True)
    df["time:timestamp"] = df.groupby("case:concept:name")["time:timestamp"].transform(
        lambda x: x.interpolate(method="ffill")
    )
    df["time:timestamp"] = df.groupby("case:concept:name")["time:timestamp"].transform(
        lambda x: x.ffill() if pd.isna(x.iloc[0]) else x
    )

    df = df.replace("nan", "")

    return df


def convert_column_dtype(column: pd.Series, dtype: str) -> pd.Series:
    """
    Convert a pandas Series to specified dtype with proper NA handling.

    Parameters:
    column (pd.Series): Column to convert
    dtype (str): Target data type

    Returns:
    pd.Series: Converted column
    """
    type_converters = {
        "int64": lambda col: pd.to_numeric(
            col.replace(['', 'nan', 'NaN', 'NULL', 'null'], np.nan),
            errors='coerce'
        ).astype('Int64'),
        "float": lambda col: col.astype(float) if col.name != "time:timestamp" else col.astype(str),
        "float64": lambda col: col.astype(float) if col.name != "time:timestamp" else col.astype(str),
        "boolean": lambda col: col.astype(bool),
        "date": lambda col: col.astype(str),
        "string": lambda col: col.astype(str),
        "object": lambda col: col.astype(str)
    }

    converter = type_converters.get(dtype)
    return converter(column) if converter else column
