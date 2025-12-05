from __future__ import annotations

import datetime
import sys
from collections.abc import Sequence
from xml.etree import ElementTree as StdlibET

import numpy as np
import pandas as pd
from defusedxml import ElementTree as SafeET
from scipy.stats import norm

XES_NAMESPACE = "http://www.xes-standard.org/"
NS = {"": XES_NAMESPACE}
NA_VALUES = {
    "",
    "NA",
    "nan",
    "NaN",
    "null",
    "NULL",
    "<NA>",
    "NaT",
    "&lt;NA&gt;",
    "&lt;nan&gt;",
    "&lt;NaN&gt;",
    "&lt;null&gt;",
    "&lt;NULL&gt;",
    "&lt;NaT&gt;",
}


def clean_xes_file(xml_file: str, output_file: str) -> None:
    """
    Clean XES file by removing empty strings, NA values, and HTML-encoded NA strings.

    Parameters:
    xml_file (str): Path to input XES file
    output_file (str): Path to output cleaned XES file
    """
    tree = SafeET.parse(xml_file)
    root = tree.getroot()
    StdlibET.register_namespace("", XES_NAMESPACE)

    for event in root.findall(".//event", NS):
        to_remove = []
        for elem in event:
            value = elem.get("value", "").strip()
            if value.upper() in {x.upper() for x in NA_VALUES}:
                to_remove.append(elem)

        for elem in to_remove:
            event.remove(elem)

    tree.write(output_file, encoding="utf-8", xml_declaration=True)


def generate_df(
    synthetic_event_log_sentences: Sequence[Sequence[str]],
    cluster_dict: dict[str, Sequence[float]],
    dict_dtypes: dict[str, dict[str, str]],
    start_epoch: Sequence[float],
) -> pd.DataFrame:
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
    print("Creating DF-Event Log from synthetic Data")
    transformed_sentences = transform_sentences(
        synthetic_event_log_sentences, cluster_dict, dict_dtypes, start_epoch
    )
    df = create_dataframe_from_sentences(transformed_sentences, dict_dtypes)
    df = reorder_and_sort_df(df)

    return df


def reorder_and_sort_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort and re-order standard process columns if they exist.

    The DataFrame is sorted by ``case:concept:name`` and ``time:timestamp`` when
    both columns are present. Those columns plus ``concept:name`` are then
    moved to the beginning in the canonical order.

    Parameters:
    df (pd.DataFrame): DataFrame to be reordered and sorted.

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


def create_start_epoch(start_epoch: Sequence[float]) -> datetime.datetime:
    """
    Sample a starting epoch constrained by the provided bounds.

    The helper draws from a normal distribution with the supplied mean and
    standard deviation and retries until the value lies within ``[min, max]``.

    Parameters:
    start_epoch (list[float]): ``[mean, std, min_bound, max_bound]``.

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


def transform_sentences(
    synthetic_event_log_sentences: Sequence[Sequence[str]],
    cluster_dict: dict[str, Sequence[float]],
    dict_dtypes: dict[str, dict[str, str]],
    start_epoch: Sequence[float],
) -> list[list[str]]:
    """
    Convert tokenized sentences into enriched event-level representations.

    Each word is rewritten based on the column metadata and the cluster
    distributions while timestamps are advanced relative to ``start_epoch``.

    Parameters:
    synthetic_event_log_sentences: Tokenized synthetic sentences.
    cluster_dict: Dictionary describing generated attribute distributions.
    dict_dtypes: Attribute datatype configuration from YAML.
    start_epoch: Statistical bounds for timestamp generation.

    Returns:
    list: Processed synthetic event log sentences.
    """
    transformed_sentences = []
    for sentence, case_id in zip(
        synthetic_event_log_sentences, range(len(synthetic_event_log_sentences))
    ):
        print(
            "\r"
            + "Converting into Event Log "
            + str(round((case_id + 1) / len(synthetic_event_log_sentences) * 100, 1))
            + "% Complete",
            end="",
        )
        sys.stdout.flush()

        temp_sentence = [
            "case:concept:name==" + str(datetime.datetime.now().timestamp()).replace(".", "")
        ]
        epoch = create_start_epoch(start_epoch)
        for word in sentence:
            temp_sentence, epoch = process_word(
                word, temp_sentence, dict_dtypes, cluster_dict, epoch
            )

        transformed_sentences.append(temp_sentence)

    print("\n")

    return transformed_sentences


def process_word(
    word: str,
    temp_sentence: list[str],
    dict_dtypes: dict[str, dict[str, str]],
    cluster_dict: dict[str, Sequence[float]],
    epoch: datetime.datetime,
) -> tuple[list[str], datetime.datetime]:
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

    dtype_mapping = dict_dtypes["attribute_datatypes"]
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
            previous_timestamp = temp_sentence[-1].split("==")[1]
            recovered_timestamp = datetime.datetime.strptime(
                previous_timestamp, "%Y-%m-%dT%H:%M:%S.%f+00:00"
            )
            recovered_timestamp = recovered_timestamp + datetime.timedelta(seconds=1)
            timestamp_string = recovered_timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
        temp_sentence.append(f"time:timestamp=={timestamp_string}")

    return temp_sentence, epoch


def create_dataframe_from_sentences(
    transformed_sentences: Sequence[Sequence[str]], dict_dtypes: dict[str, dict[str, str]]
) -> pd.DataFrame:
    """
    Build a pandas DataFrame from transformed synthetic sentences.

    The parser reconstructs case-level dictionaries per trace, applies the
    configured dtypes, sorts traces chronologically, and interpolates missing
    timestamps.

    Parameters:
    transformed_sentences: Enriched synthetic sentences.
    dict_dtypes: Dictionary of attribute dtypes (under ``attribute_datatypes``).

    Returns:
    pd.DataFrame: DataFrame created from the synthetic sentences.
    """
    parsed_data = []
    removed_traces = 0

    for idx, sentence in enumerate(transformed_sentences):
        try:
            case_dict = {
                word.split("==")[0]: word.split("==")[1]
                for word in sentence
                if word.split("==")[0].startswith("case:")
            }
            if "case:concept:name" not in case_dict:
                case_dict["case:concept:name"] = f"case_{idx}"
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

    dtype_mapping = dict_dtypes["attribute_datatypes"]

    for key, value in dtype_mapping.items():
        if key in df.columns:
            df[key] = convert_column_dtype(df[key], value)

    if "time:timestamp" not in df.columns:
        df["time:timestamp"] = [
            pd.Timestamp("2000-01-01").strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
        ] * len(df)
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
            col.replace(["", "nan", "NaN", "NULL", "null"], np.nan), errors="coerce"
        ).astype("Int64"),
        "float": lambda col: col.astype(float) if col.name != "time:timestamp" else col.astype(str),
        "float64": lambda col: col.astype(float)
        if col.name != "time:timestamp"
        else col.astype(str),
        "boolean": lambda col: col.astype(bool),
        "date": lambda col: col.astype(str),
        "string": lambda col: col.astype(str),
        "object": lambda col: col.astype(str),
    }

    converter = type_converters.get(dtype)
    return converter(column) if converter else column
