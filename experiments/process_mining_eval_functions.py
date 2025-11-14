from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pm4py
from pm4py.algo.evaluation.earth_mover_distance import algorithm as emd_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.statistics.variants.log import get as variants_module


def event_distribution(log: Any) -> pd.Series:
    """Return the count of each event type in a PM4Py event log as a pandas Series.

    :param log: The PM4Py event log.
    :type log: pm4py.objects.log.log.EventLog
    :returns: The count of each event type in the event log.
    :rtype: pd.Series
    """
    df = pm4py.convert_to_dataframe(log)
    count_data = df["concept:name"].value_counts()

    return count_data


def calculate_trace_length_distribution(log: Any) -> pd.Series:
    """Return the count of each trace length in a PM4Py event log as a pandas Series.

    :param log: The PM4Py event log.
    :type log: pm4py.objects.log.log.EventLog
    :returns: The count of each trace length in the event log.
    :rtype: pd.Series
    """
    df = pm4py.convert_to_dataframe(log)
    count_data = df["case:concept:name"].value_counts()
    # Get the count of each trace length
    count_data = count_data.value_counts()
    # Sort the series by the index
    count_data = count_data.sort_index()

    print(count_data)

    # Make index numeric
    # count_data.index = pd.to_numeric(count_data.index)

    return count_data


def calc_hellinger(real_data: Any, synthetic_data: Any, input_type: str = "column") -> float:
    """
    Calculate Hellinger distance between two distributions or columns.

    Args:
        data1: First distribution/column (pandas Series or column)
        data2: Second distribution/column (pandas Series or column)

    Returns:
        float: Hellinger distance
    """
    if input_type == "column":
        dist1 = real_data.value_counts()
        dist2 = synthetic_data.value_counts()
    else:
        dist1 = real_data
        dist2 = synthetic_data

    # Align distributions
    all_indices = sorted(set(dist1.index) | set(dist2.index))
    dist1_aligned = pd.Series(0, index=all_indices)
    dist2_aligned = pd.Series(0, index=all_indices)

    dist1_aligned[dist1.index] = dist1
    dist2_aligned[dist2.index] = dist2

    # Convert to probabilities
    p1 = dist1_aligned / dist1_aligned.sum()
    p2 = dist2_aligned / dist2_aligned.sum()

    hellinger = np.sqrt(np.sum((np.sqrt(p1.values) - np.sqrt(p2.values)) ** 2)) / np.sqrt(2.0)
    return float(hellinger)


def calculate_throughput_time(log: Any) -> list[float]:
    """Calculate per-trace throughput time (first to last timestamp)."""
    # Remove all timestamps that are NaN
    df = pm4py.convert_to_dataframe(log)
    df = df.dropna(subset=["time:timestamp"])
    # Transform df back to log
    log = pm4py.convert_to_event_log(df)

    all_case_durations = pm4py.get_all_case_durations(log)

    return [float(duration) for duration in all_case_durations]


def calculate_fitness(alignments: list[dict[str, Any]]) -> float:
    """Calculate the fitness of an event log according to the calculated alignments.

    :param alignments: A list of PM4PY alignments.
    :type alignments: List[object]
    :return: The average fitness of the alignments.
    :rtype: float
    """
    fitness_values: list[float] = []
    for alignment in alignments:
        raw_value = alignment.get("fitness")
        if raw_value is not None:
            fitness_values.append(float(raw_value))

    if not fitness_values:
        return 0.0

    fitness = sum(fitness_values) / len(fitness_values)

    return fitness


def calculate_earth_mover_distance(real_log: Any, synthetic_log: Any) -> float:
    """
    Compute the Earth Mover's Distance (EMD) between two event logs.

    :param real_log: The reference event log.
    :param synthetic_log: The generated event log.
    :return: EMD score describing the cost to transform one distribution.
    """
    real_language = variants_module.get_language(real_log)
    synthetic_language = variants_module.get_language(synthetic_log)
    earth_mover_distance = emd_evaluator.apply(synthetic_language, real_language)

    return float(earth_mover_distance)


def calculate_petri_nets(log: Any, threshold: float) -> dict[str, list[Any]]:
    """
    Discover Petri nets using inductive and heuristic mining algorithms.

    :param threshold: Dependency threshold for heuristic mining.
    :param log: Event log to analyze.
    :return: Mapping of miner name to (net, initial marking, final marking).
    """
    net_inductive, initial_marking_inductive, final_marking_inductive = (
        pm4py.discover_petri_net_inductive(log)
    )
    net_heuristics, initial_marking_heuristics, final_marking_heuristics = (
        pm4py.discover_petri_net_heuristics(log, dependency_threshold=threshold)
    )
    petri_net_list = [
        [net_inductive, initial_marking_inductive, final_marking_inductive],
        [net_heuristics, initial_marking_heuristics, final_marking_heuristics],
    ]
    petri_net_list_names = ["Inductive", "Heuristics"]
    petri_net_dict = dict(zip(petri_net_list_names, petri_net_list))
    return petri_net_dict


def compare_logs(
    real_event_log: Any, synthetic_event_log: Any, threshold: float
) -> dict[str, float]:
    """
    Compare real and synthetic logs across fitness, precision, generalization, and simplicity.

    :param real_event_log: Reference event log.
    :param synthetic_event_log: Generated event log.
    :param threshold: Dependency threshold for heuristic mining.
    :return: Dictionary containing metrics with ``real_``/``synthetic_`` prefixes.
    """
    petri_net_dict = calculate_petri_nets(real_event_log, threshold)
    petri_net_dict_synth = calculate_petri_nets(synthetic_event_log, threshold)

    results = {}

    # Calculate metrics for real event log
    for key, petri_net in petri_net_dict.items():
        alignments = pm4py.conformance_diagnostics_alignments(
            real_event_log, petri_net[0], petri_net[1], petri_net[2]
        )
        prec = float(
            pm4py.precision_alignments(real_event_log, petri_net[0], petri_net[1], petri_net[2])
        )
        gen = float(
            generalization_evaluator.apply(real_event_log, petri_net[0], petri_net[1], petri_net[2])
        )
        fitness = float(calculate_fitness(alignments))
        simp = float(simplicity_evaluator.apply(petri_net[0]))

        results[f"real_{key}_Fitness"] = fitness
        results[f"real_{key}_Precision"] = prec
        results[f"real_{key}_Generalization"] = gen
        results[f"real_{key}_Simplicity"] = simp

    # Calculate metrics for synthetic event log
    for key, petri_net in petri_net_dict.items():
        alignments = pm4py.conformance_diagnostics_alignments(
            synthetic_event_log, petri_net[0], petri_net[1], petri_net[2]
        )
        prec = float(
            pm4py.precision_alignments(
                synthetic_event_log, petri_net[0], petri_net[1], petri_net[2]
            )
        )
        gen = float(
            generalization_evaluator.apply(
                synthetic_event_log, petri_net[0], petri_net[1], petri_net[2]
            )
        )
        fitness = float(calculate_fitness(alignments))
        simp = float(simplicity_evaluator.apply(petri_net_dict_synth[key][0]))

        results[f"synthetic_{key}_Fitness"] = fitness
        results[f"synthetic_{key}_Precision"] = prec
        results[f"synthetic_{key}_Generalization"] = gen
        results[f"synthetic_{key}_Simplicity"] = simp

    return results
