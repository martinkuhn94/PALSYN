import numpy as np
import pm4py
import pandas as pd
from pm4py.statistics.variants.log import get as variants_module
from pm4py.algo.evaluation.earth_mover_distance import algorithm as emd_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator


def event_distribution(log):
    """Return the count of each event type in a PM4Py event log as a pandas Series.

    :param log: The PM4Py event log.
    :type log: pm4py.objects.log.log.EventLog
    :returns: The count of each event type in the event log.
    :rtype: pd.Series
    """
    df = pm4py.convert_to_dataframe(log)
    count_data = df['concept:name'].value_counts()

    return count_data


def calculate_trace_length_distribution(log):
    """Return the count of each trace length in a PM4Py event log as a pandas Series.

    :param log: The PM4Py event log.
    :type log: pm4py.objects.log.log.EventLog
    :returns: The count of each trace length in the event log.
    :rtype: pd.Series
    """
    df = pm4py.convert_to_dataframe(log)
    count_data = df['case:concept:name'].value_counts()
    # Get the count of each trace length
    count_data = count_data.value_counts()
    # Sort the series by the index
    count_data = count_data.sort_index()
    # Make index numeric
    count_data.index = pd.to_numeric(count_data.index)

    return count_data


def calc_hellinger(data1, data2):
    """
    Calculate Hellinger distance between two distributions or columns.

    Args:
        data1: First distribution/column (pandas Series or column)
        data2: Second distribution/column (pandas Series or column)

    Returns:
        float: Hellinger distance
    """
    # Convert data to value counts if they're columns
    if not isinstance(data1.index, pd.Index) or data1.index.dtype == 'int64':
        dist1 = data1.value_counts()
        dist2 = data2.value_counts()
    else:
        dist1 = data1
        dist2 = data2

    # Align distributions
    all_indices = sorted(set(dist1.index) | set(dist2.index))
    dist1_aligned = pd.Series(0, index=all_indices)
    dist2_aligned = pd.Series(0, index=all_indices)

    dist1_aligned[dist1.index] = dist1
    dist2_aligned[dist2.index] = dist2

    # Convert to probabilities
    p1 = dist1_aligned / dist1_aligned.sum()
    p2 = dist2_aligned / dist2_aligned.sum()

    return np.sqrt(np.sum((np.sqrt(p1.values) - np.sqrt(p2.values)) ** 2)) / np.sqrt(2)


def calculate_throughput_time(log):
    """Calculate the throughput time of a PM4Py event log. The throughput time is defined as the time between the
    first and the last event of a trace.

    :param log: The PM4Py event log.
    :type log: pm4py.objects.log.log.EventLog
    :returns: The throughput time of the event log.
    :rtype: float
    """
    # Remove all timestamps that are NaN
    df = pm4py.convert_to_dataframe(log)
    df = df.dropna(subset=['time:timestamp'])
    # Transform df back to log
    log = pm4py.convert_to_event_log(df)

    all_case_durations = pm4py.get_all_case_durations(log)

    return all_case_durations


def calculate_fitness(alignments):
    """Calculate the fitness of an event log according to the calculated alignments.

    :param alignments: A list of PM4PY alignments.
    :type alignments: List[object]
    :return: The average fitness of the alignments.
    :rtype: float
    """
    fitness_values = [alignment.get("fitness") for alignment in alignments]
    fitness = sum(fitness_values) / len(fitness_values)

    return fitness


def calculate_earth_mover_distance(real_log, synthetic_log):
    """Calculate the earth mover distance between two event logs. The earth mover distance is defined as the
    minimum cost of turning one distribution into the other.

    :param real_log: The real event log.
    :type real_log: pm4py.objects.log.log.EventLog
    :param synthetic_log: The synthetic event log.
    :type synthetic_log: pm4py.objects.log.log.EventLog
    :return: The earth mover distance between the two event logs.
    :rtype: float
    """
    real_language = variants_module.get_language(real_log)
    synthetic_language = variants_module.get_language(synthetic_log)
    earth_mover_distance = emd_evaluator.apply(synthetic_language, real_language)

    return earth_mover_distance


def calculate_petri_nets(log, threshold):
    """Discover Petri nets using inductive and heuristic mining algorithms.

    :param threshold:
    :param log: A process event log.
    :type log: pm4py.objects.log.log.EventLog
    :return: A dictionary containing the discovered Petri nets, initial markings, and final markings.
    :rtype: dict
    """
    net_inductive, initial_marking_inductive, final_marking_inductive = pm4py.discover_petri_net_inductive(log)
    net_heuristics, initial_marking_heuristics, final_marking_heuristics = \
        pm4py.discover_petri_net_heuristics(log, dependency_threshold=threshold)
    petri_net_list = [
                      [net_inductive, initial_marking_inductive, final_marking_inductive],
                      [net_heuristics, initial_marking_heuristics, final_marking_heuristics]]
    petri_net_list_names = ["Inductive",
                            "Heuristics"]
    petri_net_dict = dict(zip(petri_net_list_names, petri_net_list))
    return petri_net_dict


def compare_logs(real_event_log, synthetic_event_log, threshold):
    """Compare the fitness of a real event Log with a synthetic event log. In this case the Petri Nets discovered
    from the real even log are used to calculate the alignments of the synthetic event log.

    :param real_event_log: The real event log.
    :type real_event_log: pm4py.objects.log.log.EventLog
    :param synthetic_event_log: The synthetic event log.
    :type synthetic_event_log: pm4py.objects.log.log.EventLog
    :param threshold: The threshold for the heuristic mining algorithm.
    :type threshold: float
    :return: Dictionary containing results for both real and synthetic data with prefixed keys
    """
    petri_net_dict = calculate_petri_nets(real_event_log, threshold)
    petri_net_dict_synth = calculate_petri_nets(synthetic_event_log, threshold)

    results = {}

    # Calculate metrics for real event log
    for key, petri_net in petri_net_dict.items():
        alignments = pm4py.conformance_diagnostics_alignments(real_event_log, petri_net[0], petri_net[1], petri_net[2])
        prec = pm4py.precision_alignments(real_event_log, petri_net[0], petri_net[1], petri_net[2])
        gen = generalization_evaluator.apply(real_event_log, petri_net[0], petri_net[1], petri_net[2])
        fitness = calculate_fitness(alignments)
        simp = simplicity_evaluator.apply(petri_net[0])

        results[f"real_{key}_Fitness"] = fitness
        results[f"real_{key}_Precision"] = prec
        results[f"real_{key}_Generalization"] = gen
        results[f"real_{key}_Simplicity"] = simp

    # Calculate metrics for synthetic event log
    for key, petri_net in petri_net_dict.items():
        alignments = pm4py.conformance_diagnostics_alignments(synthetic_event_log, petri_net[0], petri_net[1],
                                                              petri_net[2])
        prec = pm4py.precision_alignments(synthetic_event_log, petri_net[0], petri_net[1], petri_net[2])
        gen = generalization_evaluator.apply(synthetic_event_log, petri_net[0], petri_net[1], petri_net[2])
        fitness = calculate_fitness(alignments)
        simp = simplicity_evaluator.apply(petri_net_dict_synth[key][0])

        results[f"synthetic_{key}_Fitness"] = fitness
        results[f"synthetic_{key}_Precision"] = prec
        results[f"synthetic_{key}_Generalization"] = gen
        results[f"synthetic_{key}_Simplicity"] = simp

    return results
