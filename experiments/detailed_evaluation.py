from datetime import datetime
import os
import pandas as pd
import pm4py
from sdmetrics.single_column import KSComplement
from process_mining_eval_functions import (calculate_throughput_time,
                                           calculate_trace_length_distribution, calc_hellinger,
                                           compare_logs, save_descriptive_stats_to_yaml,
                                           create_process_metrics_df, calculate_petri_nets, save_petri_nets)

log_name = "Sepsis_Case"
epsilon = "e=0.1"

# Read Event Log
log_filename = "Sepsis_Cases_Event_Log.xes"
real_event_log_filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), "example_logs",  log_filename)
real_event_log = pm4py.read_xes(real_event_log_filename)

# Read synthetic event log (assuming it's already generated)
synthetic_log_filename = "LSTM_Sepsis Case_u=32_e=0.1_ep=10.xes"
synthetic_log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments", "synthetic_logs", synthetic_log_filename)
synthetic_event_log = pm4py.read_xes(synthetic_log_path)

# Convert logs to dataframes
df_real = pm4py.convert_to_dataframe(real_event_log)
df_synthetic = pm4py.convert_to_dataframe(synthetic_event_log)

# Pre-processing
df_real_for_attr = df_real.drop(columns=['time:timestamp', 'case:concept:name', 'concept:name'], axis=1)
df_synthetic_for_attr = df_synthetic.drop(columns=['time:timestamp', 'case:concept:name', 'concept:name'], axis=1)

# Make dataframes with only numeric/categorical columns
df_real_numeric = df_real_for_attr.select_dtypes(include=['int64', 'float64'])
df_synthetic_numeric = df_synthetic_for_attr.select_dtypes(include=['int64', 'float64'])
df_real_categorical = df_real_for_attr.select_dtypes(include=['object'])
df_synthetic_categorical = df_synthetic_for_attr.select_dtypes(include=['object'])

# Attribute Perspective Evaluation
results = {}
average_ks = []
for col in df_real_numeric.columns:
    if col in df_synthetic_numeric.columns and not (df_real_numeric[col].isna().all() or df_synthetic_numeric[col].isna().all()):
        data_real = df_real_numeric[col].dropna()
        data_synthetic = df_synthetic_numeric[col].dropna()
        ks_statistic = KSComplement.compute(real_data=data_real, synthetic_data=data_synthetic)
        average_ks.append(ks_statistic)

average_tv = []
for col in df_real_categorical.columns:
    if col in df_synthetic_categorical.columns and not (df_real_categorical[col].isna().all() or df_synthetic_categorical[col].isna().all()):
        data_real = df_real_categorical[col].dropna().astype(str)
        data_synthetic = df_synthetic_categorical[col].dropna().astype(str)
        tv_statistic = 1 - calc_hellinger(data_real, data_synthetic)
        average_tv.append(tv_statistic)

results["average_ks"] = sum(average_ks) / len(average_ks) if average_ks else None
results["average_tv"] = sum(average_tv) / len(average_tv) if average_tv else None

# Event-based metrics
data_real = df_real["concept:name"].dropna()
data_synthetic = df_synthetic["concept:name"].dropna()
results["tv_statistic_event_distribution"] = 1 - calc_hellinger(data_real, data_synthetic)

# Trace length distribution
trace_length_real = calculate_trace_length_distribution(real_event_log)
trace_length_synthetic = calculate_trace_length_distribution(synthetic_event_log)
results["hellinger_distance_trace_length_distribution"] = 1 - calc_hellinger(trace_length_real, trace_length_synthetic, input_type="distribution")

# Throughput time distribution
throughput_time_real = calculate_throughput_time(real_event_log)
throughput_time_synthetic = calculate_throughput_time(synthetic_event_log)
results["ks_statistic_throughput_time_distribution"] = KSComplement.compute(
    real_data=throughput_time_real,
    synthetic_data=throughput_time_synthetic
)


# Process Perspective Metrics
process_metrics = compare_logs(real_event_log, synthetic_event_log, threshold=0.5)
df_regular, df_transposed = create_process_metrics_df(process_metrics)


# Calculate overall average including all metrics
columns_to_average = [
    'average_ks',
    'average_tv',
    'tv_statistic_event_distribution',
    'hellinger_distance_trace_length_distribution',
    'ks_statistic_throughput_time_distribution',
    #'earth_mover_distance'
]

# Create DataFrame with results
df_results = pd.DataFrame([results])

# Create timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Create main results directory if it doesn't exist
results_dir = 'evaluation_results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Create subdirectory for this specific evaluation
eval_dir = os.path.join(results_dir, f'{log_name}_{epsilon}_{timestamp}')
if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)

# Save descriptive stats YAML files
yaml_path_real = os.path.join(eval_dir, 'descriptive_stats_real.yaml')
yaml_path_synthetic = os.path.join(eval_dir, 'descriptive_stats_synthetic.yaml')
save_descriptive_stats_to_yaml(df_real, yaml_path_real)
save_descriptive_stats_to_yaml(df_synthetic, yaml_path_synthetic)

# Save process metrics Excel file
process_metrics_path = os.path.join(eval_dir, 'process_metrics.xlsx')
df_transposed.to_excel(process_metrics_path, index=False)

# Save Petri Nets
# Save petri nets Real
petri_net_dict_real = calculate_petri_nets(real_event_log, threshold=0.5)
petri_net_dict_synth = calculate_petri_nets(synthetic_event_log, threshold=0.5)

# Save both real and synthetic Petri nets
save_petri_nets(petri_net_dict_real, eval_dir, f'{log_name}_{epsilon}_real')
save_petri_nets(petri_net_dict_synth, eval_dir, f'{log_name}_{epsilon}_synthetic')


# Save general metrics Excel file
metrics_path = os.path.join(eval_dir, 'evaluation_metrics.xlsx')
df_results.to_excel(metrics_path, index=False)




