from datetime import datetime
import time
import os

import pandas as pd
import pm4py
from PALSYN.synthesizer import DPEventLogSynthesizer
from sdmetrics.single_column import TVComplement, KSComplement
from PALSYN.postprocessing.log_postprocessing import clean_xes_file
from process_mining_eval_functions import (calculate_throughput_time, \
                                           calculate_trace_length_distribution, calc_hellinger)


# To run this file you need to install the following packages:
# pip install pyemd
# pip install sdmetrics
# pip install openpyxl

# Read Event Log
log_filename = "Sepsis_Cases_Event_Log.xes"
real_event_log_filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), "example_logs", log_filename)
event_log_train = pm4py.read_xes(real_event_log_filename)

event_log_name = "Sepsis_Case"
method_array = ["LSTM"]
num_epochs = 10  # Total number of epochs to train
breakpoint_interval = 5  # Save and evaluate model every 10 epochs
units_per_layer_array = [32]
epsilon_array = [None, 1, 0.1]

# Sampling
sample_size = 10000
batch_size = 200

# Dataframe result array
df_result_array = []

# Loop through all combinations
for method in method_array:
    for units in units_per_layer_array:
        for epsilon in epsilon_array:
            # Create model name
            if epsilon is None:
                epsilon_str = "inf"
            else:
                epsilon_str = str(epsilon)

            # Initialize model once
            model = DPEventLogSynthesizer(
                embedding_output_dims=128,
                epochs=num_epochs,
                batch_size=128,
                max_clusters=10,
                dropout=0.0,
                trace_quantile=0.8,
                epsilon=epsilon,
                l2_norm_clip=1.0,
                method=method,
                units_per_layer=[units],
            )

            # Initialize model architecture
            model.initialize_model(event_log_train)

            # Train in intervals defined by breakpoint_interval
            for current_epoch in range(breakpoint_interval, num_epochs + breakpoint_interval, breakpoint_interval):
                results = {"method": method, "units": units, "epsilon": epsilon_str, "epochs": current_epoch}

                # Train for breakpoint_interval epochs
                start_time = time.time()
                print(f"Training epochs {current_epoch - breakpoint_interval} to {current_epoch}")
                model.train(epochs=breakpoint_interval)

                model_name = f"models/{method}_{event_log_name}_u={units}_e={epsilon_str}_ep={current_epoch}"
                model.save_model(model_name)
                print(f"Model saved at epoch {current_epoch}: {model_name}")

                # End timer for training time
                end_time = time.time()
                training_time = end_time - start_time
                results["training_time"] = training_time

                try:
                    # Sampling time
                    start_time = time.time()
                    event_log_sample = model.sample(sample_size=sample_size, batch_size=batch_size)
                    event_log_xes = pm4py.convert_to_event_log(event_log_sample)
                    end_time = time.time()
                    sampling_time = end_time - start_time
                    results["sampling_time"] = sampling_time

                    # Save as XES File
                    xes_filename = f"synthetic_logs/{method}_{event_log_name}_u={units}_e={epsilon_str}_ep={current_epoch}.xes"
                    pm4py.write_xes(event_log_xes, xes_filename)
                except:
                    continue

                # Transform XES
                clean_xes_file(xes_filename, xes_filename)

                # Load and Process Real Event Log
                real_event_log = pm4py.read_xes(real_event_log_filename)
                synthetic_event_log = pm4py.read_xes(xes_filename)
                df_real = pm4py.convert_to_dataframe(real_event_log)
                df_synthetic = pm4py.convert_to_dataframe(synthetic_event_log)

                # region DF pre-processing
                df_real = df_real.drop(columns=['time:timestamp', 'case:concept:name', 'concept:name'], axis=1)
                df_synthetic = df_synthetic.drop(columns=['time:timestamp', 'case:concept:name', 'concept:name'],
                                                 axis=1)

                # Make dataframe with only numeric columns
                df_real_numeric = df_real.select_dtypes(include=['int64', 'float64'])
                df_synthetic_numeric = df_synthetic.select_dtypes(include=['int64', 'float64'])

                # Make dataframe with only categorical columns
                df_real_categorical = df_real.select_dtypes(include=['object'])
                df_synthetic_categorical = df_synthetic.select_dtypes(include=['object'])
                # endregion

                # region Attribute Perspective Evaluation
                average_ks = []
                for col in df_real_numeric.columns:
                    if col not in df_synthetic_numeric.columns:
                        print(f"Skipping {col} - column not found in synthetic data")
                        continue

                    if df_real_numeric[col].isna().all() or df_synthetic_numeric[col].isna().all():
                        print(f"Skipping {col} - empty column detected")
                        continue

                    data_real = df_real_numeric[col].dropna()
                    data_synthetic = df_synthetic_numeric[col].dropna()
                    ks_statistic = KSComplement.compute(real_data=data_real, synthetic_data=data_synthetic)
                    print(f"{col} KS Statistic: {ks_statistic}", "Length Real: ", len(data_real),
                          "Length Synthetic: ", len(data_synthetic))
                    average_ks.append(ks_statistic)

                average_tv = []
                for col in df_real_categorical.columns:
                    if col not in df_synthetic_categorical.columns:
                        print(f"Skipping {col} - column not found in synthetic data")
                        continue

                    if df_real_categorical[col].isna().all() or df_synthetic_categorical[col].isna().all():
                        print(f"Skipping {col} - empty column detected")
                        continue

                    data_real = df_real_categorical[col].dropna().astype(str)
                    data_synthetic = df_synthetic_categorical[col].dropna().astype(str)
                    tv_statistic = TVComplement.compute(real_data=data_real, synthetic_data=data_synthetic)
                    print(f"{col} TV Statistic: {tv_statistic}", "Length Real: ", len(data_real),
                          "Length Synthetic: ", len(data_synthetic))
                    average_tv.append(tv_statistic)

                average_ks_value = sum(average_ks) / len(average_ks) if average_ks else None
                average_tv_value = sum(average_tv) / len(average_tv) if average_tv else None

                if average_ks_value:
                    print(f"Average KS Statistic: {average_ks_value}")
                if average_tv_value:
                    print(f"Average TV Statistic: {average_tv_value}")

                results["average_ks"] = average_ks_value
                results["average_tv"] = average_tv_value

                if average_ks and average_tv:
                    weighted_ks = sum(average_ks) / len(average_ks) * (
                            len(average_ks) / (len(average_ks) + len(average_tv)))
                    weighted_tv = sum(average_tv) / len(average_tv) * (
                            len(average_tv) / (len(average_ks) + len(average_tv)))
                    print(f"Combined Resemblance: {weighted_tv + weighted_ks}")
                # endregion

                # Reload original dataframes for event-based metrics
                df_real = pm4py.convert_to_dataframe(real_event_log)
                df_synthetic = pm4py.convert_to_dataframe(synthetic_event_log)

                # Calculate TV Statistic for events
                data_real = df_real["concept:name"].dropna()
                data_synthetic = df_synthetic["concept:name"].dropna()
                tv_statistic = TVComplement.compute(real_data=data_real, synthetic_data=data_synthetic)
                results["tv_statistic_event_distribution"] = tv_statistic
                print("TV Statistic for Event Distribution: ", tv_statistic)

                # Calculate trace length distribution
                trace_length_real = calculate_trace_length_distribution(real_event_log)
                trace_length_synthetic = calculate_trace_length_distribution(synthetic_event_log)
                tv_statistic = TVComplement.compute(real_data=trace_length_real, synthetic_data=trace_length_synthetic)
                results["tv_statistic_trace_length_distribution"] = tv_statistic
                print("TV Statistic for Trace Length Distribution: ", tv_statistic)

                hellinger_distance = calc_hellinger(trace_length_real, trace_length_synthetic)
                results["hellinger_distance_trace_length_distribution"] = hellinger_distance
                print("Hellinger Distance for Trace Length Distribution: ", hellinger_distance)

                # Calculate throughput time distribution
                throughput_time_real = calculate_throughput_time(real_event_log)
                throughput_time_synthetic = calculate_throughput_time(synthetic_event_log)
                ks_statistic = KSComplement.compute(real_data=throughput_time_real,
                                                    synthetic_data=throughput_time_synthetic)
                results["ks_statistic_throughput_time_distribution"] = ks_statistic
                print("KS Statistic for Throughput Time Distribution: ", ks_statistic)

                # Add results to df_result_array
                df_result_array.append(results)

# Save final results
df_results = pd.DataFrame(df_result_array)

# Calculate the average of the specified columns
columns_to_average = [
    'average_ks',
    'average_tv',
    'tv_statistic_event_distribution',
    'tv_statistic_trace_length_distribution',
    'ks_statistic_throughput_time_distribution'
]

df_results['Average'] = df_results[columns_to_average].mean(axis=1)

# Save to Excel
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f"evaluation_result_{event_log_name}_{timestamp}.xlsx"
df_results.to_excel(filename, index=False)