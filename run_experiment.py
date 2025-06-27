import os
import pm4py
import time
import gc
import pandas as pd  # Import pandas for data handling and Excel export
from PALSYN.synthesizer import DPEventLogSynthesizer
from PALSYN.postprocessing.log_postprocessing import clean_xes_file


def process_models(
        models_directory: str,
        output_directory: str,
        sample_size: int = 10000,
        batch_size: int = 250,
        runs: int = 1,
) -> None:
    """
    Loads models from a specified directory (assuming models are folders),
    samples event logs from each model 'runs' times, and saves the generated logs
    to an output directory. The synthesizer and model are re-instantiated
    and re-loaded for each run to ensure a clean state and prevent memory accumulation.
    Execution times for each sampling run are saved to an Excel file.

    Args:
        models_directory (str): The path to the directory containing the model folders.
        output_directory (str): The path to the directory where
                                the sampled event logs will be saved.
        sample_size (int): The number of events to sample for each run.
        batch_size (int): The batch size to use during the sampling process.
        runs (int): The number of event logs to sample for each model.
    """
    print(f"Attempting to create output directory: {output_directory}")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")
    else:
        print(f"Output directory already exists: {output_directory}")

    print(f"Checking models in directory: {models_directory}")
    if not os.path.exists(models_directory):
        print(f"ERROR: Models directory not found at: {models_directory}")
        return
    if not os.path.isdir(models_directory):
        print(f"ERROR: Path is not a directory: {models_directory}")
        return

    # List to store results for the Excel file
    execution_times_data = []

    found_models = False
    for model_name in os.listdir(models_directory):
        model_path = os.path.join(models_directory, model_name)
        print(f"Found item: {model_name} at path: {model_path}")

        if os.path.isdir(model_path):
            found_models = True
            print(f"Processing model folder: {model_name}")

            # Create a subfolder for each model within the output directory
            output_model_dir = os.path.join(output_directory, model_name)
            if not os.path.exists(output_model_dir):
                os.makedirs(output_model_dir)
                print(f"Created model specific output directory: {output_model_dir}")

            # Loop 'runs' times to sample multiple event logs for the current model
            for i in range(runs):
                print(f"  Starting run {i + 1}/{runs} for model: {model_name}")
                start_time = time.time()  # Start timing for this run
                try:
                    palsyn_model = DPEventLogSynthesizer()
                    print(f"  Loading model for run {i + 1}: {model_path}")
                    palsyn_model.load(model_path)

                    event_log = palsyn_model.sample(
                        sample_size=sample_size, batch_size=batch_size
                    )
                    event_log_xes = pm4py.convert_to_event_log(event_log)

                    unix_timestamp = int(time.time())
                    xes_filename = os.path.join(
                        output_model_dir, f"{model_name}_{unix_timestamp}.xes"
                    )

                    pm4py.write_xes(event_log_xes, xes_filename)
                    clean_xes_file(xes_filename, xes_filename)
                    print(
                        f"  Successfully processed and saved log for {model_name} "
                        f"(run {i + 1}) as {os.path.basename(xes_filename)}"
                    )

                    del palsyn_model
                    gc.collect()

                except Exception as e:
                    print(f"Error during run {i + 1} for model {model_name}: {e}")
                    # If an error occurs, record 0 for time or handle as needed
                    execution_time = 0.0
                else:  # This block executes if no exception occurred
                    end_time = time.time()  # End timing for this run
                    execution_time = end_time - start_time
                    print(f"  Execution time for run {i + 1}: {execution_time:.2f} seconds")

                # Store the results for Excel export
                execution_times_data.append(
                    {
                        "Model Name": model_name,
                        "Run Number": i + 1,
                        "Sample Size": sample_size,
                        "Batch Size": batch_size,
                        "Execution Time (s)": execution_time,
                        "Timestamp (XES)": unix_timestamp,
                        "XES File": os.path.basename(xes_filename) if 'xes_filename' in locals() else 'Error',
                    }
                )
        else:
            print(f"Skipping {model_name} as it is not a directory (it might be a file).")

    if not found_models:
        print(f"No valid model directories were found in: {models_directory}")

    # After all models and runs are processed, save execution times to Excel
    if execution_times_data:
        df_results = pd.DataFrame(execution_times_data)
        excel_timestamp = int(time.time())
        excel_filename = os.path.join(
            output_directory, f"sampling_performance_{excel_timestamp}.xlsx"
        )
        try:
            df_results.to_excel(excel_filename, index=False)
            print(f"\nExecution times saved to: {excel_filename}")
        except Exception as e:
            print(f"ERROR: Could not save execution times to Excel file: {e}")
    else:
        print("\nNo sampling data to save to Excel.")


if __name__ == "__main__":
    models_path = "models"
    output_path = "output_event_logs"
    sampling_size = 500
    batch_size = 128
    number_of_runs = 2  # Define how many event logs to sample per model

    process_models(
        models_path, output_path, sampling_size, batch_size, number_of_runs
    )