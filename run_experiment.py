import os
import pm4py
import time
from PALSYN.synthesizer import DPEventLogSynthesizer
from PALSYN.postprocessing.log_postprocessing import clean_xes_file


def process_models(
    models_directory: str,
    output_directory: str,
    sample_size: int = 10000,
    batch_size: int = 250,
    runs: int = 1,  # Added 'runs' parameter with a default value of 1
) -> None:
    """
    Loads models from a specified directory (assuming models are folders),
    samples event logs from each model 'runs' times, and saves the generated logs
    to an output directory.

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

    palsyn_model = DPEventLogSynthesizer()

    print(f"Checking models in directory: {models_directory}")
    if not os.path.exists(models_directory):
        print(f"ERROR: Models directory not found at: {models_directory}")
        return
    if not os.path.isdir(models_directory):
        print(f"ERROR: Path is not a directory: {models_directory}")
        return

    found_models = False
    for model_name in os.listdir(models_directory):
        model_path = os.path.join(models_directory, model_name)
        print(f"Found item: {model_name} at path: {model_path}")

        if os.path.isdir(model_path):
            found_models = True
            print(f"Loading model folder: {model_name}")
            try:
                palsyn_model.load(model_path)

                # Create a subfolder for each model within the output directory
                output_model_dir = os.path.join(output_directory, model_name)
                if not os.path.exists(output_model_dir):
                    os.makedirs(output_model_dir)
                    print(f"Created model specific output directory: {output_model_dir}")

                # Loop 'runs' times to sample multiple event logs for the current model
                for i in range(runs):
                    print(f"  Sampling run {i+1}/{runs} for model: {model_name}")
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
                        f"(run {i+1}) as {os.path.basename(xes_filename)}"
                    )
            except Exception as e:
                print(f"Error processing model {model_name}: {e}")
        else:
            print(f"Skipping {model_name} as it is not a directory (it might be a file).")

    if not found_models:
        print(f"No valid model directories were found in: {models_directory}")


if __name__ == "__main__":
    models_path = "models"
    output_path = "output_event_logs"
    sampling_size = 1000
    batch_size = 100
    number_of_runs = 3  # Define how many event logs to sample per model

    process_models(
        models_path, output_path, sampling_size, batch_size, number_of_runs
    )