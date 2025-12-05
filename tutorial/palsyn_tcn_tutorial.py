import os
from pathlib import Path

import pm4py

from PALSYN.postprocessing.log_postprocessing import clean_xes_file
from PALSYN.synthesizer import TCNSynthesizer

# Note: run `pip install -e .` from the repository root once so the local PALSYN package
# is installed in your environment before executing this tutorial script.

# Paths used throughout the tutorial.
LOG_PATH = Path("example_logs/Road_Traffic_Fine_Management_Process_short.xes")
TUTORIAL_ROOT = Path("tutorial")
MODEL_DIR = TUTORIAL_ROOT / "tcn_model"
CHECKPOINT_PATH = MODEL_DIR / "checkpoints" / "best.keras"

# Ensure output folders exist for checkpoints/models.
os.makedirs(CHECKPOINT_PATH.parent, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 1) Load the example Road Traffic Fine Management log.
event_log = pm4py.read_xes(str(LOG_PATH))

# 2) Configure the synthesizer. Each dict mirrors the BaseSynthesizer groups:
#    - pre_processing: controls clustering, trace trimming, and random seed.
#    - model: training hyperparameters + TCN architecture choices.
#    - dp_optimizer: privacy budget plus DP-SGD optimizer settings. Epsilon controls the
#      privacy budget: ~1-3 offers strong privacy, 5-10 is moderate, and values >=15 essentially
#      prioritize utility over privacy guarantees.
synthesizer = TCNSynthesizer(
    pre_processing={
        "max_clusters": 15,
        "trace_quantile": 0.9,
        "seed": 42,
    },
    model={
        "embedding_output_dims": 128,
        "epochs": 5,
        "batch_size": 128,
        "validation_split": 0.15,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "filters_per_layer": [64, 32],
        "dropout": 0.0,
        "kernel_size": 3,
        "activation": "relu",
        "padding": "causal",
        "dilation_base": 2,
    },
    dp_optimizer={
        "epsilon": 15.0,
        "learning_rate": 5e-4,
        "l2_norm_clip": 1.0,
    },
)

# 3) Train on the event log with the specified hyperparameters.
synthesizer.fit(event_log)

# 4) Persist the trained model weights plus preprocessing artifacts.
synthesizer.save_model(str(MODEL_DIR))

# 5) Reload the model from disk to demonstrate inference-only usage.
loaded_model = TCNSynthesizer.load(str(MODEL_DIR))

# 6) Sample 6000 synthetic traces (batch size controls sampler throughput).
synthetic_df = loaded_model.sample(sample_size=6000, batch_size=128)

# 7) Convert the DataFrame to an event log for export.
synthetic_event_log = pm4py.convert_to_event_log(synthetic_df)

# 8) Save as XES file and run the post-processing cleanup.
xes_filename = TUTORIAL_ROOT / "road_fines_tcn_e=15.xes"
pm4py.write_xes(synthetic_event_log, str(xes_filename))
clean_xes_file(str(xes_filename), str(xes_filename))

# 9) Save as XLSX file for quick inspection.
synthetic_event_df = pm4py.convert_to_dataframe(synthetic_event_log)
synthetic_event_df["time:timestamp"] = synthetic_event_df["time:timestamp"].astype(str)
synthetic_event_df.to_excel(TUTORIAL_ROOT / "road_fines_tcn_e=15.xlsx", index=False)
