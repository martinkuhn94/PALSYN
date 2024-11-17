import pm4py
from PBLES.event_log_dp_lstm import EventLogDpLstm
import tensorflow as tf

# Check for GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))


# Read Event Log
# Road_Traffic_Fine_Management_Process.xes
#path = "example_logs/Road_Traffic_Fine_Management_Process_short.xes"
#path = "example_logs/Hospital_Billing_Event Log.xes"
#path = "example_logs/BPI Challenge 2017_short.xes"
path = "example_logs/Sepsis_Cases_Event_Log.xes"

event_log = pm4py.read_xes(path)
#event_log = event_log[["case:concept:name", "concept:name", "time:timestamp"]]



# Train Model
bi_lstm_model = EventLogDpLstm(
    embedding_output_dims=128,
    epochs=5,
    batch_size=32,
    max_clusters=5,
    dropout=0.0,
    trace_quantile=0.75,
    epsilon=10,
    l2_norm_clip=1.0,
    method="GRU",
    units_per_layer=[32],
    num_attention_heads=2,
)

bi_lstm_model.fit(event_log)
bi_lstm_model.save_model("models/GRU_Sepsis_Case_u=32_e=1")

# Print Epsilon to verify Privacy Guarantees
print(bi_lstm_model.epsilon)


event_log = bi_lstm_model.sample(sample_size=1050, batch_size=32)
event_log_xes = pm4py.convert_to_event_log(event_log)

# Save as XES File
xes_filename = "sepsis_case.xes"
pm4py.write_xes(event_log_xes, xes_filename)

# Save as XSLX File for quick inspection
df = pm4py.convert_to_dataframe(event_log_xes)
df["time:timestamp"] = df["time:timestamp"].astype(str)
df.to_excel("sepsis_case.xlsx", index=False)