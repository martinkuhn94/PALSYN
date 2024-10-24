import pm4py
from PBLES.event_log_dp_lstm import EventLogDpLstm
import tensorflow as tf

# Check for GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# Read Event Log
# Road_Traffic_Fine_Management_Process.xes
#path = "example_logs/Road_Traffic_Fine_Management_Process_short.xes"
#path = "example_logs/Hospital Billing - Event Log_short.xes"
path = "example_logs/BPI Challenge 2017_short.xes"
event_log = pm4py.read_xes(path)

# Train Model
bi_lstm_model = EventLogDpLstm(embedding_output_dims=32, epochs=1, batch_size=512,
                                          max_clusters=5, dropout=0.0, trace_quantile=0.9, epsilon=10,
                                          l2_norm_clip=1.0, method="GRU", units_per_layer=[16, 8])

bi_lstm_model.fit(event_log)
bi_lstm_model.save_model("models/DP-GRU_BPI_Challenge_2017_u=32_e=10")

# Print Epsilon to verify Privacy Guarantees
print(bi_lstm_model.epsilon)
