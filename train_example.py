import pm4py
from PBLES.event_log_dp_lstm import EventLogDpLstm

# Read Event Log
path = "Sepsis_Cases_Event_Log.xes"
event_log = pm4py.read_xes(path)

# Train Model
pbles_model = EventLogDpLstm(lstm_units=32, embedding_output_dims=16, epochs=3, batch_size=16,
                               max_clusters=10, trace_quantile=0.5, noise_multiplier=0.0)

pbles_model.fit(event_log)
pbles_model.save_model("models/DP_Bi_LSTM_e=inf_Sepsis_Cases_Event_Log_test")

# Print Epsilon to verify Privacy Guarantees
print(pbles_model.epsilon)
