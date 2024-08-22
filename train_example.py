import pm4py
from PBLES.event_log_dp_lstm import EventLogDpLstm

# Read Event Log
path = "Sepsis_Cases_Event_Log.xes"
event_log = pm4py.read_xes(path)

# Train Model
bi_lstm_model = EventLogDpLstm(lstm_units=64, embedding_output_dims=16, epochs=3, batch_size=16,
                               max_clusters=25, dropout=0.0, trace_quantile=0.5, noise_multiplier=1.1,
                               l2_norm_clip=1.5)

bi_lstm_model.fit(event_log)
bi_lstm_model.save_model("models/DP_Bi_LSTM_e=inf_Sepsis_Cases_Event_Log")

# Print Epsilon to verify Privacy Guarantees
print(bi_lstm_model.epsilon)
