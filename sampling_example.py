import pm4py
from PBLES.event_log_dp_lstm import EventLogDpLstm

# Load Model
lstmmodel = EventLogDpLstm()
lstmmodel.load("models/DP_Bi_LSTM_e=01_Sepsis_Case")

# Sample
event_log = lstmmodel.sample(160, 16, temperature=1.0)
event_log_xes = pm4py.convert_to_event_log(event_log)

# Save as XES File
xes_filename = "DP_Bi_LSTM_Sepsis_Case_event_log_e=01.xes"
pm4py.write_xes(event_log_xes, xes_filename)
