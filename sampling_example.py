import pm4py
from PBLES.event_log_dp_lstm import EventLogDpLstm

# Load Model
pbles_model = EventLogDpLstm()
pbles_model.load("models/DP-GRU_Road_Fines_u=32_e=0.1")

# Sample
event_log = pbles_model.sample(sample_size=200, batch_size=16)
event_log_xes = pm4py.convert_to_event_log(event_log)

# Save as XES File
xes_filename = "Synthetic_Road_Fines.xes"
pm4py.write_xes(event_log_xes, xes_filename)

# Save as XSLX File for quick inspection
df = pm4py.convert_to_dataframe(event_log_xes)
df['time:timestamp'] = df['time:timestamp'].astype(str)
df.to_excel("Synthetic_Road_Fines.xlsx", index=False)
