import pm4py
from PBLES.event_log_dp_lstm import EventLogDpLstm
from PBLES.postprocessing.log_postprocessing import clean_xes_file

# Load Model
pbles_model = EventLogDpLstm()
pbles_model.load("models/Bi-LSTM_Road_Fines_u=32_e=inf")

# Sample
event_log = pbles_model.sample(sample_size=1000, batch_size=50)
event_log_xes = pm4py.convert_to_event_log(event_log)

# Save as XES File
xes_filename = "road_fines_e=inf.xes"
pm4py.write_xes(event_log_xes, xes_filename)
clean_xes_file(xes_filename, xes_filename)

# Save as XSLX File for quick inspection
df = pm4py.convert_to_dataframe(event_log_xes)
df["time:timestamp"] = df["time:timestamp"].astype(str)
df.to_excel("road_fines_e=inf.xlsx", index=False)
