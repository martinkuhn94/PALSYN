import pm4py
from PBLES.event_log_dp_lstm import EventLogDpLstm

# Load Model
pbles_model = EventLogDpLstm()
pbles_model.load("models/GRU_Hospital_Billing_u=32_e=1")

# Sample
event_log = pbles_model.sample(sample_size=4006, batch_size=64)
event_log_xes = pm4py.convert_to_event_log(event_log)

# Save as XES File
xes_filename = "hospital_billing.xes"
pm4py.write_xes(event_log_xes, xes_filename)

# Save as XSLX File for quick inspection
df = pm4py.convert_to_dataframe(event_log_xes)
df["time:timestamp"] = df["time:timestamp"].astype(str)
df.to_excel("hospital_billing.xlsx", index=False)
