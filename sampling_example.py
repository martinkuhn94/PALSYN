import pm4py
from PALSYN.synthesizer import DPEventLogSynthesizer
from PALSYN.postprocessing.log_postprocessing import clean_xes_file

# Load Model
palsyn_model = DPEventLogSynthesizer()
palsyn_model.load("experiments/models/LSTM_hospital_billing_u=32_e=inf_ep=10")

# Sample
event_log = palsyn_model.sample(sample_size=100000, batch_size=1000)
event_log_xes = pm4py.convert_to_event_log(event_log)

# Save as XES File
xes_filename = "LSTM_Hospital_Billing_u=32_e=inf_ep=10.xes"
pm4py.write_xes(event_log_xes, xes_filename)
clean_xes_file(xes_filename, xes_filename)

# Save as XSLX File for quick inspection
df = pm4py.convert_to_dataframe(event_log_xes)
df["time:timestamp"] = df["time:timestamp"].astype(str)
df.to_excel("LSTM_Hospital_Billing_u=32_e=inf_ep=10.xlsx", index=False)
