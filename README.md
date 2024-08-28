# PBLES (Private Bi-LSTM Event Log Synthesizer)

## Overview

PBLES (Private Bi-LSTM Event Log Synthesizer) is a tool designed to generate process-oriented synthetic healthcare data.
It addresses the privacy concerns in healthcare data sharing by integrating differential privacy techniques. 
By doing so, it can make it easier for researches to share synthetic data with stakeholders, 
facilitating AI and process mining research in healthcare.However, legal compliance, such as adherence to GDPR or 
other similar regulations, must be confirmed before sharing data, even if strong differential private guarantees are used.

## Features

- **Process-Oriented Data Generation:** Handles the complexity of healthcare data processes.
- **Multiple Perspectives:** Considers various perspectives of healthcare data, not just control-flow.
- **Differential Privacy:** Ensures privacy by incorporating differential privacy techniques.

## Installation

To install PBLES, first clone the repository:

```bash
git clone https://github.com/martinkuhn94/PBLES.git
```

Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model 
For the training of the model, the stacked layers are configured with 32, 16 and 8 LSTM units respectively, and an embedding dimension of 16. The model trains for 3 epochs with a batch size of 16. The number of clusters for numerical attributes is set to 10, and to speed up the training, only the top 50% quantile of traces by length are considered, in this example. The noise multiplier is set to 0.0, which means that the model is trained without differential privacy. To train the model with differential privacy, set the noise multiplier to a value greater than 0.0. The epsilon value can be retrieved after training the model.
```bash
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
```

### Sampling Event Logs 
To sample synthetic event logs, use the following example with a trained model can be used. The sample size is set to 160, and the batch size is set to 16. The synthetic event log is saved as a XES file.
Pretrained models can be found in the "models" folder.
```bash
import pm4py
from PBLES.event_log_dp_lstm import EventLogDpLstm

# Load Model
pbles_model = EventLogDpLstm()
pbles_model.load("models/DP_Bi_LSTM_e=inf_Sepsis_Case")

# Sample
event_log = pbles_model.sample(sample_size=160, batch_size=16)
event_log_xes = pm4py.convert_to_event_log(event_log)

# Save as XES File
xes_filename = "Synthetic_Sepsis_Case_Event_Log.xes"
pm4py.write_xes(event_log_xes, xes_filename)

# Save as XSLX File for quick inspection
df = pm4py.convert_to_dataframe(event_log_xes)
df['time:timestamp'] = df['time:timestamp'].astype(str)
df.to_excel("Synthetic_Sepsis_Case_Event_Log.xlsx", index=False)
```

## Future Work
Future work will focus on enhancing the algorithm and making it available on PyPI.

## Contribution

We welcome contributions from the community. If you have any suggestions or issues, please create a GitHub issue or a pull request. 


## License
This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

