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
To train the model, use the following example with a given event log:

```bash
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
bi_lstm_model.save_model("models/DP_Bi_LSTM_e=inf_Diabetes")

# Print Epsilon to verify Privacy Guarantees
print(bi_lstm_model.epsilon)
```

### Sampling Event Logs 
To sample synthetic event logs, use the following example with a trained model:

```bash
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
```

## Contribution

We welcome contributions from the community. If you have any suggestions or issues, please create a GitHub issue or a pull request. 


## License
This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

