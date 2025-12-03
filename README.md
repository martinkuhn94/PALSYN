# PALSYN (Private Autoregressive Log Synthesizer)

## Overview

PALSYN (Private Autoregressive Log Synthesizer) is a tool designed to generate process-oriented synthetic event logs.
It addresses the privacy concerns by integrating differential privacy. 
By doing so, it can make it easier for researches to share synthetic data with stakeholders, 
facilitating AI and process mining research. However, legal compliance, such as adherence to GDPR or 
other similar regulations, must be confirmed before sharing data, even if strong differential private guarantees are used. 

A detailed explanation of the algorithm and its workings can be found in our preprint:
[PALSYN: A Method for Synthetic Multi-Perspective Event Log Generation with Differential Private Guarantees](https://www.researchsquare.com/article/rs-6565248/v1)

> **Research tag v0.0.1-research-alpha**  
> This tag corresponds to the exact implementation used to generate the results reported in the paper
> _"PALSYN: A Method for Synthetic Multi-Perspective Event Log Generation with Differential Private Guarantees"_.
> Later updates may introduce new models or streamline the approach, so use this tag if you need the precise version of the code used in the publication.


## Features

- **Process-Oriented Data Generation:** Handles the complexity of process-oriented data (Event Logs).
- **Multiple Perspectives:** Considers various perspectives or attributes of the data, not just control-flow.
- **Differential Privacy:** Ensures privacy by incorporating differential privacy techniques.

## Installation
Choose the workflow that best matches your setup.

### 1. Minimal Runtime (requirements.txt)
```bash
git clone https://github.com/martinkuhn94/PALSYN.git
cd PALSYN
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Editable Install (`pip install .`)
```bash
git clone https://github.com/martinkuhn94/PALSYN.git
cd PALSYN
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install .
```

### 3. Development Environment (`pip install -e .[dev]`)
```bash
git clone https://github.com/martinkuhn94/PALSYN.git
cd PALSYN
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .[dev]
```
This third option installs tooling such as Ruff, mypy, pytest, coverage, and type stubs that are referenced in `pyproject.toml`.

## Usage

### Training the Model
The example below mirrors `train_example.py`. It trains a Bi-LSTM with 128-dimensional embeddings, two recurrent layers (32/16 units), dropout, and a 90th percentile trace filter.

```python
import pm4py
from PALSYN.synthesizer import DPEventLogSynthesizer

xes_file_path = "example_logs/Road_Traffic_Fine_Management_Process_short.xes"
event_log = pm4py.read_xes(xes_file_path)

palsyn_model = DPEventLogSynthesizer(
    embedding_output_dims=128,
    epochs=5,
    batch_size=128,
    max_clusters=15,
    dropout=0.3,
    trace_quantile=0.9,
    l2_norm_clip=1.0,
    method="Bi-LSTM",
    units_per_layer=[32, 16],
)

palsyn_model.fit(event_log)
palsyn_model.save_model("models/Bi-LSTM_Road_Fines_u=32_e=inf")
```

### Sampling Event Logs
After training or loading a saved model, sample synthetic traces and export them to XES/Excel as shown in `sampling_example.py`.

```python
import pm4py

from PALSYN.postprocessing.log_postprocessing import clean_xes_file
from PALSYN.synthesizer import DPEventLogSynthesizer

palsyn_model = DPEventLogSynthesizer()
palsyn_model.load("models/Bi-LSTM_Road_Fines_u=32_e=inf")

event_log = palsyn_model.sample(sample_size=5600, batch_size=100)
event_log_xes = pm4py.convert_to_event_log(event_log)

xes_filename = "road_fines_e=inf.xes"
pm4py.write_xes(event_log_xes, xes_filename)
clean_xes_file(xes_filename, xes_filename)

df = pm4py.convert_to_dataframe(event_log_xes)
df["time:timestamp"] = df["time:timestamp"].astype(str)
df.to_excel("road_fines_e=inf.xlsx", index=False)
```

## Future Work
Future work will focus on enhancing the algorithm and making it available on PyPI.

## Contribution

We welcome contributions from the community. If you have any suggestions or issues, please create a GitHub issue or a pull request. 


## License
This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details. 



## Funding 
This research is funded by the German Federal Ministry of Education and Research (BMBF) and NextGenerationEU (European Union) in the project KI-AIM under the funding code 16KISA115K.

