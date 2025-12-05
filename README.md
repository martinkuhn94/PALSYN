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
Choose the workflow that best matches your setup. The tutorials and CLI scripts assume the project is installed in editable mode so `PALSYN` can be imported from anywhere—run one of the options below before executing `tutorial/*.py`.

### 1. Standard Install (`pip install .`)
```bash
git clone https://github.com/martinkuhn94/PALSYN.git
cd PALSYN
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install .
```

### 2. Editable/Development Install (`pip install -e .[dev]`)
```bash
git clone https://github.com/martinkuhn94/PALSYN.git
cd PALSYN
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .[dev]
```
This option installs Ruff, mypy, pytest, coverage, and type stubs defined in `pyproject.toml`, making it ideal for contributors who want local linting and typing parity with CI.

## Usage

### Training the Model
The example below matches the configuration used in `tutorial/palsyn_lstm_tutorial.py`. It shows the grouped dictionaries (`pre_processing`, `model`, `dp_optimizer`) that control preprocessing, architecture, and differential-privacy settings.

```python
import pm4py
from PALSYN.synthesizer import LSTMSynthesizer

event_log = pm4py.read_xes("example_logs/your_log.xes")

palsyn_model = LSTMSynthesizer(
    pre_processing={
        "max_clusters": 15,
        "trace_quantile": 0.9,
        "seed": 42,
    },
    model={
        "embedding_output_dims": 128,
        "epochs": 5,
        "batch_size": 128,
        "validation_split": 0.15,
        "units_per_layer": [32, 16],
        "dropout": 0.0,
        "bidirectional": True,
    },
    dp_optimizer={
        "epsilon": 15.0,
        "learning_rate": 5e-4,
        "l2_norm_clip": 1.0,
    },
)

palsyn_model.fit(event_log)
palsyn_model.save_model("models/your_model_run")
```

For end-to-end, runnable walkthroughs (train -> save -> load -> sample) see the scripts in `tutorial/`:

- `tutorial/palsyn_lstm_tutorial.py` - LSTM backbone, epsilon=15
- `tutorial/palsyn_rnn_tutorial.py` - SimpleRNN baseline using the same hyperparameters
- `tutorial/palsyn_gru_tutorial.py` - GRU backbone, also mirroring the LSTM config
- `tutorial/palsyn_tcn_tutorial.py` - Temporal convolutional (TCN) backbone

### Sampling Event Logs
After training or loading a saved model, sample synthetic traces and export them to XES/Excel with the snippet below.

```python
import pm4py

from PALSYN.postprocessing.log_postprocessing import clean_xes_file
from PALSYN.synthesizer import LSTMSynthesizer

palsyn_model = LSTMSynthesizer.load("models/your_model_run")

event_log = palsyn_model.sample(sample_size=5600, batch_size=100)
event_log_xes = pm4py.convert_to_event_log(event_log)

xes_filename = "your_model_run.xes"
pm4py.write_xes(event_log_xes, xes_filename)
clean_xes_file(xes_filename, xes_filename)

df = pm4py.convert_to_dataframe(event_log_xes)
df["time:timestamp"] = df["time:timestamp"].astype(str)
df.to_excel("your_model_run.xlsx", index=False)
```

## Future Work
Future work will focus on enhancing the algorithm and making it available on PyPI.

## Contribution

We welcome contributions from the community. If you have any suggestions or issues, please create a GitHub issue or a pull request. 


## License
This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details. 



## Funding 
This research is funded by the German Federal Ministry of Education and Research (BMBF) and NextGenerationEU (European Union) in the project KI-AIM under the funding code 16KISA115K.

