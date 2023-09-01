import os
import pathlib

data_path = None # TODO

### Data Paths ###
sega_path = data_path / "SEGA"

### RAW Data Paths ###
raw_sega_path = data_path / "SEGA" / "RAW"

### Parsed Data Paths ###
parsed_sega_path = data_path / "SEGA" / "PARSED"

### Training Paths ###
project_path = None # TODO
checkpoints_path = project_path / "Checkpoints"
logs_path = project_path / "Logs"
figures_path = project_path / "Figures"
models_path = project_path / "Models"