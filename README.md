# SIMCFT

## Overview
This script is designed for training the SIMCFT model on trajectory datasets. SIMCFT is intended for metric learning tasks involving trajectory data, capable of learning embeddings that can be used for trajectory similarity assessment, clustering, or retrieval.

## Dependencies
- Python 3.x
- PyTorch
- NumPy
- argparse
- Other dependencies as required by the utility scripts (utils.py, etc.).

## Installation
1. Ensure you have Python 3.x installed on your system. It’s recommended to use a virtual environment:

 ```sh
python3 -m venv SIMCFT-env
source SIMCFT-env/bin/activate
Install the required Python packages:
 ```

2. Install the required Python packages:
 ```sh
pip install torch numpy argparse
```

## Preparing Your Dataset
You need to prepare your trajectory dataset in JSON format with the following fields:
- `trajs`: List of trajectory indices.
- `origin_trajs`: List of original trajectories with (lon, lat) coordinates.

Make sure you have separate files for training, validation, and testing datasets, for example:



## Running the Script
To train the SIMCFT model, prepare your dataset in the specified format and adjust parameters according to your dataset characteristics.

1. **Prepare your trajectory dataset** in the required format, ensuring you have training, validation, and test sets ready.
2. **Adjust parameters** in the `parameters.py` script or prepare to override them via command-line arguments when running the training script.

### Example Command
Run the script with the desired parameters:
```sh
python train.py --batch_size 32 --epoch_num 100 --learning_rate 0.001 --device cuda
