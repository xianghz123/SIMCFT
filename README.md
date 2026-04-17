# SIMCFT Training Pipeline

## Overview
This project provides a simple training pipeline for SIMCFT on trajectory data.

SIMCFT learns trajectory embeddings with two channels:
- a **status channel** for motion dynamics
- a **spatial-context channel** for grid-based spatial information

The learned embeddings can be used for trajectory similarity and retrieval.

## Files
- `parameters.py`: configuration
- `generate_grid2idx.py`: build grid vocabulary
- `build_cell_graph.py`: build cell transition graph
- `train_node2vec.py`: train cell embeddings
- `generate_simcft_dataset.py`: generate JSON datasets
- `train_simcft.py`: train the SIMCFT model

Shared modules:
- `traj2grid.py`
- `simcft_dataset.py`
- `simcft_model.py`

## Dependencies
- Python 3.x
- PyTorch
- NumPy
- pandas
- gensim

Install:
bash
pip install torch numpy pandas gensim

## Data
- Prepare the split CSV files in data/porto/.

## Run
- python generate_grid2idx.py
- python build_cell_graph.py
- python train_node2vec.py
- python generate_simcft_dataset.py
- python train_simcft.py

## Note
Default paths are defined in `parameters.py`.
