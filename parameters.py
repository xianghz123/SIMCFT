# parameters.py
# Shared configuration for the Porto experiment in SimCFT.

import os

# =========================
# Dataset name
# =========================
dataset_name = "porto"

# =========================
# Project root
# =========================
# This file is expected to be placed in the project root.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# =========================
# Directory settings
# =========================
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", dataset_name)
MODEL_ROOT = os.path.join(PROJECT_ROOT, "model", dataset_name)

raw_dir = os.path.join(DATA_ROOT, "raw")
split_dir = os.path.join(DATA_ROOT, "split")
grid_dir = os.path.join(DATA_ROOT, "grid")
train_dir = os.path.join(DATA_ROOT, "train")
valid_dir = os.path.join(DATA_ROOT, "valid")
test_dir = os.path.join(DATA_ROOT, "test")

model_dir = MODEL_ROOT

# =========================
# Geographic bounds (Porto)
# =========================
min_lon = -8.735152
max_lon = -8.156309
min_lat = 40.953673
max_lat = 41.307945

# =========================
# Raw / split file names
# =========================
raw_csv_name = "train.csv"

train_csv_name = "porto_train.csv"
valid_csv_name = "porto_valid.csv"
test_csv_name = "porto_test.csv"

# =========================
# Porto CSV column names
# =========================
trip_id_col = "TRIP_ID"
timestamp_col = "TIMESTAMP"
missing_data_col = "MISSING_DATA"
polyline_col = "POLYLINE"

# =========================
# Grid settings
# =========================
row_num = 400
column_num = 400
grid_lower_bound = 10

# =========================
# Trajectory settings
# =========================
sample_interval = 15
porto_sample_interval = sample_interval  # kept for backward compatibility
min_traj_points = 5

# =========================
# Model / training defaults
# =========================
embedding_dim = 128
num_heads = 8
max_len = 256
learning_rate = 1e-3
temperature = 0.1