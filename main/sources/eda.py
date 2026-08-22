import numpy as np
from pathlib import Path

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary  # type: ignore

# Other imports
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pyplot import figure
import os
import argparse
import plotext  # type: ignore
from datetime import datetime
from typing import List

# Define label mapping (Index -> Disease)
label_mapping = {
    0: "Atelectasis",
    1: "Effusion",
    2: "Infiltration",
    3: "No Finding",
    4: "Nodule",
    5: "Pneumothorax"
}

# Get the absolute path of the dataset
BASE_DIR = Path(__file__).resolve().parent
Y_train_path = BASE_DIR / "data" / "Y_train.npy"

# Load Y_train.npy
Y_train = np.load(Y_train_path)

# Apply mapping to labels
mapped_labels = np.vectorize(label_mapping.get)(Y_train)

# Print first 10 samples as an example
print("First 10 mapped labels:", mapped_labels[:10])

# Save the mapped labels to a new file (optional)
np.save(BASE_DIR / "data" / "Y_train_mapped.npy", mapped_labels)

print("Mapped labels saved successfully!")

# Get unique labels and their counts
unique_labels, counts = np.unique(Y_train, return_counts=True)

# Print the unique labels and their counts
print("Unique labels in Y_train:", unique_labels)
print("Counts per label:", counts)
