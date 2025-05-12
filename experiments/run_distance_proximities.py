#Imports
import sys
sys.path.insert(0, '/yunity/arusty/PF-GAP')
import pandas as pd
import os
import numpy as np
from independent_distance.distance_helpers import compute_distance_matrix

#Build data
labels = pd.read_csv('data/labels.csv')
time_series = pd.read_csv('data/time_series.csv')

# Compute and save distance matrices for all available metrics
metrics = ["soft_dtw", "soft_dtw_normalized", #"dtw", 
                      "euclidean", "manhattan", "cosine", "return_correlation"]
prox_dir = "prox_files"

for metric in metrics:
    print(f"Computing {metric} distance matrix...")
    dist_matrix = compute_distance_matrix(time_series, metric=metric)
    np.save(os.path.join(prox_dir, f"{metric}_matrix.npy"), dist_matrix)