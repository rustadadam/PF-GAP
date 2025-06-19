#Imports
import sys
sys.path.insert(0, '/yunity/arusty/PF-GAP')
import pandas as pd
import os
import numpy as np
from independent_distance.distance_helpers import compute_distance_matrix

#Build data
labels = np.array(pd.read_csv("/yunity/arusty/PF-GAP/data/ftse_100_sectors.csv")).flatten()
time_series = pd.read_csv("/yunity/arusty/PF-GAP/data/ftse_100_close_prices.csv", index_col=0)
prox_dir = "/yunity/arusty/PF-GAP/data/ftse_100/results"


# Compute and save distance matrices for all available metrics
metrics = [
        "dtw", "soft_dtw", "soft_dtw_normalized", 
        "euclidean", "manhattan", "cosine", "return_correlation",
        "shape_dtw"
        ]

for metric in metrics:
    print(f"Computing {metric} distance matrix...")
    dist_matrix = compute_distance_matrix(time_series, metric=metric)
    np.save(os.path.join(prox_dir, f"{metric}_matrix.npy"), dist_matrix)