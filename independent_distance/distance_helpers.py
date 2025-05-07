# File for functions to implement distance calculations for times series

from tslearn.metrics import cdist_dtw, cdist_soft_dtw, cdist_soft_dtw_normalized
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
import numpy as np

def return_correlation_distance(time_series_list):
    """
    Compute the return correlation distance matrix for a list of time series.

    Parameters:
        time_series_list (list): A list of time series data, where each time series is a 2D array.

    Returns:
        numpy.ndarray: The computed return correlation distance matrix.
    """
    n = len(time_series_list)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                distance_matrix[i, j] = 0
            else:
                ts1_returns = np.diff(time_series_list[i].flatten())
                ts2_returns = np.diff(time_series_list[j].flatten())
                correlation = np.corrcoef(ts1_returns, ts2_returns)[0, 1]
                distance_matrix[i, j] = 1 - correlation  # Convert correlation to distance

    return distance_matrix

def compute_distance_matrix(time_series_list, metric="dtw"):
    """
    Compute the distance matrix for a list of time series using the specified metric.

    Parameters:
        time_series_list (list): A list of time series data, where each time series is a 2D array.
        metric (str): The distance metric to use. Options are "dtw", "soft_dtw", "soft_dtw_normalized",
                      "euclidean", "manhattan", "cosine", "return_correlation".

    Returns:
        numpy.ndarray: The computed distance matrix.
    """
    if metric == "dtw":
        return cdist_dtw(time_series_list)
    elif metric == "soft_dtw":
        return cdist_soft_dtw(time_series_list)
    elif metric == "soft_dtw_normalized":
        return cdist_soft_dtw_normalized(time_series_list)
    elif metric == "euclidean":
        return euclidean_distances(time_series_list)
    elif metric == "manhattan":
        return manhattan_distances(time_series_list)
    elif metric == "cosine":
        return cosine_distances(time_series_list)
    elif metric == "return_correlation":
        return return_correlation_distance(time_series_list)
    else:
        raise ValueError("Unsupported metric. Choose from 'dtw', 'soft_dtw', 'soft_dtw_normalized', 'euclidean', 'manhattan', 'cosine', or 'return_correlation'.")
