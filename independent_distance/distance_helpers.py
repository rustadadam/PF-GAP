# File for functions to implement distance calculations for times series

from tslearn.metrics import cdist_dtw, cdist_soft_dtw, cdist_soft_dtw_normalized
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances

def compute_distance_matrix(time_series_list, metric="dtw"):
    """
    Compute the distance matrix for a list of time series using the specified metric.

    Parameters:
        time_series_list (list): A list of time series data, where each time series is a 2D array.
        metric (str): The distance metric to use. Options are "dtw", "soft_dtw", "soft_dtw_normalized",
                      "euclidean", "manhattan", "cosine".

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
    else:
        raise ValueError("Unsupported metric. Choose from 'dtw', 'soft_dtw', 'soft_dtw_normalized', 'euclidean', 'manhattan', or 'cosine'.")
