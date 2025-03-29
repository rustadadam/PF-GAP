import numpy as np
import matplotlib.pyplot as plt

def plot_random_time_series(data, indices=None, n=1):
    """
    Plot one or more time series from XtrainC with all their channels.

    Parameters:
    indices : list or array of integers, optional
        Indices of the time series to plot. If None, n random time series will be selected.
    n : int, default=1
        Number of random time series to plot when indices is None.
    """
    num_series = data.shape[0]
    # Choose indices
    if indices is None:
        indices = np.random.choice(num_series, size=n, replace=False)
    else:
        indices = np.array(indices)
    
    n_channels = data.shape[1]
    n_timesteps = data.shape[2]
    
    # Create subplots for each time series for easy comparison
    fig, axs = plt.subplots(1, len(indices), figsize=(6 * len(indices), 4))
    if len(indices) == 1:
        axs = [axs]
        
    for ax, idx in zip(axs, indices):
        for ch in range(n_channels):
            ax.plot(np.arange(n_timesteps), data[idx, ch, :], label=f'Channel {ch+1}')
        ax.set_title(f"Time Series Index {idx}")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Value")
        ax.legend()
    
    plt.tight_layout()
    plt.show()