import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_is_fitted
from distutils.version import LooseVersion
import sklearn
if LooseVersion(sklearn.__version__) >= LooseVersion("0.24"):
    # In sklearn version 0.24, forest module changed to be private.
    from sklearn.ensemble._forest import _generate_unsampled_indices
    from sklearn.ensemble import _forest as forest
    from sklearn.ensemble._forest import _generate_sample_indices
else:
    # Before sklearn version 0.24, forest was public, supporting this.
    from sklearn.ensemble.forest import _generate_unsampled_indices # Remove underscore from _forest
    from sklearn.ensemble.forest import _generate_sample_indices # Remove underscore from _forest
    from sklearn.ensemble import forest


class ProximityMixin:

    def _get_oob_samples(self, data):
        
        """This is a helper function for get_oob_indices. 

        Parameters
        ----------
        data : array_like (numeric) of shape (n_samples, n_features)

        """
        n = len(data)
        oob_samples = []
        for tree in self._estimator.estimators_:
            # Here at each iteration we obtain out-of-bag samples for every tree.
            oob_indices = _generate_unsampled_indices(tree.random_state, n, n)
            oob_samples.append(oob_indices)

        return oob_samples

    def get_oob_indices(self, data):
        
        """This generates a matrix of out-of-bag samples for each decision tree in the forest

        Parameters
        ----------
        data : array_like (numeric) of shape (n_samples, n_features)


        Returns
        -------
        oob_matrix : array_like (n_samples, n_estimators) 

        """
        n = len(data)
        num_trees = self._estimator.n_estimators
        oob_matrix = np.zeros((n, num_trees))
        oob_samples = self._get_oob_samples(data)

        for t in range(num_trees):
            matches = np.unique(oob_samples[t])
            oob_matrix[matches, t] = 1

        return oob_matrix.astype(int)

    def _get_in_bag_samples(self, data):

        """This is a helper function for get_in_bag_indices. 

        Parameters
        ----------
        data : array_like (numeric) of shape (n_samples, n_features)

        """

        n = len(data)
        in_bag_samples = []
        for tree in self._estimator.estimators_:
        # Here at each iteration we obtain in-bag samples for every tree.
            in_bag_sample = _generate_sample_indices(tree.random_state, n, n)
            in_bag_samples.append(in_bag_sample)
        return in_bag_samples

    def get_in_bag_counts(self, data):
        
        """This generates a matrix of in-bag samples for each decision tree in the forest

        Parameters
        ----------
        data : array_like (numeric) of shape (n_samples, n_features)


        Returns
        -------
        in_bag_matrix : array_like (n_samples, n_estimators) 

        """
        n = len(data)
        num_trees = self._estimator.n_estimators
        in_bag_matrix = np.zeros((n, num_trees))
        in_bag_samples = self._get_in_bag_samples(data)

        for t in range(num_trees):
            matches, n_repeats = np.unique(in_bag_samples[t], return_counts = True)
            in_bag_matrix[matches, t] += n_repeats


        return in_bag_matrix

    def prox_fit(self, X, x_test = None):
        self.leaf_matrix = self._estimator.apply(X)
            
        if x_test is not None:
            n_test = np.shape(x_test)[0]
            
            self.leaf_matrix_test = self.apply(x_test)
            self.leaf_matrix = np.concatenate((self.leaf_matrix, self.leaf_matrix_test), axis = 0)
                                
        if self.prox_method == 'oob':
            self.oob_indices = self.get_oob_indices(X)
            
            if x_test is not None:
                self.oob_indices = np.concatenate((self.oob_indices, np.ones((n_test, self.n_estimators))))
            
            self.oob_leaves = self.oob_indices * self.leaf_matrix

        if self.prox_method == 'rfgap':

            self.oob_indices = self.get_oob_indices(X)
            self.in_bag_counts = self.get_in_bag_counts(X)

            
            if x_test is not None:
                self.oob_indices = np.concatenate((self.oob_indices, np.ones((n_test, self.n_estimators))))
                self.in_bag_counts = np.concatenate((self.in_bag_counts, np.zeros((n_test, self.n_estimators))))                
                            
            self.in_bag_indices = 1 - self.oob_indices

            self.in_bag_leaves = self.in_bag_indices * self.leaf_matrix
            self.oob_leaves = self.oob_indices * self.leaf_matrix
        
    def get_proximity_vector(self, ind):
        #Implement Checks:
        if self.prox_method not in ['oob', 'original', 'rfgap']:
            raise ValueError("Invalid proximity method. Choose 'oob', 'original', or 'rfgap'.")

        n, num_trees = self.leaf_matrix.shape
        
        prox_vec = np.zeros((1, n))
        if self.prox_method == 'oob':
            if self.triangular:
                ind_oob_leaves = np.nonzero(self.oob_leaves[ind, :])[0]
                tree_counts = np.sum(
                    self.oob_indices[ind, ind_oob_leaves] ==
                    self.oob_indices[ind:, ind_oob_leaves], axis=1
                )
                tree_counts[tree_counts == 0] = 1
                prox_counts = np.sum(
                    self.oob_leaves[ind, ind_oob_leaves] ==
                    self.oob_leaves[ind:, ind_oob_leaves], axis=1
                )
                prox_vec = np.divide(prox_counts, tree_counts)
                cols = np.where(prox_vec != 0)[0] + ind
                rows = np.ones(len(cols), dtype=int) * ind
                data = prox_vec[cols - ind]
            else:
                ind_oob_leaves = np.nonzero(self.oob_leaves[ind, :])[0]
                tree_counts = np.sum(
                    self.oob_indices[ind, ind_oob_leaves] ==
                    self.oob_indices[:, ind_oob_leaves], axis=1
                )
                tree_counts[tree_counts == 0] = 1
                prox_counts = np.sum(
                    self.oob_leaves[ind, ind_oob_leaves] ==
                    self.oob_leaves[:, ind_oob_leaves], axis=1
                )
                prox_vec = np.divide(prox_counts, tree_counts)
                cols = np.nonzero(prox_vec)[0]
                rows = np.ones(len(cols), dtype=int) * ind
                data = prox_vec[cols]

        elif self.prox_method == 'original':
            if self.triangular:
                tree_inds = self.leaf_matrix[ind, :]
                prox_vec = np.sum(tree_inds == self.leaf_matrix[ind:, :], axis=1)
                cols = np.where(prox_vec != 0)[0] + ind
                rows = np.ones(len(cols), dtype=int) * ind
                data = prox_vec[cols - ind] / num_trees
            else:
                tree_inds = self.leaf_matrix[ind, :]
                prox_vec = np.sum(tree_inds == self.leaf_matrix, axis=1)
                cols = np.nonzero(prox_vec)[0]
                rows = np.ones(len(cols), dtype=int) * ind
                data = prox_vec[cols] / num_trees

        elif self.prox_method == 'rfgap':
            oob_trees = np.nonzero(self.oob_indices[ind, :])[0]
            in_bag_trees = np.nonzero(self.in_bag_indices[ind, :])[0]
            terminals = self.leaf_matrix[ind, :]
            matches = terminals == self.in_bag_leaves 
            match_counts = np.where(matches, self.in_bag_counts, 0)
            ks = np.sum(match_counts, axis=0)
            ks[ks == 0] = 1
            ks_in = ks[in_bag_trees]
            ks_out = ks[oob_trees]
            S_out = np.count_nonzero(self.oob_indices[ind, :])
            prox_vec = np.sum(
                np.divide(match_counts[:, oob_trees], ks_out), axis=1
            ) / S_out

            if self.non_zero_diagonal:
                S_in = np.count_nonzero(self.in_bag_indices[ind, :])
                if S_in > 0:
                    prox_vec[ind] = np.sum(
                        np.divide(match_counts[ind, in_bag_trees], ks_in)
                    ) / S_in
                else:
                    prox_vec[ind] = np.sum(
                        np.divide(match_counts[ind, in_bag_trees], ks_in)
                    )
                prox_vec = prox_vec / np.max(prox_vec)
                prox_vec[ind] = 1

            cols = np.nonzero(prox_vec)[0]
            rows = np.ones(len(cols), dtype=int) * ind
            data = prox_vec[cols]

        return data.tolist(), rows.tolist(), cols.tolist()


    def get_proximities(self):
        from scipy import sparse

        check_is_fitted(self)
        n, _ = self.leaf_matrix.shape

        prox_vals, rows, cols = self.get_proximity_vector(0)
        for i in range(1, n):
            if self._estimator.verbose and i % 100 == 0:
                print('Finished with {} rows'.format(i))
            prox_val_temp, rows_temp, cols_temp = self.get_proximity_vector(i)
            prox_vals.extend(prox_val_temp)
            rows.extend(rows_temp)
            cols.extend(cols_temp)

        if self.triangular and self.prox_method != 'rfgap':
            prox_sparse = sparse.csr_matrix(
                (
                    np.array(prox_vals + prox_vals),
                    (np.array(rows + cols), np.array(cols + rows))
                ),
                shape=(n, n)
            )
            prox_sparse.setdiag(1)
        else:
            prox_sparse = sparse.csr_matrix(
                (np.array(prox_vals), (np.array(rows), np.array(cols))),
                shape=(n, n)
            )

        if self.force_symmetric:
            prox_sparse = (prox_sparse + prox_sparse.transpose()) / 2

        if self.matrix_type == 'dense':
            return np.array(prox_sparse.todense())
        else:
            return prox_sparse

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