"""FreshPRINCEClassifier.

Pipeline classifier using the full set of TSFresh features and a
RotationForestClassifier.
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["FreshPRINCEClassifier"]

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from aeon.classification.base import BaseClassifier
from aeon.classification.sklearn import RotationForestClassifier
from aeon.transformations.collection.feature_based import TSFresh
from aeon.classification.feature_based import FreshPRINCEClassifier #Regressor doesn't exist yet (At least, I didn't find one with my minimal search)


from helpers import ProximityMixin  # Import the ProximityMixin

def rotation_forest_apply(self, X):
    """
    Apply the RotationForestClassifier to the input data and return the leaf indices.

    Parameters
    ----------
    X : 2d ndarray or DataFrame of shape = [n_cases, n_attributes]
        The input data to apply to the trees.

    Returns
    -------
    leaf_matrix : np.ndarray
        A matrix where each row corresponds to a sample and each column corresponds
        to the leaf index of a tree in the ensemble.
    """
    if not hasattr(self, "_is_fitted") or not self._is_fitted:
        from sklearn.exceptions import NotFittedError

        raise NotFittedError(
            f"This instance of {self.__class__.__name__} has not "
            f"been fitted yet; please call `fit` first."
        )

    # Data preprocessing
    X = self._check_X(X)
    X = X[:, self._useful_atts]
    X = (X - self._min) / self._ptp

    # Collect leaf indices from each tree
    leaf_matrix = []
    for tree, pca, group in zip(self.estimators_, self._pcas, self._groups):
        # Transform the data using the PCA for this tree
        X_transformed = np.concatenate(
            [pca[i].transform(X[:, group[i]]) for i in range(len(group))], axis=1
        )
        X_transformed = np.nan_to_num(
            X_transformed, False, 0, np.finfo(np.float32).max, np.finfo(np.float32).min
        )
        # Get the leaf indices from the decision tree
        leaf_indices = tree.apply(X_transformed)
        leaf_matrix.append(leaf_indices)

    # Combine leaf indices from all trees into a single matrix
    return np.column_stack(leaf_matrix)

class FreshPRINCE_GAP(FreshPRINCEClassifier, ProximityMixin):
    """
    Fresh Pipeline with RotatIoN forest Classifier.

    This classifier simply transforms the input data using the TSFresh [1]_
    transformer with comprehensive features and builds a RotationForestClassifier
    estimator using the transformed data.

    Parameters
    ----------
    default_fc_parameters : str, default="comprehensive"
        Set of TSFresh features to be extracted, options are "minimal", "efficient" or
        "comprehensive".
    n_estimators : int, default=200
        Number of estimators for the RotationForestClassifier ensemble.
    base_estimator : BaseEstimator or None, default="None"
        Base estimator for the ensemble. By default, uses the sklearn
        `DecisionTreeClassifier` using entropy as a splitting measure.
    pca_solver : str, default="auto"
        Solver to use for the PCA ``svd_solver`` parameter in rotation forest. See the
        scikit-learn PCA implementation for options.
    verbose : int, default=0
        Level of output printed to the console (for information only).
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    chunksize : int or None, default=None
        Number of series processed in each parallel TSFresh job, should be optimised
        for efficient parallelisation.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    prox_method : str, default="rfgap"
        Proximity method to use.
    matrix_type : str, default="sparse"
        Type of matrix to use for proximities.
    triangular : bool, default=True
        Whether to use a triangular matrix for proximities.
    force_symmetric : bool, default=False
        Whether to force the proximity matrix to be symmetric.
    non_zero_diagonal : bool, default=False
        Whether to enforce a non-zero diagonal in the proximity matrix.
    """

    def __init__(
        self,
        default_fc_parameters="comprehensive",
        n_estimators=200,
        base_estimator=None,
        pca_solver="auto",
        verbose=0,
        n_jobs=1,
        chunksize=None,
        random_state=None,
        prox_method="rfgap",
        matrix_type="sparse",
        triangular=True,
        force_symmetric=False,
        non_zero_diagonal=False,
    ):
        self.default_fc_parameters = default_fc_parameters
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.pca_solver = pca_solver

        self.verbose = verbose
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.random_state = random_state

        # Proximity-related attributes
        self.prox_method = prox_method
        self.matrix_type = matrix_type
        self.triangular = triangular
        self.force_symmetric = force_symmetric
        self.non_zero_diagonal = non_zero_diagonal

        self.n_cases_ = 0
        self.n_channels_ = 0
        self.n_timepoints_ = 0

        self._rotf = None
        self._tsfresh = None

        super().__init__()

    def fit(self, X, y, static=None):
        """Fit Classifier to training data.

        Parameters
        ----------
        X : 3D np.ndarray
            The training data shape = (n_cases, n_channels, n_timepoints).
        y : 1D np.ndarray
            The training labels, shape = (n_cases).
        static : None or array-like, default=None
            The static features for samples in X.

        Returns
        -------
        self :
            Reference to self.
        """
        #Add case where channels is missing
        if X.ndim == 2:
            X = np.expand_dims(X, axis=1)
        elif X.ndim != 3:
            raise ValueError(f"X must be 2D or 3D, but got {X.ndim} dimensions.")

        X_t = self._fit_fp_shared(X, y) # ._tsfresh is the transformation model. ._rotf is the prediction


        # Add static data to X_t if provided
        if static is not None:
            X_t = np.hstack([X_t, static])

        self._rotf.fit(X_t, y)

        #Implement custom built apply method
        self._rotf.apply = lambda X: rotation_forest_apply(self._rotf, X)

        #Set up _rotfl
        self._rotf.verbose = 0


        self._estimator = self._rotf


        # From the proximity mixin
        self.prox_fit(X_t, None)

        self.is_fitted = True

        return self

    def predict(self, X, static=None):
        """
        Predict class labels for samples in X, optionally using static features.

        Parameters
        ----------
        X : 3D or 2D np.ndarray
            The input data shape = (n_cases, n_channels, n_timepoints) or (n_cases, n_timepoints).
        static : None or array-like, default=None
            The static features for samples in X.

        Returns
        -------
        y_pred : np.ndarray
            Predicted class labels.
        """
        # Add case where channels is missing
        if X.ndim == 2:
            X = np.expand_dims(X, axis=1)
        elif X.ndim != 3:
            raise ValueError(f"X must be 2D or 3D, but got {X.ndim} dimensions.")

        # Transform X using the fitted TSFresh transformer
        X_t = self._tsfresh.transform(X)

        # Add static data to X_t if provided
        if static is not None:
            X_t = np.hstack([X_t, static])

        # Use the fitted estimator to predict
        return self._estimator.predict(X_t)

    def get_extend(self, X, static=None):
        """
        Predict class labels for samples in X, optionally using static features.

        Parameters
        ----------
        X : 3D or 2D np.ndarray
            The input data shape = (n_cases, n_channels, n_timepoints) or (n_cases, n_timepoints).
        static : None or array-like, default=None
            The static features for samples in X.

        Returns
        -------
        y_pred : np.ndarray
            Predicted class labels.
        """
        # Add case where channels is missing
        if X.ndim == 2:
            X = np.expand_dims(X, axis=1)
        elif X.ndim != 3:
            raise ValueError(f"X must be 2D or 3D, but got {X.ndim} dimensions.")

        # Transform X using the fitted TSFresh transformer
        X_t = self._tsfresh.transform(X)

        # Add static data to X_t if provided
        if static is not None:
            X_t = np.hstack([X_t, static])

        # Use the fitted estimator to predict
        return self.prox_extend(X_t)