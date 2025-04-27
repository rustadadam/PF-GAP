"""Time Series Forest (TSF) Classifier.

Interval-based TSF classifier, extracts basic summary features from random intervals.
"""
import sys
sys.path.insert(0, '/yunity/arusty/PF-GAP')

__maintainer__ = []
__all__ = ["TimeSeriesForestClassifier"]

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier

from aeon.base._estimators.interval_based.base_interval_forest import BaseIntervalForest
from aeon.classification import BaseClassifier
from aeon.classification.sklearn import ContinuousIntervalTree
from helpers import ProximityMixin
from aeon.classification.interval_based import TimeSeriesForestClassifier




class TSP_GAP(TimeSeriesForestClassifier, ProximityMixin):
    """Time series forest (TSF) classifier.

    Time series forest is an ensemble of decision trees built on random intervals [1]_.
    Overview: Input n series length m.
    For each tree
        - sample sqrt(m) intervals,
        - find mean, std and slope for each interval, concatenate to form new
        data set,
        - build a decision tree on new data set.
    Ensemble the trees with averaged probability estimates.

    This implementation deviates from the original in minor ways. It samples
    intervals with replacement and does not use the tree splitting criteria
    refinement described in [1] (this can be done with the CITClassifier base
    estimator).

    Parameters
    ----------
    base_estimator : BaseEstimator or None, default=None
        scikit-learn BaseEstimator used to build the interval ensemble. If None, use a
        simple decision tree.
    n_estimators : int, default=200
        Number of estimators to build for the ensemble.
    n_intervals : int, str, list or tuple, default="sqrt"
        Number of intervals to extract per tree for each series_transformers series.

        An int input will extract that number of intervals from the series, while a str
        input will return a function of the series length (may differ per
        series_transformers output) to extract that number of intervals.
        Valid str inputs are:
            - "sqrt": square root of the series length.
            - "sqrt-div": sqrt of series length divided by the number
                of series_transformers.

        A list or tuple of ints and/or strs will extract the number of intervals using
        the above rules and sum the results for the final n_intervals. i.e. [4, "sqrt"]
        will extract sqrt(n_timepoints) + 4 intervals.

        Different number of intervals for each series_transformers series can be
        specified using a nested list or tuple. Any list or tuple input containing
        another list or tuple must be the same length as the number of
        series_transformers.

        While random interval extraction will extract the n_intervals intervals total
        (removing duplicates), supervised intervals will run the supervised extraction
        process n_intervals times, returning more intervals than specified.
    min_interval_length : int, float, list, or tuple, default=3
        Minimum length of intervals to extract from series. float inputs take a
        proportion of the series length to use as the minimum interval length.

        Different minimum interval lengths for each series_transformers series can be
        specified using a list or tuple. Any list or tuple input must be the same length
        as the number of series_transformers.
    max_interval_length : int, float, list, or tuple, default=np.inf
        Maximum length of intervals to extract from series. float inputs take a
        proportion of the series length to use as the maximum interval length.

        Different maximum interval lengths for each series_transformers series can be
        specified using a list or tuple. Any list or tuple input must be the same length
        as the number of series_transformers.

        Ignored for supervised interval_selection_method inputs.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding n_estimators.
        Default of 0 means n_estimators are used.
    contract_max_n_estimators : int, default=500
        Max number of estimators when time_limit_in_minutes is set.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Specify the parallelisation backend implementation in joblib, if None a 'prefer'
        value of "threads" is used by default.
        Valid options are "loky", "multiprocessing", "threading" or a custom backend.
        See the joblib Parallel documentation for more details.

    Attributes
    ----------
    n_cases_ : int
        The number of train cases in the training set.
    n_channels_ : int
        The number of dimensions per case in the training set.
    n_timepoints_ : int
        The length of each series in the training set.
    n_classes_ : int
        Number of classes. Extracted from the data.
    classes_ : ndarray of shape (n_classes_)
        Holds the label for each class.
    total_intervals_ : int
        Total number of intervals per tree from all representations.
    estimators_ : list of shape (n_estimators) of BaseEstimator
        The collections of estimators trained in fit.
    intervals_ : list of shape (n_estimators) of BaseCollectionTransformer
        Stores the interval extraction transformer for all estimators.

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/
     java/tsml/classifiers/interval_based/TSF.java>`_.

    References
    ----------
    .. [1] H.Deng, G.Runger, E.Tuv and M.Vladimir, "A time series forest for
       classification and feature extraction", Information Sciences, 239, 2013

    Examples
    --------
    >>> from aeon.classification.interval_based import TimeSeriesForestClassifier
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12,
    ...                              return_y=True, random_state=0)
    >>> clf = TimeSeriesForestClassifier(n_estimators=10, random_state=0)
    >>> clf.fit(X, y)
    TimeSeriesForestClassifier(n_estimators=10, random_state=0)
    >>> clf.predict(X)
    array([0, 1, 0, 1, 0, 0, 1, 1, 1, 0])
    """

    _tags = {
        "capability:multivariate": True,
        "capability:train_estimate": True,
        "capability:contractable": True,
        "capability:multithreading": True,
        "algorithm_type": "interval",
    }

    def __init__(
        self,
        #ProximityMixin
        prox_method = "rfgap",
        matrix_type = "sparse",
        triangular = True,
        force_symmetric = False,
        non_zero_diagonal = False,
        #TSF
        base_estimator=None,
        n_estimators=200,
        n_intervals="sqrt",
        min_interval_length=3,
        max_interval_length=np.inf,
        time_limit_in_minutes=None,
        contract_max_n_estimators=500,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.prox_method = prox_method
        self.matrix_type = matrix_type
        self.triangular  = triangular
        self.non_zero_diagonal = non_zero_diagonal
        self.force_symmetric = force_symmetric

        super().__init__(
            base_estimator=None,
            n_estimators=200,
            n_intervals="sqrt",
            min_interval_length=3,
            max_interval_length=np.inf,
            time_limit_in_minutes=None,
            contract_max_n_estimators=500,
            random_state=None,
            n_jobs=1,
            parallel_backend=None
        )

        #This must be overridden to use the ProximityMixin
        self.interval_selection_method = "supervised"

    def fit(self, X, y, X_static=None):
        # Call the parent class's _fit method to train the forest
        super()._fit(X, y)
        self.is_fitted = True

        return self

        Xt = self._predict_setup(X)

       
        for i in range(self._n_estimators):
            interval_features = np.empty((Xt[0].shape[0], 0))

            for r in range(len(Xt)):
                f = self.intervals_[i][r].transform(Xt[r])
                interval_features = np.hstack((interval_features, f))

            if isinstance(self.replace_nan, str) and self.replace_nan.lower() == "nan":
                interval_features = np.nan_to_num(
                    interval_features, False, np.nan, np.nan, np.nan
                )
            elif isinstance(self.replace_nan, (int, float)):
                interval_features = np.nan_to_num(
                    interval_features,
                    False,
                    self.replace_nan,
                    self.replace_nan,
                    self.replace_nan,
                )

            # For the interval code
            self._estimator = self.estimators_[i]
            self._estimator.n_estimators = 1
            self.prox_fit(interval_features, None)
        
        self.is_fitted = True

        return super().get_proximities()
        

        self.is_fitted = True
        return self


class ProximityRandomForest(RandomForestClassifier, ProximityMixin):
    def __init__(self, base_estimators, intervals, series_transformers, tsf):
        # Initialize the RandomForestClassifier with the provided estimators
        super().__init__(n_estimators=len(base_estimators))
        self.estimators_ = list(base_estimators)  # Ensure it's a list, not a tuple
        self.intervals_ = intervals  # Interval transformers from TSF
        self.series_transformers = series_transformers  # Series transformers from TSF
        self.tsf = tsf

    def fit(self, X, y=None):
        # Override fit to ensure compatibility with ProximityMixin
        raise NotImplementedError("ProximityRandomForest does not support fitting. Use pre-trained estimators.")

    def prox_fit(self, X, x_test=None):
        # Transform the input data using the interval-based transformations for all estimators
        Xt = self.tsf._predict_setup(X)

        # Calculate the leaf indices for each estimator
        leaf_indices = Parallel(n_jobs=self.tsf._n_jobs)(
            delayed(self._prox_fit_for_estimator)(
                Xt,
                self.estimators_[i],
                self.intervals_[i],
            )
            for i in range(len(self.estimators_))
        )
        self.leaf_matrix = np.column_stack(leaf_indices)  # Combine into a single matrix

        if x_test is not None:
            Xt_test = self.tsf._predict_setup(x_test)
            test_leaf_indices = Parallel(n_jobs=self.tsf._n_jobs)(
                delayed(self._prox_fit_for_estimator)(
                    Xt_test,
                    self.estimators_[i],
                    self.intervals_[i],
                )
                for i in range(len(self.estimators_))
            )
            self.test_leaf_matrix = np.column_stack(test_leaf_indices)

    def _prox_fit_for_estimator(self, Xt, estimator, intervals):
        # Transform the data for the specific estimator
        interval_features = np.empty((Xt[0].shape[0], 0))

        for r in range(len(Xt)):
            f = intervals[r].transform(Xt[r])
            interval_features = np.hstack((interval_features, f))

        if isinstance(self.tsf.replace_nan, str) and self.tsf.replace_nan.lower() == "nan":
            interval_features = np.nan_to_num(
                interval_features, False, np.nan, np.nan, np.nan
            )
        elif isinstance(self.replace_nan, (int, float)):
            interval_features = np.nan_to_num(
                interval_features,
                False,
                self.tsf.replace_nan,
                self.tsf.replace_nan,
                self.tsf.replace_nan,
            )

        # Apply the estimator to get leaf indices
        return estimator.apply(interval_features)
