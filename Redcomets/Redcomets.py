"""Random EnhanceD Co-eye for Multivariate Time Series (RED CoMETS).

Ensemble of symbolically represented time series using random forests as the base
classifier.
"""

__maintainer__ = ["zy18811"]
__all__ = ["REDCOMETS"]

from collections import Counter

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

from aeon.classification.base import BaseClassifier
from aeon.transformations.collection import Normalizer
from aeon.transformations.collection.dictionary_based import SAX, SFA
from aeon.utils.validation._dependencies import _check_soft_dependencies


class REDCOMETS(BaseClassifier):
    """
    Random EnhanceD Co-eye for Multivariate Time Series (RED CoMETS).

    Ensemble of symbolically represented time series using random forests as the base
    classifier as described in [1]_. Based on Co-eye [2]_.

    Parameters
    ----------
    variant : int, default=3
        RED CoMETS variant to use from {1, 2, 3, 4, 5, 6, 7, 8, 9} to use as per [1]_.
        Defaults to RED CoMETS-3. Variants 4-9 only support multivariate problems.
    perc_length : int or float, default=5
        Percentage of time series length used to determinne number of lenses during
        pair selection.
    n_trees : int, default=100
        Number of trees used by each random forest sub-classifier.
    random_state : int, RandomState instance or None, default=None
        If ``int``, random_state is the seed used by the random number generator;
        If ``RandomState`` instance, ``random_state`` is the random number generator;
        If ``None``, the random number generator is the ``RandomState`` instance used
        by ``np.random``.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Specify the parallelisation backend implementation in joblib,
        if ``None`` a 'prefer' value of "threads" is used by default.
        Valid options are "loky", "multiprocessing", "threading" or a custom backend.
        See the joblib Parallel documentation for more details.

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    classes_ : list
        The unique class labels.

    See Also
    --------
    SAX, SFA

    Notes
    -----
    Adapted from the implementation at https://github.com/zy18811/RED-CoMETS

    References
    ----------
    .. [1] Luca A. Bennett and Zahraa S. Abdallah, "RED CoMETS: An Ensemble Classifier
       for Symbolically Represented Multivariate Time Series." In proceedings of the
       8th Workshop on Advanced Analytics and Learning on Temporal Data (AALTD 2023).
    .. [2] Zahraa S. Abdallah and Mohamed Medhat Gaber, "Co-eye: a multi-resolution
       ensemble classifier for symbolically approximated time series."
       Machine Learning (2020).

    Examples
    --------
    >>> from aeon.classification.dictionary_based import REDCOMETS
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = REDCOMETS()  # doctest: +SKIP
    >>> clf.fit(X_train, y_train)  # doctest: +SKIP
    REDCOMETS(...)
    >>> y_pred = clf.predict(X_test)  # doctest: +SKIP
    """

    _tags = {
        "python_dependencies": "imblearn",
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "dictionary",
    }

    def __init__(
        self,
        variant=3,
        perc_length=5,
        n_trees=100,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
        static=None, 
        debug_mode=False
    ):
        assert variant in [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.variant = variant

        assert 0 < perc_length <= 100
        self.perc_length = perc_length

        self.n_trees = n_trees

        self.random_state = random_state
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

        self._n_channels = None
        self.static = static
        self.debug_mode = debug_mode

        super().__init__()

    def _prepare_static(self, X):
        static = np.asarray(self.static)
        # If the transformed X has extra columns, pad static with two dummy columns.
        col_diff = X.shape[0] - static.shape[0]
        if col_diff > 0: #NOTE: test this for many different cases
            static = np.vstack([static, np.zeros((col_diff, static.shape[1]))]) #TODO: Instead of adding ZEROS: clone the static features from the minority class. This assumes the static features are the same.

        if static.shape[0] == X.shape[0]:
            return static
        elif static.shape[0] == 1:
            return np.tile(static, (X.shape[0], 1)) 
        else:
            raise ValueError(
                f"Static feature array has {static.shape[0]} rows but expected {X.shape[0]}"
            )

    def _fit(self, X, y):
        """Build a REDCOMETS classifier from the training set (X, y).

        Parameters
        ----------
        X : np.ndarray
            3D np.ndarray of shape (n_cases, n_channels, n_timepoints)
            The training data.
        y : np.ndarray
            1D np.ndarray of shape (n_cases)
            The class labels.

        Returns
        -------
        self :
            Reference to self.
        """
        self._n_channels = X.shape[1]

        if self._n_channels == 1:  # Univariate
            assert self.variant in [1, 2, 3]
            (
                self.sfa_transforms,
                self.sfa_clfs,
                self.sax_transforms,
                self.sax_clfs,
            ) = self._build_univariate_ensemble(np.squeeze(X), y)
        else:  # Multivariate

            if self.variant in [1, 2, 3]:  # Concatenate
                X_concat = X.reshape(*X.shape[:-2], -1)
                (
                    self.sfa_transforms,
                    self.sfa_clfs,
                    self.sax_transforms,
                    self.sax_clfs,
                ) = self._build_univariate_ensemble(X_concat, y)

            elif self.variant in [4, 5, 6, 7, 8, 9]:  # Ensemble
                (
                    self.sfa_transforms,
                    self.sfa_clfs,
                    self.sax_transforms,
                    self.sax_clfs,
                ) = self._build_dimension_ensemble(X, y)

    def _build_univariate_ensemble(self, X, y):
        """Build RED CoMETS ensemble from the univariate training set (X, y).

        Parameters
        ----------
        X : np.ndarray
            2D np.ndarray of shape (n_cases, n_timepoints)
            The training data.
        y : np.ndarray
            1D np.ndarray of shape (n_cases)
            The class labels.

        Returns
        -------
        sfa_transforms :
            List of ``SFA()`` instances with random word length and alpabet size
        sfa_clfs :
            List of ``(RandomForestClassifier(), weight)`` tuples fitted on `SFA`
            transformed training data
        sax_transforms :
            List of ``SAX()`` instances with random word length and alpabet size
        sax_clfs :
            List of ``(RandomForestClassifier(), weight)`` tuples fitted on `SAX`
            transformed training data
        """
        _check_soft_dependencies(
            "imbalanced-learn",
            package_import_alias={"imbalanced-learn": "imblearn"},
            severity="error",
            obj=self,
        )

        from imblearn.over_sampling import SMOTE, RandomOverSampler

        X = Normalizer().fit_transform(X).squeeze()

        if self.variant in [1, 2, 3]:
            perc_length = self.perc_length / self._n_channels
        else:
            perc_length = self.perc_length

        n_lenses = max(2 * int(perc_length * X.shape[1] // 100), 2)

        min_neighbours = min(Counter(y).items(), key=lambda k: k[1])[1] #These are determined by how many times each class label appears
        max_neighbours = max(Counter(y).items(), key=lambda k: k[1])[1]

        if self.debug_mode:
            print(f"min_neighbours: {min_neighbours}")
            print(f"max_neighbours: {max_neighbours}")
            print("\n")

        if min_neighbours == max_neighbours: #NOTE: SMOTE could add extra data depending on if we need too because of the min_neighbors requirement.
            X_smote = X
            y_smote = y

        else:
            if min_neighbours > 5:
                min_neighbours = 6
            try:
                X_smote, y_smote = SMOTE(
                    sampling_strategy="all",
                    k_neighbors=NearestNeighbors(
                        n_neighbors=min_neighbours - 1, n_jobs=self.n_jobs
                    ),
                    random_state=self.random_state,
                ).fit_resample(X, y)

            except ValueError:
                X_smote, y_smote = RandomOverSampler(
                    sampling_strategy="all", random_state=self.random_state
                ).fit_resample(X, y)
                
        if self.debug_mode:
            print(f"X_smote shape: {X_smote.shape}")
            print(f"y_smote shape: {y_smote.shape}")
            print("\n")

        lenses = self._get_random_lenses(X_smote, n_lenses)
        sfa_lenses = lenses[: n_lenses // 2]
        sax_lenses = lenses[n_lenses // 2 :]

        cv = np.min([5, len(y_smote) // len(list(set(y_smote)))])

        sfa_transforms = [
            SFA(
                word_length=w,
                alphabet_size=a,
                window_size=X_smote.shape[1], #NOTE: SMOTE could add extra data
                binning_method="equi-width",
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
            for w, a in sfa_lenses
        ]

        sfa_clfs = []
        for sfa in sfa_transforms:
            sfa_dics = sfa.fit_transform(X_smote, y_smote) #NOTE: SMOTE could add extra data
            X_sfa = np.array([sfa.word_list(list(d.keys())[0]) for d in sfa_dics[0]])

            if self.debug_mode:
                print(f"X_sfa shape: {X_sfa.shape}")

            if self.static is not None:
                X_sfa = np.hstack([X_sfa, self._prepare_static(X_sfa)])

            rf = RandomForestClassifier(
                n_estimators=self.n_trees,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            rf.fit(X_sfa, y_smote)

            if self.variant == 1:
                weight = 1
            elif self.variant == 3:
                weight = cross_val_score(
                    rf, X_sfa, y_smote, cv=cv, n_jobs=self.n_jobs
                ).mean()

            else:
                weight = None

            sfa_clfs.append((rf, weight))

        sax_transforms = [
            SAX(n_segments=w, alphabet_size=a, znormalized=True) for w, a in sax_lenses
        ]

        sax_clfs = []
        for X_sax in self._parallel_sax(sax_transforms, X_smote):
            if self.static is not None:
                X_sax = np.hstack([X_sax, self._prepare_static(X_sax)])
            rf = RandomForestClassifier(
                n_estimators=self.n_trees,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            rf.fit(X_sax, y_smote)

            if self.variant == 1:
                weight = 1
            elif self.variant == 3:
                weight = cross_val_score(
                    rf, X_sax, y_smote, cv=cv, n_jobs=self.n_jobs
                ).mean()
            else:
                weight = None

            sax_clfs.append((rf, weight))

        return sfa_transforms, sfa_clfs, sax_transforms, sax_clfs

    def _build_dimension_ensemble(self, X, y):
        """Build an ensemble of univariate RED CoMETS ensembles over dimensions.

        Parameters
        ----------
        X : np.ndarray
            3D np.ndarray of shape (n_cases, n_channels, n_timepoints)
            The training data.
            ``n_channels > 1``
        y : np.ndarray
            1D np.ndarray of shape (n_cases)
            The class labels.

        Returns
        -------
        sfa_transforms : list
            List of lists of ``SFA()`` instances with random word length and alpabet
            size
        sfa_clfs : list
            List of lists of ``(RandomForestClassifier(), weight)`` tuples fitted on
            `SFA` transformed training data
        sax_transforms : list
            List of lists of ``SAX()`` instances with random word length and alpabet
            size
        sax_clfs : list
            List of lists ``(RandomForestClassifier(), weight)`` tuples fitted on `SAX`
            transformed training data
        """
        sfa_transforms = []
        sfa_clfs = []
        sax_transforms = []
        sax_clfs = []

        for d in range(self._n_channels):
            X_d = X[:, d, :]

            (
                sfa_trans_d,
                sfa_clfs_d,
                sax_trans_d,
                sax_clfs_d,
            ) = self._build_univariate_ensemble(X_d, y)

            sfa_transforms.append(sfa_trans_d)
            sfa_clfs.append(sfa_clfs_d)

            sax_transforms.append(sax_trans_d)
            sax_clfs.append(sax_clfs_d)

        return sfa_transforms, sfa_clfs, sax_transforms, sax_clfs

    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : np.ndarray
            3D np.ndarray of shape (n_cases, n_channels, n_timepoints)
            The data to make predictions for.

        Returns
        -------
        y : np.ndarray
            1D np.ndarray of shape (n_cases)
            Predicted class labels.
        """
        return np.array(
            [self.classes_[i] for i in self._predict_proba(X).argmax(axis=1)]
        )

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : np.ndarray
            3D np.ndarray of shape (n_cases, n_channels, n_timepoints)
            The data to make predict probabilities for.

        Returns
        -------
        y : np.ndarray
            2D np.ndarray of shape (n_cases, n_classes_)
            Predicted probabilities using the ordering in ``classes_``.
        """
        if X.shape[1] == 1:  # Univariate
            return self._predict_proba_unvivariate(np.squeeze(X))
        else:  # Multivariate
            if self.variant in [1, 2, 3]:  # Concatenate
                X_concat = X.reshape(*X.shape[:-2], -1)
                return self._predict_proba_unvivariate(X_concat)
            elif self.variant in [4, 5, 6, 7, 8, 9]:
                return self._predict_proba_dimension_ensemble(X)  # Ensemble

    def _predict_proba_unvivariate(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in univariate X.

        Parameters
        ----------
        X : np.ndarray
            2D np.ndarray of shape (n_cases, n_timepoints)
            The data to make predict probabilities for.

        Returns
        -------
        y : np.ndarray
            2D np.ndarray of shape (n_cases, n_classes_)
            Predicted probabilities using the ordering in ``classes_``.
        """
        X = Normalizer().fit_transform(X).squeeze()

        pred_mat = np.zeros((X.shape[0], self.n_classes_))

        for sfa, (rf, weight) in zip(self.sfa_transforms, self.sfa_clfs):
            sfa_dics = sfa.transform(X)
            X_sfa = np.array([sfa.word_list(list(d.keys())[0]) for d in sfa_dics[0]])
            if self.static is not None:
                X_sfa = np.hstack([X_sfa, self._prepare_static(X_sfa)])
            rf_pred_mat = rf.predict_proba(X_sfa)

            if self.variant == 2:
                weight = np.mean(rf_pred_mat.max(axis=1))

            pred_mat += rf_pred_mat * weight

        for X_sax, (rf, weight) in zip(
            self._parallel_sax(self.sax_transforms, X), self.sax_clfs
        ):
            if self.static is not None:
                X_sax = np.hstack([X_sax, self._prepare_static(X_sax)])
            rf_pred_mat = rf.predict_proba(X_sax)

            if self.variant == 2:
                weight = np.mean(rf_pred_mat.max(axis=1))

            pred_mat += rf_pred_mat * weight

        pred_mat /= np.sum(pred_mat, axis=1).reshape(-1, 1)  # Rescales rows to sum to 1
        return pred_mat

    def _predict_proba_dimension_ensemble(self, X) -> np.ndarray:
        """Predicts labels probabilities using ensemble over the dimensions.

        Parameters
        ----------
        X : np.ndarray
            3D np.ndarray of shape (n_cases, n_channels, n_timepoints)
            The data to make predict probabilities for.
            ``n_channels > 1``

        Returns
        -------
        y : np.ndarray
            2D np.ndarray of shape (n_cases, n_classes_)
            Predicted probabilities using the ordering in ``classes_``.
        """
        X = Normalizer().fit_transform(X)

        ensemble_pred_mats = None

        for d in range(self._n_channels):
            sfa_transforms = self.sfa_transforms[d]
            sfa_clfs = self.sfa_clfs[d]

            sax_transforms = self.sax_transforms[d]
            sax_clfs = self.sax_clfs[d]

            X_d = X[:, d, :]

            if self.variant in [6, 7, 8, 9]:
                dimension_pred_mats = None
            for sfa, (rf, _) in zip(sfa_transforms, sfa_clfs):
                sfa_dics = sfa.transform(X_d)
                X_sfa = np.array(
                    [sfa.word_list(list(d.keys())[0]) for d in sfa_dics[0]]
                )

                rf_pred_mat = rf.predict_proba(X_sfa)

                if self.variant in [4, 5]:
                    if ensemble_pred_mats is None:
                        ensemble_pred_mats = [rf_pred_mat]
                    else:
                        ensemble_pred_mats = np.concatenate(
                            (ensemble_pred_mats, [rf_pred_mat])
                        )

                elif self.variant in [6, 7, 8, 9]:
                    if dimension_pred_mats is None:
                        dimension_pred_mats = [rf_pred_mat]
                    else:
                        dimension_pred_mats = np.concatenate(
                            (dimension_pred_mats, [rf_pred_mat])
                        )

            for X_sax, (rf, _) in zip(
                self._parallel_sax(sax_transforms, X_d), sax_clfs
            ):
                rf_pred_mat = rf.predict_proba(X_sax)

                if self.variant in [4, 5]:
                    if ensemble_pred_mats is None:
                        ensemble_pred_mats = [rf_pred_mat]
                    else:
                        ensemble_pred_mats = np.concatenate(
                            (ensemble_pred_mats, [rf_pred_mat])
                        )

                elif self.variant in [6, 7, 8, 9]:
                    if dimension_pred_mats is None:
                        dimension_pred_mats = [rf_pred_mat]
                    else:
                        dimension_pred_mats = np.concatenate(
                            (dimension_pred_mats, [rf_pred_mat])
                        )

            if self.variant in [6, 7, 8, 9]:
                if self.variant in [6, 7]:
                    fused_dimension_pred_mat = np.sum(dimension_pred_mats, axis=0)
                elif self.variant in [8, 9]:
                    weights = np.array(
                        [np.mean(mat.max(axis=1)) for mat in dimension_pred_mats]
                    ).reshape(-1, 1)
                    fused_dimension_pred_mat = np.sum(
                        dimension_pred_mats * weights[:, np.newaxis], axis=0
                    )

                if ensemble_pred_mats is None:
                    ensemble_pred_mats = [fused_dimension_pred_mat]
                else:
                    ensemble_pred_mats = np.concatenate(
                        (ensemble_pred_mats, [fused_dimension_pred_mat])
                    )

        if self.variant in [4, 6, 7]:
            pred_mat = np.sum(np.array(ensemble_pred_mats), axis=0)
        elif self.variant in [5, 8, 9]:
            weights = np.array(
                [np.mean(mat.max(axis=1)) for mat in ensemble_pred_mats]
            ).reshape(-1, 1)
            pred_mat = np.sum(ensemble_pred_mats * weights[:, np.newaxis], axis=0)
        pred_mat /= np.sum(pred_mat, axis=1).reshape(-1, 1)  # Rescales rows to sum to 1
        return pred_mat

    def _get_random_lenses(self, X, n_lenses):
        """Randomly select <word length, alphabet size> pairs.

        Parameters
        ----------
        X : np.ndarray
            3D np.ndarray of shape (n_cases, n_channels, n_timepoints)
            The training data.
        n_lenses : int
            Number of lenses to select.

        Returns
        -------
        lenses : list of list
            Randomly selected lenses.
        """
        maxCoof = 130
        if X.shape[1] < maxCoof:
            maxCoof = X.shape[1] - 1

        n_segments = range(3, maxCoof)

        maxBin = 26
        alphas = range(3, maxBin)

        rng = check_random_state(self.random_state)
        lenses = np.transpose(
            [rng.choice(n_segments, size=n_lenses), rng.choice(alphas, size=n_lenses)]
        ).tolist()

        return lenses

    def _parallel_sax(self, sax_transforms, X):
        """Apply multiple SAX transforms to X in parallel.

        Parameters
        ----------
        sax_transforms : list
            List of ``SAX()`` instances
        X : np.ndarray
            2D np.ndarray of shape (n_cases, n_timepoints)
            The data to transform.
        """

        def _sax_wrapper(sax):
            return np.squeeze(sax.fit_transform(X))

        sax_parallel_res = Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)(
            delayed(_sax_wrapper)(sax) for sax in sax_transforms
        )
        return sax_parallel_res

    def _compute_rf_proximities(self, rf: RandomForestClassifier, X: np.ndarray) -> np.ndarray: #NOTE: Change this Proximity to RFGAP process
        """
        Compute the proximity matrix for samples in X using a single fitted RandomForestClassifier.
        Each element (i,j) is the fraction of trees in which samples i and j share the same leaf.
        """
        leaves = rf.apply(X)  # shape: (n_samples, n_trees)
        # n_samples, n_trees = leaves.shape
        # proximities = np.zeros((n_samples, n_samples))
        # for tree_idx in range(n_trees):
        #     leaf_ids = leaves[:, tree_idx]
        #     unique_leaves = np.unique(leaf_ids)
        #     for leaf in unique_leaves:
        #         idx = np.where(leaf_ids == leaf)[0]
        #         for i in idx:
        #             proximities[i, idx] += 1
        # proximities /= n_trees
        return proximities

    def get_ensemble_proximities(self, X: np.ndarray, group: str = "all") -> np.ndarray:
        """
        Compute the aggregated proximity matrix for the ensemble of random forests in the model.
        
        Parameters
        ----------
        X : np.ndarray
            Input data appropriate for the model. The data is first transformed using the 
            corresponding SFA or SAX transform.
        group : str, default="all"
            Which model group to use: "sfa", "sax", or "all" (aggregates both).
        
        Returns
        -------
        np.ndarray
            The aggregated proximity matrix computed as the weighted average of proximities 
            from the ensemble models.
        
        Note
        ----
        The method uses each (rf, weight) pair from self.sfa_clfs and self.sax_clfs along with 
        their corresponding transformations from self.sfa_transforms and self.sax_transforms.
        """
        proximities_list = []
        weights_list = []

        if group in ("sfa", "all"):
            for (rf, weight), sfa in zip(self.sfa_clfs, self.sfa_transforms):
                # Transform X using the SFA transform
                sfa_dics = sfa.transform(X) # SEE what this returns
                X_trans = np.array([sfa.word_list(list(d.keys())[0]) for d in sfa_dics[0]]) #NOTE: TWO EXTRA ROWS HERE - With GunPoint 
                if self.static is not None:
                    X_trans = np.hstack([X_trans, self._prepare_static(X_trans)])
                prox = self._compute_rf_proximities(rf, X_trans)
                proximities_list.append(prox)
                weights_list.append(weight if weight is not None else 1)

        if group in ("sax", "all"):
            for (rf, weight), sax in zip(self.sax_clfs, self.sax_transforms):
                # Transform X using the SAX transform
                X_trans = sax.fit_transform(X)
                X_trans = np.squeeze(X_trans)
                if self.static is not None:
                    X_trans = np.hstack([X_trans, self._prepare_static(X_trans)])
                prox = self._compute_rf_proximities(rf, X_trans)
                proximities_list.append(prox)
                weights_list.append(weight if weight is not None else 1)

        if not proximities_list:
            raise ValueError("No models available for group: " + group)


        # Maybe there is a more intuitive way to do this
        total_weight = sum(weights_list)
        aggregated = sum(prox * w for prox, w in zip(proximities_list, weights_list)) / total_weight
        return aggregated
    
    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        dict
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return {
            "variant": 3,
            "n_trees": 3,
        }
