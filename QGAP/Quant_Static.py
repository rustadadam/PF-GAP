# Quant model adjusted to handle Static variables. 

#aeon imports
from aeon.transformations.collection.interval_based import QUANTTransformer
from aeon.classification import BaseClassifier
from aeon.base._base import _clone_estimator
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np

#Overide QUANTClassifier to handle static variables
class StaticQuantClassifier():
    def __init__(
        self,
        interval_depth=6,
        quantile_divisor=4,
        estimator=None,
        class_weight=None, # or auto?
        random_state=None,
    ):
        self.interval_depth = interval_depth
        self.quantile_divisor = quantile_divisor
        self.estimator = estimator
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y, static = None):
        """Fit the estimator to training data.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_cases, n_channels, n_timepoints)
            The training data.
        y : 1D np.ndarray of shape (n_cases)
            The class labels for fitting, indices correspond to instance indices in X

        Returns
        -------
        self :
            Reference to self.
        """
        self._transformer = QUANTTransformer(
            interval_depth=self.interval_depth,
            quantile_divisor=self.quantile_divisor,
        )

        X_t = self._transformer.fit_transform(X, y)

        if static is not None:
            X_t = np.hstack((X_t, static))

        self._estimator = _clone_estimator(
            (
                ExtraTreesClassifier(
                    n_estimators=200,
                    max_features=0.1,
                    criterion="entropy",
                    class_weight=self.class_weight,
                    random_state=self.random_state,
                )
                if self.estimator is None
                else self.estimator
            ),
            self.random_state,
        )

        self._estimator.fit(X_t, y)

        #Create Proximities
        self.proximites = self._estimator.apply(X_t)

        return self

    def predict(self, X, static = None):
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape (n_cases, n_channels, n_timepoints)
            The testing data.

        Returns
        -------
        y : array-like of shape (n_cases)
            Predicted class labels.
        """
        X = self._transformer.transform(X)

        if static is not None:
            X = np.hstack((X, static))

        return self._estimator.predict(X)
    
    def predict_proximities(self, X, static = None):
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape (n_cases, n_channels, n_timepoints)
            The testing data.

        Returns
        -------
        y : array-like of shape (n_cases)
            Predicted class labels.
        """
        X = self._transformer.transform(X)

        if static is not None:
            X = np.hstack((X, static))

        return self._estimator.apply(X)

    def predict_proba(self, X, static = None):
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape (n_cases, n_channels, n_timepoints)
            The testing data.

        Returns
        -------
        y : array-like of shape (n_cases, n_classes_)
            Predicted probabilities using the ordering in classes_.
        """
        X = self._transformer.transform(X)

        if static is not None:
            X = np.hstack((X, static))

        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(X)
        else:
            dists = np.zeros((X.shape[0], self.n_classes_))
            preds = self._estimator.predict(X)
            for i in range(0, X.shape[0]):
                dists[i, self._class_dictionary[preds[i]]] = 1
            return dists
