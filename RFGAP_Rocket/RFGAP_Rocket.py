from aeon.transformations.collection.convolution_based import MultiRocket, MiniRocket
from rfgap import RFGAP
import numpy as np

#Build the RFGAP Rocket Class
class RFGAP_Rocket():
    """RFGAP Rocket Class. 
    Time series classification using RFGAP and Rocket.
    """

    def __init__(self, classification_type = True, rocket = "Multi", **kwargs):
        """Initialize the RFGAP Rocket Class.

        Parameters
        ----------
        classification : bool, optional
            If True, use classification. If False, use regression. The default is True.

        rocket : str, optional
            The type of Rocket to use. The default is "Multi".
            Options are "Multi" or "Mini".

        kwargs : dict, optional
            Additional arguments to pass to the Rocket classes.
            The default is {}.
        """
        if rocket.lower() == "multi":
            self.rocket = MultiRocket(**kwargs)
        elif rocket.lower() == "mini":
            self.rocket = MiniRocket(**kwargs)
        else:
            raise ValueError("rocket must be 'Multi' or 'Mini'")

        #Initialize rfgap
        self.rf_gap = RFGAP(prediction_type = classification_type)

        
    def fit(self, X, y, static = None):
        """Fit the RFGAP Rocket Class.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_timestamps, n_features)
            The input time series data.

        y : array-like, shape (n_samples,)
            The target labels.

        static : array-like, shape (n_samples, n_static_features), optional
            The static features. The default is None.
        """

        #Transform the time series data using Rocket
        self.rocket.fit(X)
        X_transformed = self.rocket.transform(X)

        #Append Static features to the transformed data
        if static is not None:
            X_transformed = np.concatenate((X_transformed, static), axis=1)

        #Fit the RFGAP model
        self.rf_gap.fit(X_transformed, y)

    def get_proximities(self):
        return self.rf_gap.get_proximities()
    
    def predict(self, X, static = None):
        """Predict using the RFGAP Rocket Class.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_timestamps, n_features)
            The input time series data.
        static : array-like, shape (n_samples, n_static_features), optional
            The static features. The default is None.
        Returns
        -------
        array-like, shape (n_samples,)
            The predicted labels.
        """
        #Transform the time series data using Rocket
        X_transformed = self.rocket.transform(X)

        #Append Static features to the transformed data
        if static is not None:
            X_transformed = np.concatenate((X_transformed, static), axis=1)

        #Predict using the RFGAP model
        return self.rf_gap.predict(X_transformed)
        