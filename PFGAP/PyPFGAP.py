import os
import numpy as np
import subprocess
import Application.proxUtil as proxUtil

class PyPFGAP:
    def __init__(self, train_file = "~/data/train", test_file = "~/data/test", num_trees=18, r=5):
        """
        Initialize the PyPFGAP class.

        Parameters
        ----------
        train_file : str
            Path to training data file.
        test_file : str
            Path to the testing data file.
        num_trees : int, default=18
            Number of trees for the proximity calculation.
        r : int, default=5
            Parameter for proximity calculation.
        """
        self.train_file = train_file
        self.test_file = test_file
        self.num_trees = num_trees
        self.r = r
        self.prox = None
        self.labels = None

    def fit(self, X_train, y_train, X_test = None, y_test = None, static=None):
        """
        Runs the Java process on training and test data.
        """

        if X_test is None:
            X_test = X_train
        if y_test is None:
            y_test = y_train

        try:
            # Paths for train/test TSVs
            tsv_train = os.path.expanduser(self.train_file)
            tsv_test = os.path.expanduser(self.test_file)

            # Flatten multidimensional arrays
            if X_train.ndim > 2:
                X_train = X_train.reshape(X_train.shape[0], -1)
            if X_test.ndim > 2:
                X_test = X_test.reshape(X_test.shape[0], -1)

            # Ensure y is a column vector
            if y_train.ndim == 1:
                y_train = y_train.reshape(-1, 1)
            if y_test.ndim == 1:
                y_test = y_test.reshape(-1, 1)

            # Combine y as first column and X horizontally
            data_train = np.hstack([y_train, X_train])
            data_test  = np.hstack([y_test,  X_test])
        except Exception as e:
            print(f"Error reshaping data: {e}")
            return None

        # Helper to write a TSV
        def _write_tsv(path, data):
            d = os.path.dirname(path)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            # overwrite / create file
            open(path, 'w').close()
            np.savetxt(path, data, delimiter='\t', fmt='%s')

        # Write train and test TSVs
        _write_tsv(tsv_train, data_train)
        _write_tsv(tsv_test,  data_test)

        # Call the proximity calculator
        proxUtil.getProx(
            tsv_train,
            tsv_test,
            num_trees=self.num_trees,
            r=self.r
        )
        
        prox,labels = proxUtil.getProxArrays()

        if os.path.exists(tsv_train):
            os.remove(tsv_train)
        if os.path.exists(tsv_test):
            os.remove(tsv_test)

        self.prox = prox

        return prox
        
    def get_proximities(self):
    
        return self.prox

# example use:
# mytrain = "/home/ben/Documents/classes/CS7675/Project/UCRArchive_2018/ArrowHead/ArrowHead_TRAIN.tsv"
# mytest = "/home/ben/Documents/classes/CS7675/Project/UCRArchive_2018/ArrowHead/ArrowHead_TEST.tsv"
# model = PyPFGAP(mytrain, mytest, num_trees=18, r=5)
# model.fit()
# prox, labels = model.get_proximities()