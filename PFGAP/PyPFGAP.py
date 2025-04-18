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

    def run_Java(self, X, y, static = None):
        """
        Runs the Java proces
        """
        # Call the Java process to calculate proximities
        # Expand the user path and create the full file path
        tsv_path = os.path.expanduser(self.train_file)

        try:
            # Ensure y is a column vector
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            # Combine X and y horizontally
            data = np.hstack([X, y])
            # Save combined data to a TSV file
            np.savetxt(tsv_path, data, delimiter='\t', fmt='%s')
        except Exception as e:
            print(f"Error writing TSV file: {e}")

        proxUtil.getProx(tsv_path, None, num_trees=18, r=5)
        prox,labels = proxUtil.getProxArrays()

        if os.path.exists(tsv_path):
            os.remove(tsv_path)


        return prox
        

        

    def get_proximities(self):
        """
        Retrieve the proximity matrix and labels.

        Returns
        -------
        prox : np.ndarray
            The proximity matrix.
        labels : np.ndarray
            The labels corresponding to the proximity matrix.
        """
        # Load the proximity matrix and labels from files
        prox_file = "proximity_matrix.csv"
        labels_file = "labels.csv"

        try:
            self.prox = np.loadtxt(prox_file, delimiter=',')
            self.labels = np.loadtxt(labels_file, delimiter=',')
        except Exception as e:
            print(f"Error loading proximity or labels: {e}")
        finally:
            # Clean up temporary files
            if os.path.exists(prox_file):
                os.remove(prox_file)
            if os.path.exists(labels_file):
                os.remove(labels_file)

        return self.prox, self.labels

# example use:
# mytrain = "/home/ben/Documents/classes/CS7675/Project/UCRArchive_2018/ArrowHead/ArrowHead_TRAIN.tsv"
# mytest = "/home/ben/Documents/classes/CS7675/Project/UCRArchive_2018/ArrowHead/ArrowHead_TEST.tsv"
# model = PyPFGAP(mytrain, mytest, num_trees=18, r=5)
# model.fit()
# prox, labels = model.get_proximities()