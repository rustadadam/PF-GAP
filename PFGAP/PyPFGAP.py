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
        try:
            tsv_path = os.path.expanduser(self.train_file)

            if X.ndim > 2:
                X = X.reshape(X.shape[0], -1)

            # Ensure y is a column vector
            if y.ndim == 1:
                y = y.reshape(-1, 1)

            # Combine y as first column and X horizontally
            data = np.hstack([y, X])
        except Exception as e:
            print(f"Error reshaping data: {e}")
            return None
        
        # Save combined data to a TSV file
        # ensure the output directory exists and create an empty file
        dirpath = os.path.dirname(tsv_path)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        open(tsv_path, 'w').close()
        np.savetxt(tsv_path, data, delimiter='\t', fmt='%s')
    

        proxUtil.getProx(tsv_path, None, num_trees=18, r=5)
        prox,labels = proxUtil.getProxArrays()

        if os.path.exists(tsv_path):
            os.remove(tsv_path)

        self.pox = prox

        return prox
        

        

    def get_proximities(self):
    
        return self.prox

# example use:
# mytrain = "/home/ben/Documents/classes/CS7675/Project/UCRArchive_2018/ArrowHead/ArrowHead_TRAIN.tsv"
# mytest = "/home/ben/Documents/classes/CS7675/Project/UCRArchive_2018/ArrowHead/ArrowHead_TEST.tsv"
# model = PyPFGAP(mytrain, mytest, num_trees=18, r=5)
# model.fit()
# prox, labels = model.get_proximities()