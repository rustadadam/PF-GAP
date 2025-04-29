This archive contains python notebook files, data files, and other files. Below
is a brief description of each file.


Python notebook files
1. The Demo.ipynb file demonstrates the usage of the PF-GAP model.
2. The getLabels.ipynb assigns numeric labels for the classes.
3. The techutil.ipynb file creates a binary model for tech/utility stocks only.
4. The TechUtil.ipynb file examines the previous model further: outlier scores.

Data files
1. The GunPoint_TEST.tsv and GunPoint_TRAIN.tsv files are from the UCR 2018 
   archive and are used in the Demo.ipynb file to show how PF-GAP is used.
2. The Labeled_11-01-2024_GICS.csv file contains label information for the 
   S&P500 stocks on November 1st, 2024
3. The S&P500_2024-10.csv file contains the stock price (percentage) time series
   for each of the stocks in the previous file for October 2024. (yfinance.)

Other Files
1. The PFGAP.jar file is used to run the PF-GAP model in java. Models are saved
   as serialized .ser files. proximities for training data is saved as .txt.
2. The PFGAP_eval.jar file can be used to deserialize saved .ser files so that
   saved models can be evaluated on data.
3. The proxUtil.py file contains python wrapper commands in order to call the 
   PF-GAP model from python. See the Demo.ipynb file for a tutorial.
