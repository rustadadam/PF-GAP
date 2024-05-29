# PF-GAP

This repository provides an implementation of PF-GAP. This implementation makes extensive use of the java-based implementation of proximity forests, available at the following link: https://github.com/fpetitjean/ProximityForest/tree/master

# Requirements

Java 17, python 3.

# Using PF-GAP

It is not necessary to clone this repository to use PF-GAP. One only needs the files in the "Application" folder, namely the .jar file and the proxUtil.py file. The proxUtil.py file contains the function "getProx" which calls the .jar file, applying PF-GAP to the training (and test) files specified in the function arguments. Other parameters may be passed to this function as well, including the desired number of trees and r parameter. The proxUtil.py file is not strictly necessary, but rather provides a convenient mechanism for calling PF-GAP in python.

The .jar file, when run, creates .txt files, one called "ForestProximities.txt" and the other called "ytrain.txt," which files contain the array of forest proximities and the training labels, respectively. These can be read into python using the "getProxArrays" function, whose default arguments are the file names just mentioned. The forest proximity array can be symmetrized by means of the "SymmetrizeProx" function, and within-class outlier scores can be obtained using the "getOutlierScores" function.
