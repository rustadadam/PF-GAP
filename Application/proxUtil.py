import subprocess
import numpy as np

def getProx(trainfile, testfile, out="output", repeats=1, num_trees=10, r=5, on_tree="true", shuffle="true", export=1, verbosity=1, csv_has_header="false", target_column="first"):
    msgList = ['java', '-jar', '-Xmx1g', 'PFGAPjava.jar']
    # Mostly, trainfile, testfile, num_trees, and r are what will be tampered with.
    msgList.extend(["-train=" + trainfile])
    msgList.extend(["-test=" + testfile])
    msgList.extend(["-out=" + out])
    msgList.extend(["-repeats=" + str(repeats)])
    msgList.extend(["-trees=" + str(num_trees)])
    msgList.extend(["-r=" + str(r)])
    msgList.extend(["-on_tree=" + on_tree])
    msgList.extend(["-shuffle=" + shuffle])
    msgList.extend(["-export=" + str(export)])
    msgList.extend(["-verbosity=" + str(verbosity)])
    msgList.extend(["-csv_has_header=" + csv_has_header]) # we mean this to work primarily with tsv files, actually.
    msgList.extend(["-target_column=" + target_column])

    subprocess.call(msgList)
    return


def getProxArrays(proxfile="ForestProximities.txt", yfile="ytrain.txt"):
    f1 = open(proxfile)
    f2 = f1.read()
    f2 = f2.replace("{","[")
    f2 = f2.replace("}","]")
    #exec("Arr = np.array(" + f2 + ")")
    proxArr = eval("np.array(" + f2 + ")")
    #f1.close()
    f1 = open(yfile)
    f2 = f1.read()
    f2 = f2.replace("{", "[")
    f2 = f2.replace("}", "]")
    yArr = eval("np.array(" + f2 + ")")
    return proxArr, yArr


def SymmetrizeProx(Pmat):
    PMat = (Pmat + Pmat.transpose()) / 2
    return PMat


# Here is a function to compute within-class outliers.
def getRawOutlierScores(proxArray, ytrain):
    # the proxArray should be symmetrized first.
    uniqueLabels = np.unique(ytrain)
    mydict = {label:[] for label in uniqueLabels}
    # find out which indices have which class labels:
    for i in range(ytrain.shape[0]):
        label = ytrain[i]
        mydict[label].extend([i])
    # now find the outlier score for each datapoint
    scores = []
    for i in range(ytrain.shape[0]):
        label = ytrain[i]
        PiList = [proxArray[i][k] for k in mydict[label]]
        Pn = np.sum(PiList)
        if Pn == 0:
            print("Warning: index " + str(i) + " has within-class proximity of 0. Changing to 1e-6.")
            Pn = 1e-6
        scores.extend([ytrain.shape[0]/Pn])
    print(scores)
    return scores



# example use:
'''mytrain = "/home/ben/Documents/classes/CS7675/Project/UCRArchive_2018/ArrowHead/ArrowHead_TRAIN.tsv"
mytest = "/home/ben/Documents/classes/CS7675/Project/UCRArchive_2018/ArrowHead/ArrowHead_TEST.tsv"
getProx(mytrain, mytest, num_trees=18, r=5)
prox,labels = getProxArrays()
prox = SymmetrizeProx(prox)
getRawOutlierScores(prox,labels)'''
