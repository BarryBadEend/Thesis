import scipy.io
import h5py
import numpy as np
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import stats

def zscorenorm(unnormalizeddata):
    #the 4 dimensions are trials: 1000, channels: 64, frequency: 73 and time: 41
    #time, frequencies, channels, trials
    print(unnormalizeddata.shape)
    result = stats.zscore(unnormalizeddata, axis=2)
    return(result)

def createfolds(wholedata):
    folds = []
    for x in range(5):
        folds.append(wholedata[x*200:200*(x+1)])
    return(folds)
def testfolds(foldsx,foldsy):
    results = []
    for x in range(len(foldsx)):
        score = 0
        testdata = np.array(foldsx[x])
        testresults = np.array(foldsy[x])
        traindata = []
        trainresults = []
        for y in range(len(foldsx)):
            if y != x:
                traindata.append(foldsx[y])
                trainresults.append(foldsy[y])
        traindata = np.array(traindata)
        trainresults = np.array(trainresults)
        trainresults = np.reshape(trainresults,800)
        traindata = np.reshape(traindata,(800,totaaldimensies))
        clf = svm.SVC()
        clf.fit(traindata,trainresults)
        print(testdata.shape)
        predictions = clf.predict(testdata)
        for y in range(len(predictions)):
            if int(predictions[y]) == int(testresults[y]):
                score +=1
        print("fold "+ str(x+1) + " has a performance of "+ str(score/len(predictions)))
        results.append(score/len(predictions))
    return(results)
    
ruwedata = h5py.File('ASEEGdataRawFREQ.mat', 'r')
parameters = scipy.io.loadmat('AS_MAT.mat')
stimpresent = np.array(parameters['MAT'][0])
"""stimpresent is an array and contains 100 0's and 1's representing wether or not the stimulus is present"""
EEG = np.array(ruwedata['data']) 
EEG = EEG[:,12:30,:,:]



EEGSinglemomentnorm = zscorenorm(EEG)
totaaldimensies = 1
#print(EEGSinglemoment.shape)

for x in range(3):
    totaaldimensies *= EEGSinglemomentnorm.shape[x]
EEGSVM = np.reshape(EEGSinglemomentnorm,(1000,totaaldimensies))
EEGSVM, stimpresent = shuffle(EEGSVM,stimpresent,random_state=1)
#EEGSVM = zscorenorm(EEGSVM)
xfolds = createfolds(EEGSVM)
yfolds = createfolds(stimpresent)
testresultaten = testfolds(xfolds,yfolds)
totaal = 0
totaal = sum(testresultaten)
average = totaal / len(testresultaten)
print(average)
