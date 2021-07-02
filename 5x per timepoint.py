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
    result = stats.zscore(unnormalizeddata, axis=1)
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
        trainresults = trainresults.reshape(800)
        traindata = traindata.reshape(800,totaaldimensies)
        clf = svm.SVC()
        clf.fit(traindata,trainresults)
        predictions = clf.predict(testdata)
        for y in range(len(predictions)):
            if predictions[y] == testresults[y]:
                score +=1
        #print("fold "+ str(x+1) + "has a performance of "+ str(score/len(predictions)))
        results.append(score/len(predictions))
    return(results)
    
ruwedata = h5py.File('ASEEGdataRawFREQ.mat', 'r')
parameters = scipy.io.loadmat('AS_MAT.mat')
stimpresent = np.array(parameters['MAT'][0])
"""stimpresent is an array and contains 100 0's and 1's representing wether or not the stimulus is present"""
EEG = np.array(ruwedata['data']) 
timeframes = []
results = []
standarddeviations = []
for x in range(41):
    timeframes.append(x+1)
for timepoint in range(41):
    EEGsinglemomentnorm = EEG[timepoint,12:30,:,:]
    print(EEGsinglemomentnorm.shape)
    totaaldimensies = 1
    for x in range(2):
        totaaldimensies *= EEGsinglemomentnorm.shape[x]
    EEGSVM = EEGsinglemomentnorm.reshape(1000,totaaldimensies)
    EEGSVM = zscorenorm(EEGSVM)
    #print(EEGSVM.shape)
    present = []
    absent = []
    pr = []
    ab = []
    for x in range(len(EEGSVM)):
        if(stimpresent[x]==1):
            present.append(EEGSVM[x])
            pr.append(stimpresent[x])
        else:
            absent.append(EEGSVM[x])
            ab.append(stimpresent[x])
    present, pr = shuffle(present, pr) #this shuffles the data using a nifty little method built into sklearn which makes sure the original x,y pairs stay in place.
    #here we create our fold
    absent, ab = shuffle(absent, ab)
    randomized = []
    randomizedstim = []
    pres = 0
    abs = 0
    for x in range(1000):
        if(x%2):
            randomized.append(present[pres])
            randomizedstim.append(pr[pres])
            pres +=1
        else:
            randomized.append(absent[abs])
            randomizedstim.append(ab[abs])
            abs +=1    
        
    xfolds = createfolds(randomized)
    yfolds = createfolds(randomizedstim)
    testresultaten = testfolds(xfolds,yfolds)
    totaal = 0
    for x in testresultaten:
        totaal = totaal + x 
    standarddeviations.append(np.std(testresultaten))
    average = totaal / len(testresultaten)
    results.append(average)
    print("the average performance over 5-cross validation on timepoint number "+ str(timepoint+1)+" is "+str(average))
plt.errorbar(timeframes, results, standarddeviations,ecolor = 'red')
plt.title('Performance per timepoint')
plt.xlabel('Individual timepoint')
plt.ylabel('Performance')
plt.show()
