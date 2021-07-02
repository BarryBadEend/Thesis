import scipy.io
import h5py
from operator import add
import numpy as np
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import stats


ruwedata = h5py.File('ASEEGdataRawFREQ.mat', 'r')
parameters = scipy.io.loadmat('AS_MAT.mat')
stimpresent = np.array(parameters['MAT'][0])
"""stimpresent is an array and contains 100 0's and 1's representing wether or not the stimulus is present"""
EEG = np.array(ruwedata['data'])
EEG = EEG[:,7:12,:,:]
EEGpres = []
EEGabs = []
totaaldimensies = 1
for x in range(3):
    totaaldimensies *= EEG.shape[x]
EEG = EEG.reshape(1000,totaaldimensies)
for x in range(1000):
    if(stimpresent[x]==1):
        EEGpres.append(EEG[x])
    if(stimpresent[x]==0):
        EEGabs.append(EEG[x])

EEGpres = np.array(EEGpres)
EEGabs = np.array(EEGabs)
EEGpres = EEGpres.reshape(41,5,64,500)
EEGabs = EEGabs.reshape(41,5,64,500)
EEGpres = EEGpres.mean(axis = 2)
EEGpres = EEGpres.mean(axis = 1)
EEGabs = EEGabs.mean(axis = 2)
EEGabs = EEGabs.mean(axis = 1)
stdpres = []
stdabs = []
for x in range(41):
    stdpres.append(np.std(EEGpres[x]))
for x in range(41):
    stdabs.append(np.std(EEGabs[x]))
timepoints = []
for x in range(41):
    timepoints.append(x)
timepoints = np.array(timepoints) 
stdpres = np.array(stdpres)
EEGpres = EEGpres.mean(axis = 1)
EEGabs = EEGabs.mean(axis = 1)
EEGdif = []
for x in range(41):
    EEGdif.append(EEGpres[x]-EEGabs[x])
print(EEGdif)
print(timepoints.shape)
print(EEGpres.shape)
print(stdpres.shape)
plt.errorbar(timepoints, EEGpres,stdpres, ecolor = 'red')
plt.title('present stimuli')
plt.ylabel('average value')
plt.xlabel('timepoint')
plt.ylim([0,3])
plt.show()
plt.clf()
plt.errorbar(timepoints,EEGabs,stdabs, ecolor = 'red')
plt.title('present stimuli')
plt.ylabel('average value')
plt.xlabel('timepoint')
plt.title('absent stimuli')
plt.ylim([0,3])
plt.show()
plt.clf()
plt.errorbar(timepoints,EEGdif)
plt.title('difference per timepoint')
plt.ylabel('difference')
plt.xlabel('timepoints')
plt.show()

