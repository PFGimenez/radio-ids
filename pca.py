import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from varmax import Varmax
from hmm import HMM
from anomalydetector import *
import os

detector = Varmax((3,3))

os.chdir("mini-data")
#data = [np.fromfile(fname, dtype=np.dtype('float64')) for fname in os.listdir()]
#data = np.concatenate(data)
data = np.fromfile("1530056352292", dtype=np.dtype('float64'))

print("Files read" + str(data.shape))
#data = data.reshape(50,-1)
data = data.reshape(-1,1500)

#print("Creating train and test set")
[train_data, test_data] = train_test_split(data)
print("Train set: " + str(train_data.shape))
print("Test set: " + str(test_data.shape))

print("Features number before PCA : " + str(data.shape[1]))
print("Computing PCA…")
# the minimal amount of variance explained
n_components = 0.95

#train_data = data
#test_data = train_data

# Standardize the dataset (mean = 0, variance = 1)
# Necessary for the PCA to be useful
scaler = StandardScaler()
scaler.fit(train_data)

# Both dataset are normalized (necessary for the test set?)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

# Doing PCA
pca = PCA(n_components, svd_solver="full")

# New train set with reduced features
train_pca = pca.fit_transform(train_data)
print("Features number after PCA : " + str(train_pca.shape[1]))

print("Learning…")
detector.learn(data, None)
