import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from varmax import Varmax
from hmm import HMM
from anomalydetector import AnomalyDetector

detector = Varmax((3, 3))
#detector = HMM(5)

def read_file(filename):
    """
        Read one file
    """
    print("Reading data file " + filename)
    data = np.fromfile(filename, dtype=np.dtype('float64'))
    return data.reshape(-1, 1500)

def read_files(directory):
    """
        Read all files from a directory
    """
    print("Reading data files in directory " + directory)
    os.chdir(directory)
    data = [np.fromfile(fname, dtype=np.dtype('float64')) for fname in os.listdir()]
    print(str(len(data)) + " files read")
    data = np.concatenate(data)

#    print("Files read" + str(data.shape))
#data = data.reshape(50,-1)
    return data.reshape(-1, 1500)

def split_data(data):
    """
        Split into train set and test set
    """
#print("Creating train and test set")
    [train_data, test_data] = train_test_split(data)
    print("Train set: " + str(train_data.shape))
    print("Test set: " + str(test_data.shape))
    return (train_data, test_data)


def standardize(data):
    """
        Standardize the dataset (mean = 0, variance = 1)
        Necessary for the PCA to be useful
    """
    train_data = data[0]
    test_data = data[1]
    scaler = StandardScaler()
    scaler.fit(train_data)

    # Both dataset are normalized (necessary for the test set?)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    return (train_data, test_data)

def do_PCA(data, n_components):
    """
        do the PCA
        n_components is the minimal amount of variance explained
    """
    train_data = data[0]
    test_data = data[1]
    before = train_data.shape[1]
    print("Computing PCA…")
    pca = PCA(n_components, svd_solver="full")
    train_pca = pca.fit_transform(train_data)
    print("Features number: " + str(before) + " -> " + str(train_pca.shape[1]))
    return (train_pca, pca.fit_transform(test_data))

# New train set with reduced features
(g_train_data, g_test_data) = do_PCA(standardize(split_data(read_files("mini-data"))), 0.95)
#(train_data, test_data) = do_PCA(standardize(split_data(read_file("data/1530056352292"))), 0.95)
print("Learning…")
#detector.learn(train_pca, None)
