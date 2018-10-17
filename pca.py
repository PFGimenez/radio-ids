import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from varmax import Varmax
from armax import Armax
from hmm import HMM
from anomalydetector import AnomalyDetector

detector = Armax((10, 3))
#detector = Varmax((3, 0))
#detector = HMM(5)

explained_variance = 0.95

# unused
#i_blocs = [(0, 500), (500, 1000), (1000, 1250), (1250, 1500), (1000, 1500)]
#frame_shape = (50, 1500)
#directory = "data"


def read_files_format(directory, i_blocs, frame_shape):
    num = 0
    files = sorted(os.listdir(path))
    for f in files:
        f = path + "/" + f
        if os.path.isfile(f):
            array = np.fromfile(f, dtype='float64')
            array = np.reshape(array, frame_shape)
            for i in range(len(i_blocs)):
                i_bloc = i_blocs[i]
                #for time in range(frame_shape[0]):
                #    for freq in range(i_bloc[0], i_bloc[1]):
                #        cumul = cumul + array[time][freq]
                #value = np.max(array[i_bloc[0]:i_bloc[1], ])
                #value = array[:, i_bloc[0]:i_bloc[1]].shape[0]
                array[0,0] = array[0,2]
                array[0,1] = array[0,2]

                value = np.max(array[:, i_bloc[0]:i_bloc[1]])
                value = (value + 80) * 255 / 80.0

                #value = np.mean(array[:, i_bloc[0]:i_bloc[1]])
                #value = (value + 80) * 255 / 80.0

                #value = np.max(np.max(array[time][i_bloc[0]:i_bloc[1]]) for time in range(frame_shape[0]))
                #cumul = np.sum(np.sum(array[time][i_bloc[0]:i_bloc[1]]) for time in range(frame_shape[0]))
                #print cumul
                time_series[i] = np.append(time_series[i], value)
                #print [len(x) for x in time_series]
            print(num)
            num = num + 1

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

def center(data):
    """
        Center the data (mean = 0)
    """
    train_data = data[0]
    test_data = data[1]
    scaler = StandardScaler(with_std = False)
    scaler.fit(train_data)

    # Both dataset are normalized (necessary for the test set?)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    return (train_data, test_data)


def standardize(data):
    """
        Standardize the dataset (mean = 0, variance = 1)
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
#    print("Variance explanation : "+str(pca.explained_variance_ratio_))
    return (train_pca, pca.fit_transform(test_data))

def evaluate(detector, test_data):
    for i in range(len(test_data) - 1):
        prediction = detector.predict(test_data[:i],test_data[i + 1])

# New train set with reduced features
#np.savetxt("mini.csv", read_files("mini-data"), delimiter=",")

#(g_train_data, g_test_data) = do_PCA(split_data(read_files("mini-data")), explained_variance)
(g_train_data, g_test_data) = do_PCA(split_data(read_file("data/1530056352292")), explained_variance)

print("Learning…")
detector.learn(g_train_data, None)
print("Predicting…")
#detector.predict(g_test_data)
