import os
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from varmax import Varmax
from armax import Armax
from autoencodercnn import CNN
from hmm import HMM
from anomalydetector import AnomalyDetector

#detector = Armax((20, 10), manhattan_distances, 10)
#detector = Varmax((3, 0))
#detector = HMM(5, 0.1)

explained_variance = 0.95

# unused
#i_blocs = [(0, 500), (500, 1000), (1000, 1250), (1250, 1500), (1000, 1500)]
#frame_shape = (50, 1500)
#directory = "data"
# def read_files_format(directory, i_blocs, frame_shape):
#     num = 0
#     files = sorted(os.listdir(path))
#     for f in files:
#         f = path + "/" + f
#         if os.path.isfile(f):
#             array = np.fromfile(f, dtype='float64')
#             array = np.reshape(array, frame_shape)
#             for i in range(len(i_blocs)):
#                 i_bloc = i_blocs[i]
#                 array[0,0] = array[0,2]
#                 array[0,1] = array[0,2]

#                 value = np.max(array[:, i_bloc[0]:i_bloc[1]])
#                 value = (value + 80) * 255 / 80.0
#                 time_series[i] = np.append(time_series[i], value)
#             print(num)
#             num = num + 1

def read_file(filename):
    """
        Read one file
    """
    print("Reading data file " + filename)
    data = np.fromfile(filename, dtype=np.dtype('float64'))
    print(data.reshape(-1,1500)[:,1])
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

    scaler.fit(train_data)

    # Both dataset are normalized (necessary for the test set?)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    return (train_data, test_data)

def detect_signal(data, minimal_size_f, minimal_size_t, maximal_pause_size_f, maximal_pause_size_t, rising_threshold, falling_threshold, size_before_f, size_before_t, size_after_f, size_after_t):
    """
        Isole des rectangles 2D de signal
    """
    rectangles = []
    for i in range(2):
        lines = data[i]
        for j in lines.shape[0]:
            intervals_t.append(detect_signal_1D(lines[j,:], minimal_size_t, maximal_pause_size_t, rising_threshold, falling_threshold, size_before_t, size_after_t))
        for j in lines.shape[1]:
            intervals_f.append(detect_signal_1D(lines[:,j], minimal_size_f, maximal_pause_size_f, rising_threshold, falling_threshold, size_before_f, size_after_f))

def detect_signal_1D(data, minimal_size, maximal_pause_size, rising_threshold, falling_threshold, size_before, size_after):
    """
        Isole une plage 1D (temps ou fréquence) de signal
    """
    out = []
    start = None
    last_powerful = None
    for i in range(len(data)):
        last_one = (i == len(data) - 1)
        if data[i] > rising_threshold and start == None:
            # Nouveau début de frame ?
            start = i
            last_powerful = i
        elif start != None and data[i] > falling_threshold:
            # Dès qu'on repasse au-dessus du seuil de fin
            last_powerful = i
        if start != None and (data[i] <= falling_threshold or last_one):
            # Ce n'est plus puissant
            if i - last_powerful > maximal_pause_size or last_one:
                # Pause trop longue : frame terminée
                if last_powerful - start + 1 >= minimal_size:
                    first = max(start - size_before, 0)
                    last = min(len(data) - 1, last_powerful + size_after)
                    out.append((first, last))
                start = None
                last_powerful = None
    return out

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

def evaluate(detector, test_data, distance):
    for i in range(len(test_data) - 1):
        prediction = detector.predict(test_data[:i], test_data[i + 1], distance)

# New train set with reduced features
#np.savetxt("mini-pca.csv", do_PCA(split_data(read_files("mini-data")), explained_variance)[0], delimiter=",")



#(g_train_data, g_test_data) = do_PCA(split_data(read_files("data-test")), explained_variance)

# NO PCA
(g_train_data, g_test_data) = split_data(read_files("data-test"))
#(g_train_data, g_test_data) = do_PCA(split_data(read_file("data/1530056352292")), explained_variance)

#test = [10, 10, 1, 10, 10, 10, 10, 1, 1, 1, 1, 10, 1, 1, 1, 10, 10, 10, 10, 10, 1, 1, 1, 1, 1, 1, 10, 10]
#out = detect_signal_1D(test, 2, 2, 5, 5, 1, 1)

#detector = CNN(g_train_data.shape)

# TODO pour CNN : réduire la taille des données (retirer les features inutiles ou découper par canaux)

print("Learning…")
#detector.learn(g_train_data[:,1].reshape(-1,1))
print("Saving…")
#pickle.dump(detector, open('armax','wb'))
print("Loading…")
#detector = pickle.load(open('armax','rb'))
print("Predicting…")
#detector.predict(g_test_data)
