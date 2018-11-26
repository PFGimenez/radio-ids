import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import math
from keras import backend as K

def normalize(data, val_min, val_max):
    return (data - val_min) / (val_max - val_min)

def denormalize(data, val_min, val_max):
    return data * (val_max - val_min) + val_min

def crop_all(data, size_x, size_y):
    return np.array([crop_sample(s, size_x, size_y) for s in data])

def crop_sample(data, size_x, size_y):
    shape_x = data.shape[0]
    shape_y = data.shape[1]
    return data[int((shape_x - size_x) / 2) : int((shape_x + size_x) / 2),
                int((shape_y - size_y) / 2) : int((shape_y + size_y) / 2)]

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
    intervals_t.append(detect_signal_1D(lines[j,:], minimal_size_t, maximal_pause_size_t, rising_threshold, falling_threshold, size_before_t, size_after_t))
    intervals_f.append(detect_signal_1D(lines[:,j], minimal_size_f, maximal_pause_size_f, rising_threshold, falling_threshold, size_before_f, size_after_f))
    changed = True
    while changed:
        pass # TODO ?

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

def decompose_raw(data, shape, overlap=0):
    shape_x = shape[0]
    shape_y = shape[1]
    step_x = round(shape_x * (1 - overlap))
    step_y = round(shape_y * (1 - overlap))
    x = math.floor((data.shape[0] - shape_x) / step_x) + 1
    y = math.floor((data.shape[1] - shape_y) / step_y) + 1
    out = np.empty((x, y, shape_x, shape_y))
    for i in range(x):
        for j in range(y):
            out[i,j] = data[i * step_x : i * step_x + shape_x, j * step_y : j * step_y + shape_y]
#    print(out.shape)
    return out

def decompose(data, shape, overlap=0):
    out = decompose_raw(data, shape, overlap)
    shape_x = shape[0]
    shape_y = shape[1]
    step_x = round(shape_x * (1 - overlap))
    step_y = round(shape_y * (1 - overlap))
    x = math.floor((data.shape[0] - shape_x) / step_x) + 1
    y = math.floor((data.shape[1] - shape_y) / step_y) + 1

    if K.image_data_format() == 'channels_first':
        return out.reshape(x*y, 1, shape_x, shape_y)
    else:
        return out.reshape(x*y, shape_x, shape_y, 1)

def read_file(filename):
    """
        Read one file
    """
#    print("Reading data file " + filename)
    data = np.fromfile(filename, dtype=np.dtype('float64'))
#    print(data.reshape(-1,1500)[:,1])
    return data.reshape(-1, 1500)

def get_files_names(directory_list, pattern=""):
    """
        Renvoie la liste des fichiers, triée
    """
    names = [os.path.join(d, s) for d in directory_list for s in sorted(os.listdir(d)) if pattern in s]
    print(names)
    return names

def read_files(files_list):
    data = []
    i = 1
    for fname in files_list:
        if i % 100 == 0:
            print(i,"/",len(files_list))
        i += 1
        data.append(np.fromfile(fname, dtype=np.dtype('float64')).reshape(-1,1500))
#    print(str(len(data)) + " files read")

#    print("Files read" + str(data.shape))
#data = data.reshape(50,-1)
    return np.array(data)

def read_directories(directories):
    """
        Read all files from several directories
    """
    # TODO not tested
    out = []
    for d in directories:
        out.append(read_directory(d))
    return np.concatenate(out)

def read_directory(directory):
    """
        Read all files from a directory
    """

    print("Reading data files from directory " + directory)
    files_list = [os.path.join(directory, fname) for fname in sorted(os.listdir(directory))]
    return read_files(files_list)

def split_data(data):
    """
        Split into train set and test set
    """
#print("Creating train and test set")
    [train_data, test_data] = train_test_split(data)
    print("Train set: " + str(train_data.shape))
    print("Test set: " + str(test_data.shape))
    return (train_data, test_data)

def get_event_list(data, temporal_step, spectral_step):
    data = np.concatenate(data)
    data = decompose_raw(data, (temporal_step, spectral_step))
    data = data.reshape((data.shape[0], data.shape[1], -1))
#    print(data.shape)
#    data = np.concatenate((np.amax(data, axis=2),
#                          np.mean(data, axis=2)
#                          ), axis=1)
    data = np.amax(data, axis=2)
#    print(data.shape)
    return data

def test_prediction(data, model):
    predictions = []
    for i in range(1,len(data)):
        predictions.append(model.predict_list(data[:i,:]))
    return predictions

