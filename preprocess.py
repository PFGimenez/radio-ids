import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.random import sample_without_replacement
import math
from keras import backend as K
from config import Config
import matplotlib.pyplot as plt

_config = Config()
_waterfall_dim = _config.get_config_eval('waterfall_dimensions')

def show_histo(data, log=False, flatten=False):
    """
        Affiche un histogramme de data
    """
    plt.hist(data.flatten() if flatten else data, log=log, bins=100)
    plt.title("Histogram")
    plt.show()

def quantify(data):
    assert np.all(data < 0) # pour être sûr qu'on ne l'utilise pas deux fois de suite
    data[data >= 0] = -10 # qui devient ensuite 12
    data[data < -60] = 0
    data[data < -50] = 1
    data[data < -40] = 2
    data[data < -35] = 3
    data[data < -30] = 4
    data[data < -27.5] = 5
    data[data < -25] = 6
    data[data < -22.5] = 7
    data[data < -20] = 8
    data[data < -17.5] = 9
    data[data < -15] = 10
    data[data < -12.5] = 11
    data[data < 0] = 12
    data = data/12

def subsample(data, prop=0.01):
    """
        Return only a subsample
    """
    return data[sample_without_replacement(data.shape[0], int(data.shape[0]*prop))]

def normalize(data, val_min, val_max):
#    data = (data - val_min) / (val_max - val_min)
#    data[data < 0] = 0
#    data[data > 1] = 1
    return data

def denormalize(data, val_min, val_max):
#    return data * (val_max - val_min) + val_min
    return data

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
    # spectral overlap is always 0
    step_x = round(shape_x * (1 - overlap))
    step_y = shape_y
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
    # spectral overlap is always 0
    step_x = round(shape_x * (1 - overlap))
    step_y = shape_y
    x = math.floor((data.shape[0] - shape_x) / step_x) + 1
    y = math.floor((data.shape[1] - shape_y) / step_y) + 1

    if K.image_data_format() == 'channels_first':
        return out.reshape(x*y, 1, shape_x, shape_y)
    else:
        return out.reshape(x*y, shape_x, shape_y, 1)

def read_file(filename, quant=False, int_type=True):
    """
        Read one file
    """
#    print("Reading data file " + filename)
    data = np.fromfile(filename, dtype=np.dtype('int8' if int_type else 'float64'))

    if quant:
        quantify(data)
    return data.reshape(_waterfall_dim)

def get_files_names(directory_list, pattern=""):
    """
        Renvoie la liste des fichiers, triée
    """
    names = [os.path.join(d, s) for d in directory_list for s in sorted(os.listdir(d)) if pattern in s]
    return names

def read_files_from_timestamp(date_min, date_max, directory_list, quant=False, int_type=True):
    names = [os.path.join(d, s) for d in directory_list for s in sorted(os.listdir(d)) if int(s) > date_min and int(s) < date_max]
    return read_files(names, quant=quant, int_type=int_type)

def read_files(files_list, quant=False, int_type=True):
    data = []
    i = 1
    for fname in files_list:
        if i % 100 == 0:
            print(i,"/",len(files_list))
        i += 1
        data.append(np.fromfile(fname, dtype=np.dtype('int8' if int_type else 'float64')).reshape(_waterfall_dim))
    data = np.array(data)
    if quant:
        quantify(data)
#    print(str(len(data)) + " files read")

#    print("Files read" + str(data.shape))
#data = data.reshape(50,-1)
    return data

def read_directory_with_timestamps(directory,quant=False, int_type=True):
    """
        Read all files from a directory into a dictonary with timestamp
    """
    out_data = []
    out_time = []
    s = sorted(os.listdir(directory))
    for fname in s:
        fname = os.path.join(directory, fname)
        out_data.append(read_file(fname,quant=quant, int_type=int_type))
        out_time.append(int(os.path.split(fname)[1]))
    return (out_data, out_time)

def read_directory(directory,quant=False, int_type=True):
    """
        Read all files from a directory
    """

    print("Reading data files from directory " + directory)
    files_list = [os.path.join(directory, fname) for fname in sorted(os.listdir(directory))]
    return read_files(files_list,quant=quant,int_type=int_type)

def split_data(data):
    """
        Split into train set and test set
    """
#print("Creating train and test set")
    [train_data, test_data] = train_test_split(data)
    print("Train set: " + str(train_data.shape))
    print("Test set: " + str(test_data.shape))
    return (train_data, test_data)

def test_prediction(data, model):
    """
        Liste des prédictions pour data selon model
    """
    predictions = []
    for i in range(1,len(data)):
        predictions.append(model.predict(data[:i,:]))
    return predictions

