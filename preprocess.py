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
_waterfall_duration = _config.get_config_eval("waterfall_duration")
_delta_t = int(_waterfall_duration / _waterfall_dim[0])

# _l_high = [-65,-50,-35,-20,-10,0]
# _l_low= [-65,-50,-35,-20,-10,0]
_l_high = [-50,-45,-40,-35,-30,-25,-20,-15,-10,0]
_l_low = [-50,-45,-40,-35,-30,-25,-20,-15,-10,0]

def show_histo(data, log=False, flatten=False):
    """
        Affiche un histogramme de data
    """
    plt.hist(data.flatten() if flatten else data, log=log, bins=100)
    plt.title("Histogram")
    plt.show()

def dequantify(data):
    # print("Dequantification…")
    valmax = len(_l_high)-1
    data[data == 0] = -80
    dl = data[:,0:2000]
    dh = data[:,2000:3000]
    for i in range(1,len(_l_high)):
        dh[dh == i/valmax] = (_l_high[i-1] + _l_high[i]) / 2

    valmax = len(_l_low)-1
    for i in range(1,len(_l_low)):
        dl[dl == i/valmax] = (_l_low[i-1] + _l_low[i]) / 2

    # print("Dequantification done")

def quantify(data):
    # print("Quantification…")
    valmax = len(_l_high)-1
    assert np.all(data < 0) # pour être sûr qu'on ne l'utilise pas deux fois de suite
    data[data >= 0] = -1 # qui devient ensuite 12
    dl = data[:,0:2000]
    dh = data[:,2000:3000]

    for i in range(len(_l_high)):
        dh[dh < _l_high[i]] = i/valmax

    valmax = len(_l_low)-1
    for i in range(len(_l_low)):
        dl[dl < _l_low[i]] = i/valmax
    # print("Quantification done")

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


def decompose_raw(data, shape, overlap=0,pad_t=False, pad_f=False):
    shape_x = shape[0]
    shape_y = shape[1]
    # spectral overlap is always 0
    step_x = round(shape_x * (1 - overlap))
    step_y = shape_y
    if pad_t:
        x = math.ceil((data.shape[0] - shape_x) / step_x) + 1
    else:
        x = math.floor((data.shape[0] - shape_x) / step_x) + 1
    if pad_f:
        y = math.ceil((data.shape[1] - shape_y) / step_y) + 1
    else:
        y = math.floor((data.shape[1] - shape_y) / step_y) + 1
    out = np.zeros((x, y, shape_x, shape_y))
    for i in range(x):
        for j in range(y):
            d = data[i * step_x : min(i * step_x + shape_x, data.shape[0]), j * step_y : min(j * step_y + shape_y, data.shape[1])]
            out[i,j,:d.shape[0],:d.shape[1]] = d
    return (x,y,out)

def decompose(data, shape, overlap=0,pad_t=False, pad_f=True):
    (x,y,out) = decompose_raw(data, shape, overlap,pad_t=pad_t, pad_f=pad_f)
    shape_x = shape[0]
    shape_y = shape[1]
    # spectral overlap is always 0
#    step_x = round(shape_x * (1 - overlap))
#    step_y = shape_y
#    x = math.floor((data.shape[0] - shape_x) / step_x) + 1
#    y = math.floor((data.shape[1] - shape_y) / step_y) + 1

    if K.image_data_format() == 'channels_first':
        return out.reshape(x*y, 1, shape_x, shape_y)
    else:
        return out.reshape(x*y, shape_x, shape_y, 1)

def macro_decompose(data, merge_shape, input_shape, overlap=0,pad_t=False, pad_f=True):
    """
    decompose for macro detector
    """
    # print(data.shape, merge_shape, input_shape, overlap)
    (x,y,out) = decompose_raw(data, merge_shape, overlap,pad_t=pad_t, pad_f=pad_f)
    # print(x,y,out.shape)
    out = np.mean(out, axis=(2,3))
    # print(x,y)
    # print(out.shape)
    number_split = math.floor(x / input_shape[0])
    # print(number_split, number_split * input_shape[0])
    out = out[:number_split * input_shape[0],:]
    # print(out.shape)
    out = np.array(np.vsplit(out, number_split))
    # print(out.shape)

    if K.image_data_format() == 'channels_first':
        return out.reshape(out.shape[0], 1, out.shape[1], out.shape[2])
    else:
        return out.reshape(out.shape[0], out.shape[1], out.shape[2], 1)



def read_file(filename, quant=False, int_type=True):
    """
        Read one file
    """
#    print("Reading data file " + filename)
    data = np.fromfile(filename, dtype=np.dtype('int8' if int_type else 'float64')).astype('float64').reshape(_waterfall_dim)

    if quant:
        quantify(data)
    return data

def get_files_names(directory_list, pattern=""):
    """
        Renvoie la liste des fichiers, triée
    """
    names = [os.path.join(d, s) for d in directory_list for s in sorted(os.listdir(d)) if pattern in s]
    return names

def read_files_from_timestamp(date_min, date_max, directory_list, quant=True, int_type=True):
    names = [os.path.join(d, s) for d in directory_list for s in sorted(os.listdir(d)) if int(s) > date_min - _waterfall_duration and int(s) < date_max]
    if names == []:
        return None
    first_date = int([s for d in directory_list for s in sorted(os.listdir(d)) if int(s) > date_min - _waterfall_duration and int(s) < date_max + 1000][0]) # un peu de rab au cas où (_waterfall_duration n'est qu'une moyenne)
    index_start = int((date_min - first_date) / _delta_t)
    length = int((date_max - date_min) / _delta_t)
    # print(first_date,index_start,length)
    out = np.vstack(read_files(names, quant=quant, int_type=int_type))[index_start:index_start+length,:]
    # print(out.shape)
    return out

def read_files(files_list, quant=False, int_type=True):
    data = []
    i = 1
    if len(files_list) >= 100:
        print(len(files_list))
    for fname in files_list:
        if i % 100 == 0:
            print(i,"/",len(files_list))
        i += 1

        d = np.fromfile(fname, dtype=np.dtype('int8' if int_type else 'float64')).reshape(_waterfall_dim).astype('float64')
        # print(fname)
        if quant:
            quantify(d)
        data.append(d)
    data = np.array(data)
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


"""
calculate a weighted median
@author Jack Peterson (jack@tinybike.net)
source : https://gist.github.com/tinybike/d9ff1dad515b66cc0d87
"""

def weighted_median(data, weights):
    """
    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights
    """
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(s_weights)
    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx:idx+2])
        else:
            w_median = s_data[idx+1]
    return w_median

def index_to_frequency(index):
    assert index >= 0 and index < 3000, index
    if index < 1000:
        return 400 + index / 10
    if index < 2000:
        return 800 + (index - 1000) / 10
    return 2400 + (index - 2000) / 10

