import numpy as np
import pickle
from varmax import Varmax
from armax import Armax
from autoencodercnn import CNN
from hmm import HMM
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from anomalydetector import AnomalyDetector
from preprocess import *

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


def evaluate(detector, test_data, distance):
    for i in range(len(test_data) - 1):
        prediction = detector.predict(test_data[:i], test_data[i + 1], distance)

directory = "mini-data"
directory = ["/data/data/00.raw/raw/Adr_Expe_28-08_07-10/raspi1/learn_dataset_01_October"]

autoenc = CNN((48,1488), 0.8)

try:
    print("Loading autoencoder…")
    autoenc.load("test.h5")
except Exception as e:
    print("Loading failed:",e)
    print("Learning autoencoder…")
    print("Learning from files in",directory)
    filenames = get_files_names(directory)
    autoenc.learn_autoencoder(filenames, 32)
    print("Saving autoencoder…")
    autoenc.save("test.h5")
#    (g_train_data, g_test_data) = autoenc.extract_features(g_train_data, g_test_data)

#try:
#    print("Loading detector…")
#    detector.load("detector")
#except Exception as e:
#    print("Loading failed:",e)
#    print("Learning detector…")
#    autoenc.learn(g_train_data)
#    print("Saving detector…")
#    autoenc.save("detector")

#pickle.dump(detector, open('armax','wb'))
#detector = pickle.load(open('armax','rb'))
print("Predicting…")
#detector.predict(g_test_data)
