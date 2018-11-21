import numpy as np
from varma import Varma
from arma import Arma
from autoencodercnn import CNN
from hmm import HMM
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from anomalydetector import AnomalyDetector
from preprocess import *
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

detector = HMM(5, 0.1)

explained_variance = 0.95

def evaluate(detector, test_data, distance):
    for i in range(len(test_data) - 1):
        prediction = detector.predict(test_data[:i], test_data[i + 1], distance)

np.random.seed()
random.seed()

# TODO merge from different raspi

directory = "mini-data"
#autoenc_learning_directory = ["/data/data/00.raw/raw/Adr_Expe_28-08_07-10/raspi1/learn_dataset_01_October"]
autoenc_learning_directory = get_files_names(["/data/data/00.raw/raw/Adr_Expe_28-08_07-10/raspi1/"], "01_October")
data = read_directory("data-test")
print(data.shape)
dataLearn = data[0:20,0:20,0:10].reshape(-1, 10)
print(dataLearn.shape)
detector.learn(dataLearn)
print(detector.predict(dataLearn, data[21,0:20,0:10].reshape(-1,10)[1]))
autoenc = CNN((50, 1500), (16,1472), 0.8, -150, 0)
try:
    print("Loading autoencoder…")
    autoenc.load("test.h5")
except Exception as e:
    print("Loading failed:",e)
    print("Learning autoencoder…")
    print("Learning from files in",autoenc_learning_directory)
    filenames = get_files_names(autoenc_learning_directory)
    autoenc.new_model()
    autoenc.learn_autoencoder(filenames, 32)
    print("Saving autoencoder…")
    autoenc.save("test.h5")

fig = plt.figure()

data = crop_all(read_directory("data-test2"), 16, 1472)
data_reconstructed = autoenc.reconstruct(data)

plt.imshow(np.concatenate((data_reconstructed[0,:,:], -150*np.ones(data[0,:,:].shape), data[0,:,:])), cmap='hot', interpolation='nearest', aspect='auto')
plt.show()
