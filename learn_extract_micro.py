#!/usr/bin/env python3

import numpy as np
from varma import Varma
from arma import Arma
from autoencodercnn import CNN
from hmm import HMM
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from anomalydetector import AnomalyDetector
from preprocess import *
from config import Config
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

np.random.seed()
random.seed()


# TODO merge from different raspi
with open("train_folders") as f:
    folders = f.readlines()
folders = [x.strip() for x in folders]

config = Config()

autoenc_shape = config.get_config_eval('autoenc_dimensions')
min_value = config.get_config_eval('min_value')

autoenc = CNN()
try:
    print("Loading autoencoder…")
    #autoenc.load("test.h5")
    autoenc.load()
except Exception as e:
    print("Loading failed:",e)
    print("Learning autoencoder…")
    print("Learning from files in",folders)
    filenames = get_files_names(folders)
    autoenc.new_model()
    autoenc.learn_autoencoder(filenames, 32)
    print("Saving autoencoder…")
    autoenc.save()

fig = plt.figure()

data = crop_all(read_directory("data-test2"), autoenc_shape[0], autoenc_shape[1])
data_reconstructed = autoenc.reconstruct(data)
print(data[0,:,:].shape)
plt.imshow(np.concatenate((data[0,:,:], min_value*np.ones((int(autoenc_shape[0]/3), autoenc_shape[1])), data_reconstructed[0,:,:])), cmap='hot', interpolation='nearest', aspect='auto')
plt.show()
