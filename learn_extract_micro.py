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
from extractor import MultiExtractors

np.random.seed()
random.seed()


# TODO merge from different raspi
with open("train_folders") as f:
    folders = f.readlines()
folders = [x.strip() for x in folders]

config = Config()

min_value = config.get_config_eval('min_value')

bands = config.get_config_eval('waterfall_frequency_bands')
extractors = MultiExtractors()
dims = config.get_config_eval('autoenc_dimensions')

new = False

for j in range(len(bands)):
    (i,s) = bands[j]
    try:
        m = CNN(i, s, dims[j])
        extractors.load(i, s, m)

    except Exception as e:
        print("Loading failed:",e)
        print("Learning extractor from files in",folders)
        filenames = get_files_names(folders)
        m.learn_extractor(filenames, i, s)
        extractors.add_model(m, i, s)
        new = True

if new:
    print("Saving extractors…")
    extractors.save()

exit()

fig = plt.figure()

autoenc_shape = config.get_config_eval('autoenc_dimensions')
data = crop_all(read_directory("data-test2"), autoenc_shape[0], autoenc_shape[1])
data_reconstructed = autoenc.reconstruct(data)
print(data[0,:,:].shape)
plt.imshow(np.concatenate((data[0,:,:], min_value*np.ones((int(autoenc_shape[0]/3), autoenc_shape[1])), data_reconstructed[0,:,:])), cmap='hot', interpolation='nearest', aspect='auto')
plt.show()
