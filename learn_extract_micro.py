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
epochs = config.get_config_eval('nb_epochs')

new = False

for j in range(len(bands)):
    (i,s) = bands[j]
    try:
        m = CNN(i, s, dims[j], epochs[j])
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
else:
    print("Extractors already learnt !")

fig = plt.figure()

data = read_directory("data-test2")[0:5,:,:]
data = np.concatenate(data)

data[0,0] = -150
data[0,1] = 0

with open("test_folders") as f:
    folders = f.readlines()
folders = [x.strip() for x in folders]

rmse = extractors.rmse_from_folders(folders)
print(np.percentile(rmse, 99))

data_reconstructed = extractors.reconstruct(data)[0,:,:]
data = data[:16,:1488]
print(data_reconstructed.shape)
plt.imshow(np.concatenate((data, min_value*np.ones((int(data_reconstructed.shape[0]/3), data_reconstructed.shape[1])), data_reconstructed)), cmap='hot', interpolation='nearest', aspect='auto')
#plt.savefig(config.get_config("autoenc_filename")+".png")
plt.show()
