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
max_value = config.get_config_eval('max_value')

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
    print("Learning threshold")
    with open("train_folders") as f:
        folders = f.readlines()
    folders = [x.strip() for x in folders]
    fnames = [os.path.join(directory,f) for directory in folders for f in sorted(os.listdir(directory))]
    rmse = extractors.rmse_from_folders(fnames)
    print("99% percentile threshold:",np.percentile(rmse, 99))
    print("max threshold:",np.max(rmse))
    print("Saving extractors…")
    extractors.save()
else:
    print("Extractors already learnt !")

fig = plt.figure()

data = read_directory("data-test2")[0:5,:,:]
data = np.concatenate(data)

# juste pour la heatmap…
data[20,0] = min_value
data[21,1] = max_value

#print(np.mean(rmse))
#print(np.min(rmse))
#print(np.max(rmse))
#print(np.percentile(rmse, 1))
#print(np.percentile(rmse, 99))

data_reconstructed = extractors.reconstruct(data)[0,:,:]
data = data[:16,:1488]
data[data < min_value] = min_value
data[data > max_value] = max_value
data_reconstructed[data_reconstructed > max_value] = max_value
print(data_reconstructed.shape)
#plt.imshow(np.concatenate((denormalize(normalize(data, min_value, max_value),min_value,max_value), min_value*np.ones((int(data_reconstructed.shape[0]/3), data_reconstructed.shape[1])), data)), cmap='hot', interpolation='nearest', aspect='auto')
plt.imshow(np.concatenate((denormalize(normalize(data, min_value, max_value),min_value,max_value), min_value*np.ones((int(data_reconstructed.shape[0]/3), data_reconstructed.shape[1])), data_reconstructed)), cmap='hot', interpolation='nearest', aspect='auto')
#plt.savefig(config.get_config("autoenc_filename")+".png")
plt.show()
