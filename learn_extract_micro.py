#!/usr/bin/env python3

import numpy as np
from autoencodercnn import CNN
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from models import MultiExtractors
from preprocess import *
from config import Config
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

np.random.seed()
random.seed()


with open("train_folders") as f:
    folders = f.readlines()
folders = [x.strip() for x in folders]
folders = ['data-test']
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
        extractors.load_model(m)

    except Exception as e:
        print("Loading failed:",e)
        print("Learning extractor from files in",folders)
        filenames = get_files_names(folders)
        m.learn_extractor(filenames, i, s)
        extractors.add_model(m)
        new = True
new = True # TODO virer
if new:
    extractors.save_all()
    print("Learning threshold")
    fnames = [[os.path.join(directory,f) for f in sorted(os.listdir(directory))] for directory in folders]
    extractors.learn_thresholds(fnames)
#    rmse = extractors.rmse_from_folders(fnames)
#    print("99% percentile threshold:",np.percentile(rmse, 99))
#    print("max threshold:",np.max(rmse))
    print("Saving extractors…")
    extractors.save_all()
else:
    print("Extractors already learnt !")

fig = plt.figure()

data = read_directory("data-test2")[0:5,:,:]
data = np.concatenate(data)

# juste pour la heatmap…
#data[20,0] = min_value
#data[21,1] = max_value

#print(np.mean(rmse))
#print(np.min(rmse))
#print(np.max(rmse))
#print(np.percentile(rmse, 1))
#print(np.percentile(rmse, 99))

quantify(data)
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
