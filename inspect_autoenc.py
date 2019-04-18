#!/usr/bin/env python3

import numpy as np
from autoencodercnn import CNN
from models import MultiExtractors
from preprocess import *
from config import Config
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import datetime
import sys
import time

name_attack = None
i = 1
folders = None
mode = None
save = False
while i < len(sys.argv):
    if sys.argv[i] == "-d":
        i += 1
        date_min = int(time.mktime(datetime.datetime.strptime(sys.argv[i],
            "%d/%m/%Y %H:%M:%S").timetuple()))*1000
    elif sys.argv[i] == "-f":
        i += 1
        date_max = int(time.mktime(datetime.datetime.strptime(sys.argv[i],
            "%d/%m/%Y %H:%M:%S").timetuple()))*1000
    elif sys.argv[i] == "-a":
        i += 1
        name_attack = sys.argv[i]
    elif sys.argv[i] == "-data":
        mode = "data"
    elif sys.argv[i] == "-quant":
        mode = "quant"
    elif sys.argv[i] == "-diff":
        mode = "diff"
    elif sys.argv[i] == "-autoenc":
        mode = "autoenc"
    elif sys.argv[i] == "-save":
        save = True
    elif sys.argv[i] == "-dir":
        i += 1
        if not folders:
            folders = []
        folders.append(sys.argv[i])
    else:
        print("Erreur:",sys.argv[i])
        exit()
    i += 1

if mode == None:
    print("Aucun mode ! Utilisez -data, -quant ou -autoenc")
    exit()

np.random.seed()
random.seed()

if not folders:
    with open("train_folders") as f:
        folders1 = f.readlines()
    folders = [x.strip() for x in folders1]

    with open("test_folders") as f:
        folders2 = f.readlines()
    folders += [x.strip() for x in folders2]
config = Config()


bands = config.get_config_eval('waterfall_frequency_bands')
waterfall_dimensions = config.get_config_eval('waterfall_dimensions')
extractors = MultiExtractors()
dims = config.get_config_eval('autoenc_dimensions')
epochs = config.get_config_eval('nb_epochs')

new = False
load_autoenc = (mode == "autoenc" or mode == "diff")
attack_nb_min = 0
attack_nb_max = 1

if load_autoenc:
    for j in range(len(bands)):
        (i,s) = bands[j]
        m = CNN(j)
        extractors.load_model(m)


# affiche nom fichier + date
if False:
    print(folders[0])
    s = os.listdir(folders[0]).sort(key=lambda x : int(x))
    for fname in s:
        print(fname, datetime.datetime.fromtimestamp(int(fname)/1000))
print(folders)

if name_attack:
    all_attack = np.loadtxt(os.path.join(config.get_config("section"), "logattack"), dtype='<U13')
    print(all_attack.shape)
    # all_attack = np.array([a for a in all_attack if datetime.datetime.fromtimestamp(int(a[1])/1000).day == 16])
    print(all_attack.shape)

    print(all_attack, name_attack)
    some_attack = all_attack[all_attack[:,0] == name_attack][:,1:].astype(np.integer)[attack_nb_min:attack_nb_max]
    print(some_attack)
    date_min = some_attack[0][0]
    date_max = some_attack[len(some_attack)-1][1]
    print(date_min, date_max)

if mode == "data":
    fig = plt.figure()
if mode == "quant" or mode == "data":
    data = np.vstack(read_files_from_timestamp(date_min, date_max, folders, quant=False))
if mode == "quant":
    fig, ax = plt.subplots(nrows=2, ncols=1)
    data_d = np.vstack(read_files_from_timestamp(date_min, date_max, folders, quant=True))
    dequantify(data_d)
if mode == "autoenc":
    fig, ax = plt.subplots(nrows=3, ncols=1)
if mode == "autoenc" or mode == "diff":
    data_q = np.vstack(read_files_from_timestamp(date_min, date_max, folders, quant=True))

#data = read_directory(folders[0], quant=True)[0,:,:]
#data = read_files_from_timestamp(int(some_attack[1]), int(some_attack[2]), folders, quant=True)
#data = np.vstack(data)
#data = data[10,:,:]
#data = np.zeros((50,1500))
if load_autoenc:
    data_reconstructed = extractors.reconstruct(data_q)
    data_reconstructed = np.vstack(data_reconstructed)
    print(data_reconstructed.shape)
    if save:
        data_reconstructed.tofile("reconstructed_data")
    diff = np.abs(data_reconstructed - data_q[:data_reconstructed.shape[0],:])
    # diff = np.abs(data_reconstructed[:,:900] - data_q[:data_reconstructed.shape[0],:900])
    # diff2 = np.abs(data_reconstructed[:,900:] - data_q[:data_reconstructed.shape[0],1000:])
    # diff = np.concatenate((diff, diff2), axis=1)
    diff[diff < 0.32] = 0

# vérification quantification
if mode == "quant":
    ax[0].imshow(data, cmap='hot', aspect='auto')
    ax[0].set_title("Original data")
    ax[1].imshow(data_d, cmap='hot', aspect='auto')
    ax[1].set_title("Quantified data")

# juste data
elif mode == "data":
    plt.imshow(data, cmap='hot', interpolation='nearest', aspect='auto')
    plt.title("Original data")

# vérification reconstruction
elif mode == "autoenc":
    ax[0].imshow(data_q, cmap='hot', interpolation='nearest', aspect='auto')
    ax[0].set_title("Original data")
    ax[1].imshow(data_reconstructed, cmap='hot', interpolation='nearest', aspect='auto')
    ax[1].set_title("Reconstructed data")
    ax[2].imshow(diff, cmap='hot', interpolation='nearest', aspect='auto')
    ax[2].set_title("Difference")

elif mode == "diff":
    plt.imshow(diff, cmap='hot', interpolation='nearest', aspect='auto')
    plt.title("Difference")

#plt.savefig(config.get_config("autoenc_filename")+".png")
plt.show()
