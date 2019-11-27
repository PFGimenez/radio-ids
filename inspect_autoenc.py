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
band = None
title = ""
while i < len(sys.argv):
    if sys.argv[i] == "-d":
        i += 1
        date_min = int(time.mktime(datetime.datetime.strptime(sys.argv[i],
            "%d/%m/%Y %H:%M:%S").timetuple()))*1000
        print(date_min)
    elif sys.argv[i] == "-f":
        i += 1
        date_max = int(time.mktime(datetime.datetime.strptime(sys.argv[i],
            "%d/%m/%Y %H:%M:%S").timetuple()))*1000
        print(date_max)
    elif sys.argv[i] == "-title":
        i += 1
        title = sys.argv[i]
    elif sys.argv[i] == "-D":
        i += 1
        date_min = int(sys.argv[i])
        print(datetime.datetime.fromtimestamp(date_min/1000))
    elif sys.argv[i] == "-F":
        i += 1
        date_max = int(sys.argv[i])
        print(datetime.datetime.fromtimestamp(date_max/1000))
    elif sys.argv[i] == "-a":
        i += 1
        name_attack = sys.argv[i]
    elif sys.argv[i] == "-data":
        mode = "data"
    elif sys.argv[i] == "-quant":
        mode = "quant"
    elif sys.argv[i] == "-article":
        mode = "article"
        i += 1
        band = int(sys.argv[i])
    elif sys.argv[i] == "-article-data":
        mode = "article-data"
    elif sys.argv[i] == "-article-preprocess":
        mode = "article-preprocess"
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

config = Config()
if not folders:
    with open(config.get_config("train_folders")) as f:
        folders1 = f.readlines()
    folders = [x.strip() for x in folders1]

    with open(config.get_config("test_folders")) as f:
        folders2 = f.readlines()
    folders += [x.strip() for x in folders2]


bands = config.get_config_eval('waterfall_frequency_bands')
waterfall_dimensions = config.get_config_eval('waterfall_dimensions')
extractors = MultiExtractors()
dims = config.get_config_eval('autoenc_dimensions')
epochs = config.get_config_eval('nb_epochs')

new = False
load_autoenc = (mode == "autoenc" or mode == "diff" or mode =="article")
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
if mode == "article-data":
    fig = plt.figure()
if mode == "quant" or mode == "data" or mode == "article-data":
    data = read_files_from_timestamp(date_min, date_max, folders, quant=False)
if mode == "quant":
    fig, ax = plt.subplots(nrows=2, ncols=1)
    data_d = read_files_from_timestamp(date_min, date_max, folders, quant=True)
    dequantify(data_d)
if mode == "autoenc":
    fig, ax = plt.subplots(nrows=3, ncols=1)
if mode == "article":
    fig, ax = plt.subplots(nrows=1, ncols=3)
if mode == "article-preprocess":
    fig, ax = plt.subplots(nrows=2, ncols=3)
if mode == "autoenc" or mode == "diff" or mode == "article":
    data_q = read_files_from_timestamp(date_min, date_max, folders, quant=True)
if mode == "article-preprocess":
    data = read_files_from_timestamp(date_min, date_max, folders, quant=False)
    data_q = read_files_from_timestamp(date_min, date_max, folders, quant=True)

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
    diff = np.abs(np.subtract(data_reconstructed, data_q[:data_reconstructed.shape[0],:]))
    # print("Score: ",extractors.get_score(data_q, 0))
    # diff = np.abs(data_reconstructed[:,:900] - data_q[:data_reconstructed.shape[0],:900])
    # diff2 = np.abs(data_reconstructed[:,900:] - data_q[:data_reconstructed.shape[0],1000:])
    # diff = np.concatenate((diff, diff2), axis=1)
    # diff[diff < 0.1] = 0
    l = range(3) if band is None else [band]
    for i in l:
        print("Band",i)
        (weights, data) = extractors.get_frequencies(data_q, i)
        median = weighted_median(data, weights) + 1000*i
        f = index_to_frequency(median)
        print("Index atk:",median)
        print("Frequency atk:",f)

# vérification quantification
if mode == "quant":
    ax[0].imshow(data, cmap='hot', aspect='auto')
    ax[0].set_title("Original data")
    ax[1].imshow(data_d, cmap='hot', aspect='auto')
    ax[1].set_title("Quantified data")

# juste data
elif mode == "data":
    # print(np.median(data[:1000,:]))
    # print(np.median(data[1000:2000,:]))
    # print(np.median(data[2000:,:]))
    plt.imshow(data, cmap='hot', interpolation='nearest', aspect='auto')
    plt.title("Original data")

elif mode == "article-data":
    data = data[:,2000:3000]
    im=plt.imshow(data.T, cmap='hot', interpolation='nearest', aspect='auto',extent=[0,data.shape[0]*0.0375,2500,2400])
    plt.xlabel("Temps (s)")
    plt.ylabel("Fréquence (MHz)")
    cbar = plt.colorbar()
    cbar.set_label("dBm")

# vérification reconstruction
elif mode == "autoenc":
    ax[0].imshow(data_q, cmap='hot', interpolation='nearest', aspect='auto')
    ax[0].set_title("Données d'origine")
    ax[1].imshow(data_reconstructed, cmap='hot', interpolation='nearest', aspect='auto')
    ax[1].set_title("Données reconstruites")
    ax[2].imshow(diff, cmap='hot', interpolation='nearest', aspect='auto')
    ax[2].set_title("Différence")

elif mode == "article-preprocess":
    vminq=0
    vmaxq=0.90
    data1q = data_q[:,0:1000]
    data2q = data_q[:,1000:2000]
    data3q = data_q[:,2000:3000]

    vmin=-105
    vmax=0
    data1 = data[:,0:1000]
    data2 = data[:,1000:2000]
    data3 = data[:,2000:3000]

    print(np.max(data1), np.min(data1))
    print(np.max(data2), np.min(data2))
    print(np.max(data3), np.min(data3))
    ax[0][0].imshow(data1.T, cmap='hot', interpolation='nearest', aspect='auto',vmin=vmin,vmax=vmax,extent=[0,data1.shape[0]*0.0375,500,400])
    ax[0][0].set_xlabel("Temps (s)")
    ax[0][0].set_ylabel("Fréquence (MHz)")

    im = ax[0][1].imshow(data2.T, cmap='hot', interpolation='nearest', aspect='auto',vmin=vmin,vmax=vmax,extent=[0,data1.shape[0]*0.0375,900,800])
    ax[0][1].set_xlabel("Temps (s)")

    ax[0][2].imshow(data3.T, cmap='hot', interpolation='nearest', aspect='auto',vmin=vmin,vmax=vmax,extent=[0,data1.shape[0]*0.0375,2500,2400])
    ax[0][2].set_xlabel("Temps (s)")

    ax[1][0].imshow(data1q.T, cmap='hot', interpolation='nearest', aspect='auto',vmin=vminq,vmax=vmaxq,extent=[0,data1.shape[0]*0.0375,500,400])
    ax[1][0].set_xlabel("Temps (s)")
    ax[1][0].set_ylabel("Fréquence (MHz)")

    imq = ax[1][1].imshow(data2q.T, cmap='hot', interpolation='nearest', aspect='auto',vmin=vminq,vmax=vmaxq,extent=[0,data1.shape[0]*0.0375,900,800])
    ax[1][1].set_xlabel("Temps (s)")

    ax[1][2].imshow(data3q.T, cmap='hot', interpolation='nearest', aspect='auto',vmin=vminq,vmax=vmaxq,extent=[0,data1.shape[0]*0.0375,2500,2400])
    ax[1][2].set_xlabel("Temps (s)")


    fig.subplots_adjust(right=0.8)
    cbar_axq = fig.add_axes([0.85, 0.12, 0.01, 0.3])

    cbar_ax = fig.add_axes([0.85, 0.55, 0.01, 0.3])
    cbar_ax.set_xlabel("dBm")
    fig.colorbar(im, cax=cbar_ax)
    fig.colorbar(imq, cax=cbar_axq)



elif mode == "article":
    vmax=0.85
    data_q = data_q[:,band*1000:(band+1)*1000]
    data_reconstructed = data_reconstructed[:,band*1000:(band+1)*1000]
    diff = diff[:,band*1000:(band+1)*1000]
    print(np.max(data_q))
    print(np.max(data_reconstructed))
    print(np.max(diff))
    if band==0:
        b0 = 400
        b1 = 500
    elif band==1:
        b0 = 800
        b1 = 900
    else:
        b0 = 2400
        b1 = 2500
    ax[0].imshow(data_q.T, cmap='hot', interpolation='nearest', aspect='auto',vmin=0,vmax=vmax,extent=[0,data_q.shape[0]*0.0375,b1,b0])
    ax[0].set_title("Données d'origine")
    ax[0].set_xlabel("Temps (s)")
    ax[0].set_ylabel("Fréquence (MHz)")

    im = ax[1].imshow(data_reconstructed.T, cmap='hot', interpolation='nearest', aspect='auto',vmin=0,vmax=vmax,extent=[0,data_q.shape[0]*0.0375,b1,b0])
    ax[1].set_title("Reconstruction")
    ax[1].set_xlabel("Temps (s)")
    # ax[1].set_ylabel("Frequency")

    ax[2].imshow(diff.T, cmap='hot', interpolation='nearest', aspect='auto',vmin=0,vmax=vmax,extent=[0,data_q.shape[0]*0.0375,b1,b0])
    ax[2].set_title("Différence "+title)
    ax[2].set_xlabel("Temps (s)")
    # ax[2].set_ylabel("Frequency")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax)

elif mode == "diff":
    plt.imshow(diff, cmap='hot', interpolation='nearest', aspect='auto')
    plt.title("Difference")

#plt.savefig(config.get_config("autoenc_filename")+".png")
plt.show()
