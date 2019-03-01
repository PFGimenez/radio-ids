#!/usr/bin/env python3

import numpy as np
from autoencodercnn import CNN
from models import MultiExtractors
from preprocess import *
from config import Config
import random

with open("train_folders") as f: # TODO
    folders = f.readlines()
folders = [x.strip() for x in folders]
#folders = ['data-test', 'data-test']
#folders = [folders[0]]
config = Config()

bands = config.get_config_eval('waterfall_frequency_bands')
extractors = MultiExtractors()
dims = config.get_config_eval('autoenc_dimensions')
epochs = config.get_config_eval('nb_epochs')

new = False

for j in range(len(bands)):
    (i,s) = bands[j]
    try:
        m = CNN(j)
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
#    print("Learning threshold")
#    fnames = [[os.path.join(directory,f) for f in sorted(os.listdir(directory))] for directory in folders]
#    extractors.learn_threshold(fnames)
#    print("Saving extractors…")
#    extractors.save_all()
else:
    print("Extractors already learnt !")

