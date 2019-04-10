#!/usr/bin/env python3

import numpy as np
from autoencodercnn import CNN
from models import MultiExtractors
from preprocess import *
from config import Config
import random
import sys

with open("train_folders") as f: # TODO
    folders = f.readlines()
folders = [x.strip() for x in folders]
#folders = ['data-test', 'data-test']
# folders = [folders[0]] # TODO
config = Config()
bands = config.get_config_eval('macro_waterfall_frequency_bands')
extractors = MultiExtractors()
dims = config.get_config_eval('autoenc_dimensions')
epochs = config.get_config_eval('nb_epochs')

macro_merge_shape = config.get_config_eval('macro_merge_shape')
macro_input_shape = config.get_config_eval('macro_autoenc_dimensions')[0]
new = False

m = CNN(0, macro=True)
try:
    extractors.load_model(m)

except Exception as e:
    print("Loading failed:",e)
    print("Learning extractor from files in",folders)
    filenames = get_files_names(folders)
    m.learn_extractor(filenames, bands[0][0], bands[0][1], macro=True, macro_merge_shape=macro_merge_shape, macro_input_shape=macro_input_shape)
    extractors.add_model(m)
    new = True
new = True # TODO virer
if new:
    extractors.save_all("macro_")
else:
    print("Extractors already learnt !")

