#!/usr/bin/env python3

from preprocess import *
from autoencodercnn import CNN
import sys
import os
from config import Config
from extractor import MultiExtractors
from extract_micro import extract_micro

config = Config()
prefix = config.get_config("section")
#config.set_config("autoenc_filename", "test-6.h5") # TODO
nb_features = config.get_config_eval("nb_features")

bands = config.get_config_eval('waterfall_frequency_bands')
extractors = MultiExtractors()
dims = config.get_config_eval('autoenc_dimensions')

days = os.listdir("raspi-merged")
days = [os.path.join("raspi-merged",d) for d in days]
print(days)
exit()

for j in range(len(bands)):
    (i,s) = bands[j]
    m = CNN(j)
    extractors.load(i, s, m)

with open("train_folders") as f:
    folders = f.readlines()
directories = [x.strip() for x in folders]

extract_micro(extractors, directories, config.get_config_eval("window_overlap_training"), prefix, nb_features)

with open("test_folders") as f:
    folders = f.readlines()
directories = [x.strip() for x in folders]

extract_micro(extractors, directories, config.get_config_eval("window_overlap_testing"), prefix, nb_features)


