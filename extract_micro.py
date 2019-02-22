#!/usr/bin/env python3

from preprocess import *
from autoencodercnn import CNN
import sys
import os
from config import Config
from models import MultiExtractors

def extract_micro(extractors, directories, window_overlap, prefix, nb_features):
    for d in directories:
        d2 = d.split("/")[-1]
        path = os.path.join(prefix,"features-"+d2)
        if not os.path.isfile(path):
            print("Extracting features from",d)
            filenames = get_files_names([d])

            out = []
            for i in range(len(filenames)-1):
                out.append(
                    extractors.extract_features(
                        np.concatenate(read_files([filenames[i], filenames[i+1]], quant=True)),
                        int(os.path.split(filenames[i])[1]),
                        window_overlap))

            out = np.array(out).reshape(-1, nb_features+1)
            # take only the last part of the directory
            out.tofile(path)
        else:
            print("Features already extracted: ",path)

if __name__ == "__main__":
    config = Config()
    prefix = config.get_config("section")
#config.set_config("autoenc_filename", "test-6.h5") # TODO
    nb_features = config.get_config_eval("nb_features")

    bands = config.get_config_eval('waterfall_frequency_bands')
    extractors = MultiExtractors()
    dims = config.get_config_eval('autoenc_dimensions')

    for j in range(len(bands)):
        (i,s) = bands[j]
        m = CNN(j)
        extractors.load_model(m)

    with open("train_folders") as f:
        folders = f.readlines()
    directories = [x.strip() for x in folders]

    extract_micro(extractors, directories, config.get_config_eval("window_overlap_training"), prefix, nb_features)

    with open("test_folders") as f:
        folders = f.readlines()
    directories = [x.strip() for x in folders]

    extract_micro(extractors, directories, config.get_config_eval("window_overlap_testing"), prefix, nb_features)


