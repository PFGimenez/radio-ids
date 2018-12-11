#!/usr/bin/env python3

import numpy as np
from sklearn.decomposition import PCA
from preprocess import *
from hmm import HMM
import os.path
from sklearn.externals import joblib

config = Config()
filename = os.path.join(config.get_config("section"), config.get_config("pca_filename"))
nb_features_macro = config.get_config_eval("nb_features_macro")
temporal_duration = config.get_config_eval("macro_feature_duration")
waterfall_duration = config.get_config_eval("waterfall_duration")
waterfall_length = config.get_config_eval("waterfall_dimensions")[0]

if os.path.isfile(filename):
    print("PCA already learnt!")
else:
    with open("train_folders") as f:
        folders = f.readlines()
    folders = [x.strip() for x in folders]

    filenames = get_files_names(folders)
    train_data = read_files(filenames)

    before = train_data.shape[1]
    pca = PCA(0.95, svd_solver="full")
    trained_pca = pca.fit_transform(train_data)
    print("Features number: " + str(before) + " -> " + str(train_pca.shape[1]))
    joblib.dump(trained_pca, filename)
    print("PCA saved")

