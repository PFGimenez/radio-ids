#!/usr/bin/env python3

import numpy as np
from preprocess import *
from hmm import HMM
import os.path
import joblib

def get_event_list(data, temporal_step, spectral_step):
    print(temporal_step, spectral_step)
    data = np.concatenate(data)
    data = decompose_raw(data, (temporal_step, spectral_step))
    data = data.reshape((data.shape[0], data.shape[1], -1))
#    print(data.shape)
#    data = np.concatenate((np.amax(data, axis=2),
#                          np.mean(data, axis=2)
#                          ), axis=1)
    data = np.amax(data, axis=2)
#    print(data.shape)
    return data

config = Config()
filename = os.path.join(config.get_config("section"), config.get_config("pca_filename"))
nb_features_macro = config.get_config_eval("nb_features_macro")
temporal_duration = config.get_config_eval("macro_feature_duration")
waterfall_duration = config.get_config_eval("waterfall_duration")
waterfall_length = config.get_config_eval("waterfall_dimensions")[0]

if os.path.isfile(filename):
    print("PCA already learnt!")
else:
    pca = PCA(0.95, svd_solver="full")
    trained_pca = pca.fit_transform(train_data)
    joblib.dump(pca, filename)
    print("PCA saved")

if not os.path.isfile(filename):
    files = get_files_names(["/data/data/00.raw/raw/Adr_Expe_28-08_07-10/raspi1/"], "01_October")
    data = read_directory("data-test")
#    data = read_directory("data-test")
    data = get_event_list(data, round(temporal_duration / waterfall_duration * waterfall_length), int(1500 / nb_features_macro)) # 1mn
    data.tofile(filename)
else:
    print("Features déjà extraites !")
