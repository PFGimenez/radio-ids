#!/usr/bin/env python3

import numpy as np
from sklearn.decomposition import PCA
from preprocess import *
from hmm import HMM
import os.path
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

config = Config()
train_input_filename = os.path.join(config.get_config("section"), "train_"+config.get_config("macro_features_stage_1"))
train_output_filename = os.path.join(config.get_config("section"), "train_"+config.get_config("macro_features_stage_2"))
train_time_input_filename = os.path.join(config.get_config("section"), "time_train_"+config.get_config("macro_features_stage_1"))

test_input_filename = os.path.join(config.get_config("section"), "test_"+config.get_config("macro_features_stage_1"))
test_output_filename = os.path.join(config.get_config("section"), "test_"+config.get_config("macro_features_stage_2"))
test_time_input_filename = os.path.join(config.get_config("section"), "time_test_"+config.get_config("macro_features_stage_1"))

features_per_band = config.get_config_eval("features_per_band")
nb_macro_band = config.get_config_eval("nb_macro_band")
temporal_duration = config.get_config_eval("macro_feature_duration")
waterfall_duration = config.get_config_eval("waterfall_duration")
waterfall_length = config.get_config_eval("waterfall_dimensions")[0]

nb_features_macro = config.get_config_eval("nb_features_macro")

if os.path.isfile(train_output_filename) and os.path.isfile(test_output_filename):
    print("Features stage 2 already learnt!")
else:
    train_data = np.fromfile(train_input_filename).reshape(-1, features_per_band * nb_macro_band)
    train_data_time = np.fromfile(train_time_input_filename)
    test_data = np.fromfile(test_input_filename).reshape(-1, features_per_band * nb_macro_band)
    test_data_time = np.fromfile(test_time_input_filename)
    print(train_data.shape, train_data_time.shape)
    print(test_data.shape, test_data_time.shape)

    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    before = train_data.shape[1]
    pca = PCA(nb_features_macro, svd_solver="full")
    pca.fit_transform(train_data)

    train_data = pca.transform(train_data)
    test_data = pca.transform(test_data)

    print("Features number: " + str(before) + " -> " + str(train_data.shape[1]))
    print("Explained variance:",sum(pca.explained_variance_ratio_))

    # on ajoute le timestamp
    np.c_[train_data, train_data_time]
    np.c_[test_data, test_data_time]

    print(train_data.shape,test_data.shape)

    train_data.tofile(train_output_filename)
    test_data.tofile(test_output_filename)

