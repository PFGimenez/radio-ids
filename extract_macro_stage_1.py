#!/usr/bin/env python3

import numpy as np
from sklearn.decomposition import PCA
from preprocess import *
from hmm import HMM
import os.path
from sklearn.externals import joblib

def get_event_list(data, temporal_step, spectral_step):
#    print(temporal_step, spectral_step)
#    print(data.shape)
    data = np.concatenate(data)
#    print(data.shape)
    data = decompose_raw(data, (temporal_step, spectral_step))
#    print(data.shape)
#    data = data.reshape((data.shape[1], data.shape[2], data.shape[3]))
#    print(data.shape)
#    print(data.shape)
#    data = np.concatenate((np.amax(data, axis=2),
#                          np.mean(data, axis=2)
#                          ), axis=1)
    out = np.array([np.amax(data, axis=(2,3)),
           np.mean(data, axis=(2,3)),
           np.std(data, axis=(2,3)),
           np.median(data, axis=(2,3))])
#    print(out.shape)
    out = np.hstack(out)
#    print(out.shape)
    return out

config = Config()
nb_features_macro = config.get_config_eval("nb_features_macro")
temporal_duration = config.get_config_eval("macro_feature_duration")
waterfall_duration = config.get_config_eval("waterfall_duration")
waterfall_length = config.get_config_eval("waterfall_dimensions")[0]
output = os.path.join(config.get_config("section"), config.get_config("macro_features_stage_1"))

if not os.path.isfile(output):
#    files = get_files_names(["/data/data/00.raw/raw/Adr_Expe_28-08_07-10/raspi1/"], "01_October")
#    data = read_directory("/data/data/00.raw/raw/Adr_Expe_28-08_07-10/raspi1/learn_dataset_23_August")
    data = read_directory("data-test2")
    data = get_event_list(data, round(temporal_duration / waterfall_duration * waterfall_length), int(1500 / nb_features_macro)) # 1mn
    print("Features stage 1 shape:",data.shape)
    data.tofile(output)
else:
    print("Features stage 1 already extracted!")
