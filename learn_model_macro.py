#!/usr/bin/env python3

import numpy as np
from preprocess import *
import random
from hmm import HMM

config = Config()
train_filename = os.path.join(config.get_config("section"), "train_"+config.get_config("macro_features_stage_2"))
test_filename = os.path.join(config.get_config("section"), "test_"+config.get_config("macro_features_stage_2"))
nb_features_macro = config.get_config_eval("nb_features_macro")

train_data = np.fromfile(train_filename).reshape(-1,nb_features_macro)
test_data = np.fromfile(test_filename).reshape(-1,nb_features_macro)

print(train_data.shape, test_data.shape)

detector = HMM(10)
detector.learn(train_data)
models.save(os.path.join(prefix, "macro-"+detector.__class__.__name__+".joblib"))
