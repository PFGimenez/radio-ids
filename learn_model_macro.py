#!/usr/bin/env python3

import numpy as np
from preprocess import *
from models import MultiModels
from multimodels import *
import random
from models import HMM
import os

config = Config()
train_filename = os.path.join(config.get_config("section"), "train_"+config.get_config("macro_features_stage_2"))
nb_features_macro = config.get_config_eval("nb_features_macro")

all_data = np.fromfile(train_filename).reshape(-1,nb_features_macro + 1)

print(all_data.shape)

detector = HMM(10)
output = os.path.join(config.get_config("section"), "macro-"+detector.__class__.__name__+".joblib")
periods = [period_night, period_day]
models = MultiModels()

if not os.path.isfile(output):
    for p in periods:
        detector = HMM(10)
        data = extract_period(all_data, p)
        if data.shape[0] > 0:
            print("Learning for",p.__name__,"from",data.shape[0],"examples")
            detector.learn(data[:,1:]) # should not learn the timestamp
            print("Learning threshold")
            detector.learn_threshold(data[:,1:])
            models.add_model(detector, p)
        else:
            print("No data to learn period",p.__name__)
    models.save(output)
else:
    print("Models already learnt!",output)
