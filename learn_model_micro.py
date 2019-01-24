#!/usr/bin/env python3

from multimodels import *
from preprocess import *
from config import Config
from svm import OCSVM
from local_outlier_factor import LOF
import copy
import os

config = Config()
nb_features = config.get_config_eval("nb_features")

with open("train_folders") as f:
    folders = f.readlines()
folders = [x.strip() for x in folders]

prefix = config.get_config("section")

files = [os.path.join(prefix, "features-"+d.split("/")[-1]) for d in folders]
print("Learning from",files)
all_data = np.concatenate([np.fromfile(f).reshape(-1, nb_features + 1) for f in files])

periods = [period_night, period_day]
models = MultiModels()

detector_model = OCSVM()
#detector_model = LOF()
outputname = os.path.join(prefix, "micro-"+detector_model.__class__.__name__+".joblib")

if not os.path.isfile(outputname):
    for p in periods:
        detector = copy.deepcopy(detector_model)
        data = extract_period(all_data, p)
        if data.shape[0] > 0:
            print("Learning for",p.__name__,"from",data.shape[0],"examples")
            detector.learn(data[:,1:]) # should not learn the timestamp
            models.add_model(detector, p)
        else:
            print("No data to learn period",p.__name__)

    models.save(outputname)
else:
    print("Micro model already learned!")
