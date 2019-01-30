#!/usr/bin/env python3

from multimodels import *
from preprocess import *
from config import Config
from models import LOF, OCSVM, MultiModels
import copy
import os

config = Config()
nb_features = config.get_config_eval("nb_features")

with open("train_folders") as f:
    folders = f.readlines()
folders = [x.strip() for x in folders]
#folders = [folders[0]] # TODO virer
prefix = config.get_config("section")
model_subsample = config.get_config_eval("model_subsample")

files = [os.path.join(prefix, "features-"+d.split("/")[-1]) for d in folders]
print("Learning from",files)

periods = [period_night, period_day]
models = MultiModels()

model_name = config.get_config("model_micro")
if model_name == "OCSVM":
    detector_model = OCSVM()
elif model_name == "LOF":
    detector_model = LOF()
else:
    raise ValueError("Unknown model: "+model_name)

outputname = os.path.join(prefix, "micro-"+detector_model.__class__.__name__+".joblib")
if not os.path.isfile(outputname):
    all_data = np.concatenate([np.fromfile(f).reshape(-1, nb_features + 1) for f in files])
    for p in periods:
        detector = copy.deepcopy(detector_model)
        data = extract_period(all_data, p)
        data = subsample(data, model_subsample)
        if data.shape[0] > 0:
            print("Learning for",p.__name__,"from",data.shape[0],"examples")
            detector.learn(data[:,1:]) # should not learn the timestamp
            print("Learn threshold")
            detector.learn_threshold(data[:,1:])
            models.add_model(detector, p)
        else:
            print("No data to learn period",p.__name__)
    models.save(outputname)
else:
    print("Micro model already learned!",outputname)
