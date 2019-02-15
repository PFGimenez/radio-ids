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
model_initial_subsample = config.get_config_eval("model_initial_subsample")

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
        initial_data = extract_period(all_data, p)
        if initial_data.shape[0] > 0:
            print("Initial size:",initial_data.shape[0])
            batch = subsample(initial_data, model_initial_subsample)
            step = int(initial_data.shape[0] * model_subsample)
            old_score = None
            score = None
            while old_score == None or score <= old_score:
                old_score = score
                print("Learning for",p.__name__,"from",batch.shape[0],"examples")
                best = copy.deepcopy(detector)
                detector.learn(batch[:,1:]) # should not learn the timestamp
                (new_batch, score) = detector.get_worse_score(initial_data, step)
                batch = np.vstack((batch, new_batch))
            detector = best
            print("Learn threshold")
            detector.learn_threshold(initial_data[:,1:])
            models.add_model(detector, p)
        else:
            print("No data to learn period",p.__name__)
    models.save(outputname)
else:
    print("Micro model already learned!",outputname)
