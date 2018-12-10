#!/usr/bin/env python3

import numpy as np
from preprocess import *
import random
from hmm import HMM

config = Config()
nb_features_macro = config.get_config_eval("nb_features_macro")
filename = os.path.join(config.get_config("section"), config.get_config("events_filename"))
data = np.fromfile(filename).reshape(-1,nb_features_macro)

detector = HMM(4, 0.1)
print(data.shape)
train_data = data[:1000,]
test_data = data[1000:,]
detector.learn(train_data)
predictions = test_prediction(test_data, detector)
print(np.max(predictions))
print(np.min(predictions))
print(np.mean(predictions))


#print(detector.predict_list(test_data))
