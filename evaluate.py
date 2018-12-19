#!/usr/bin/env python3

from multimodels import *
from preprocess import *
import numpy as np
from config import Config
import os
from extractor import MultiExtractors
from autoencodercnn import CNN

class Evaluator:

    """
        Recall (= true positive rate): attaques correctement trouvées / toutes les vraies attaques. Proportion d'attaques qui ont été détectées (si on prédit toujours une attaque, le recall vaut 1)
        Precision: nb attaques correctement trouvées / nb attaques détectées. Proportion de détection pertinentes (si on prédit toujours une attaque, la précision sera proche de 0)
        F-score = 2 * (precision * rappel) / (precision + rappel)
    """

    def __init__(self, identifier, all_attack):
        """
            format all_attack : shape (-1,3)
            column 0 : attack start timestamp
            column 1 : attack end itemstamp
            column 2 : attack identifier
        """
        self._id = identifier
        # TODO : ou remplacer identifier par une fonction (pour agréger toutes les attaques bluetooth par exemple)
        self._attack = all_attack[all_attack[:,2] == identifier]
        print(self._attack.shape[0],"attacks on",identifier)

    def is_in_attack(self, timestamp):
        for a in self._attack:
            if timestamp >= a[0] and timestamp <= a[1]:
                return True
        return False

    def evaluate(self, detected_positive, detected_negative):
        """
            Prediction : shape (-1,2)
            column 0 : timestamp
            column 1 : true iff detection
        """

        total_positives = len(detected_positive)
        true_positive = sum(list(map(self.is_in_attack, detected_positive)))
        false_positive = total_positives - true_positive

        total_negatives = len(detected_negative)
        false_negative = sum(list(map(self.is_in_attack, detected_negative)))
        true_negative = total_negatives - false_negative

        if true_positive + false_negative == 0:
            recall = 1
        else:
            recall = true_positive / (true_positive + false_negative)

        if total_positives == 0:
            precision = 1
        else:
            precision = true_positive / total_positives

#        print("total pos",total_positives, "total negative",total_negatives)
        if precision != 0 and recall != 0:
            fmeasure = 2*(precision + recall) / (precision * recall)
        else:
            fmeasure = float('nan')
        print("tp",true_positive, "tn",true_negative, "fp",false_positive,"fn", false_negative,"precision",precision,"recall",recall,"f-measure",fmeasure)

# lecture config

config = Config()
attack = np.loadtxt(os.path.join(config.get_config("section"), "logattack"))

identifiers = np.unique(attack[:,2])
evaluators = [Evaluator(i, attack) for i in identifiers]

nb_features = config.get_config_eval("nb_features")
nb_features_macro = config.get_config_eval("nb_features_macro")
prefix = config.get_config("section")

# chargement du jeu de données de test micro

with open("test_folders") as f:
    folders = f.readlines()
directories = [x.strip() for x in folders]

files = [os.path.join(prefix, "features-"+d.split("/")[-1]) for d in directories]
print(files)
data = np.concatenate([np.fromfile(f).reshape(-1, nb_features + 1) for f in files])
print("data micro:",data.shape)

test_macro_filename = os.path.join(config.get_config("section"), "test_"+config.get_config("macro_features_stage_2"))
data_macro = np.fromfile(test_macro_filename).reshape(-1, nb_features_macro + 1)
print("data macro:",data_macro.shape)

# modèle micro

models = MultiModels()
models.load(os.path.join(prefix, "micro-OCSVM.joblib"))

# modèle macro
models_macro = MultiModels()
models_macro.load(os.path.join(prefix, "macro-HMM.joblib"))

# autoencoders
bands = config.get_config_eval('waterfall_frequency_bands')
dims = config.get_config_eval('autoenc_dimensions')
extractors = MultiExtractors()

for j in range(len(bands)):
    (i,s) = bands[j]
    m = CNN(i, s, dims[j], 0)
    extractors.load(i, s, m)

with open("train_folders") as f:
    folders_test = f.readlines()
folders_test = [x.strip() for x in folders]


# évaluation


path_examples = os.path.join(prefix, "results-OCSVM-micro.joblib")
path_examples_macro = os.path.join(prefix, "results-HMM-macro.joblib")
path_examples_extractors = os.path.join(prefix, "results-autoenc.joblib")

def predict(models, path_examples, data):
    try:
        # chargement des prédictions si possible
        (example_pos, example_neg) = joblib.load(path_examples)
    except:
        memory_size = models.get_memory_size()
        example_pos = []
        example_neg = []
        i = 0
        memory = []

        for f in data:
            if i % 100 == 0:
                print(i,"/",len(data))
            i += 1

            if len(memory) == memory_size:
                memory.pop(0)
            memory.append(f[1:])

#            print("Memory",np.array(memory).shape) # TODO
            if models.predict(np.array(memory), f[0]):
                example_pos.append(f[0])
            else:
                example_neg.append(f[0])

        joblib.dump((example_pos, example_neg), path_examples)
    return (example_pos, example_neg)

def predict_extractors(extractors, path_examples, folders_test):
    try:
        # chargement des prédictions si possible
        (example_pos, example_neg) = joblib.load(path_examples)
    except:
        example_pos = []
        example_neg = []
        i = 0

        paths = [os.path.join(directory,f) for directory in folders_test for f in sorted(os.listdir(directory))]
        for fname in paths:
            timestamp = int(os.path.split(fname)[1])
            data = read_file(fname)
            if i % 100 == 0:
                print(i,"/",len(paths))
            i += 1

            if extractors.predict(data):
                example_pos.append(timestamp)
            else:
                example_neg.append(timestamp)

        joblib.dump((example_pos, example_neg), path_examples)
    return (example_pos, example_neg)



print("Prediction for micro…")
(example_pos, example_neg) = predict(models, path_examples, data)
print("Prediction for macro…")
(example_pos_macro, example_neg_macro) = predict(models_macro, path_examples_macro, data_macro)
print("Prediction for autoencoders…")
(example_pos_extractors, example_neg_extractors) = predict_extractors(extractors, path_examples_extractors, folders_test)

for e in evaluators:
    print("***",e._id)
    print("Results micro: ",end='')
    e.evaluate(example_pos, example_neg)
    print("Results macro: ",end='')
    e.evaluate(example_pos_macro, example_neg_macro)
#    print("Results micro and macro")
#    e.evaluate(list(set(example_pos+example_pos_macro)),
#               list(set(example_neg+example_neg_macro)))
    print("Results autoencoders: ",end='')
    e.evaluate(example_pos_extractors, example_neg_extractors)

