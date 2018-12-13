#!/usr/bin/env python3

from multimodels import *
from preprocess import *
import numpy as np
from config import Config
import os

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

        recall = true_positive / (true_positive + false_negative)
        precision = true_positive / total_positives

        print("total pos",total_positives, "total negative",total_negatives)
        print("tp",true_positive, "tn",true_negative, "fp",false_positive,"fn", false_negative)
        print("precision",precision,"recall",recall)
        if precision != 0 and recall != 0:
            print("f-measure",2*(precision + recall) / (precision * recall))

# lecture config

config = Config()
attack = np.loadtxt(os.path.join(config.get_config("section"), "logattack"))

identifiers = np.unique(attack[:,2])
evaluators = [Evaluator(i, attack) for i in identifiers]

nb_features = config.get_config_eval("nb_features")
prefix = config.get_config("section")

# chargement du jeu de données de test micro

with open("test_folders") as f:
    folders = f.readlines()
directories = [x.strip() for x in folders]

files = [os.path.join(prefix, "features-"+d.split("/")[-1]) for d in directories]
print(files)
data = np.concatenate([np.fromfile(f).reshape(-1, nb_features + 1) for f in files])
print(data.shape)

# modèle micro

models = MultiModels()
models.load(os.path.join(prefix, "micro-ocsvm.joblib"))

# chargement du jeu de données de test macro

features-macro = os.path.join(config.get_config("section"), "test_"+config.get_config("macro_features_stage_2"))

# modèle macro
models_macro = MultiModels()
models.load(os.path.join(prefix, "macro-HMM.joblib"))

# évaluation

memory_size = models.get_memory_size()
memory = []

path_examples = os.path.join(prefix, "results-ocsvm-micro.joblib")
try:
    # chargement des prédictions si possible
    (example_pos, example_neg) = joblib.load(path_examples)
except:
    example_pos = []
    example_neg = []
    i = 0

    for f in data:
        if i % 100 == 0:
            print(i,"/",len(data))
        i += 1

        if len(memory) == memory_size:
            memory.pop(0)
        memory.append(f[1:])

        if models.predict(np.array(memory), f[0]):
            example_pos.append(f[0])
        else:
            example_neg.append(f[0])

    joblib.dump((example_pos, example_neg), path_examples)

path_examples_macro = os.path.join(prefix, "results-hmm-macro.joblib")
try:
    # chargement des prédictions si possible
    (example_pos_macro, example_neg_macro) = joblib.load(path_examples_macro)
except:
    example_pos_macro = []
    example_neg_macro = []
    i = 0

    for f in data:
        if i % 100 == 0:
            print(i,"/",len(data))
        i += 1

        if len(memory) == memory_size:
            memory.pop(0)
        memory.append(f[1:])

        if models.predict(np.array(memory), f[0]):
            example_pos.append(f[0])
        else:
            example_neg.append(f[0])

    joblib.dump((example_pos, example_neg), path_examples)



for e in evaluators:
    e.evaluate(example_pos, example_neg)

