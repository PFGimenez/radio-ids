#!/usr/bin/env python3

from models import MultiModels, MultiExtractors
from preprocess import *
import numpy as np
from config import Config
import os
from autoencodercnn import CNN
import time
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import datetime

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
        if self._id == None:
            self._id = "All"
            self._attack = np.array(all_attack[:,1:].astype(np.integer))
        else:
            self._attack = np.array(all_attack[all_attack[:,0] == identifier][:,1:].astype(np.integer))
#        print(self._attack)
        self._seen_attack = []
        self._cumulative_seen_attack = []
        print(len(self._attack),"attacks on",self._id)

    def is_in_attack(self, timestamp):
        for a in self._attack:
            if timestamp >= a[0] and timestamp <= a[1]:
                return True
        return False
#        return np.any([timestamp >= a[0] and timestamp <= a[1] for a in self._attack])

    def is_in_attack_detected(self, timestamp):
        for a in self._attack:
            if timestamp >= a[0] and timestamp <= a[1]:
                if a[0] not in self._seen_attack:
                    self._seen_attack.append(a[0])
                if a[0] not in self._cumulative_seen_attack:
                    self._cumulative_seen_attack.append(a[0])
                return True
        return False

    def evaluate(self, detected_positive_dico, detected_negative_dico):
        """
            Prediction : shape (-1,2)
            column 0 : timestamp
            column 1 : true iff detection
        """
        self._seen_attack = []

        detected_positive = np.array([k for k in detected_positive_dico])
        detected_negative = np.array([k for k in detected_negative_dico])
        total_positives = len(detected_positive)
        true_positive_list = detected_positive[list(map(self.is_in_attack_detected, detected_positive))]
        false_positive_list = detected_positive[[t not in true_positive_list for t in detected_positive]]
        true_positive = len(true_positive_list)
        false_positive = total_positives - true_positive

        total_negatives = len(detected_negative)
        false_negative_list = detected_negative[list(map(self.is_in_attack, detected_negative))]
        true_negative_list = detected_negative[[t not in false_negative_list for t in detected_negative]]
        false_negative = len(false_negative_list)
        true_negative = total_negatives - false_negative

        print("Detected : ",len(self._seen_attack),"/",len(self._attack))

        true_positive_score = np.array([detected_positive_dico[t] for t in true_positive_list])
        false_negative_score = np.array([detected_negative_dico[t] for t in false_negative_list])
        false_positive_score = np.array([detected_positive_dico[t] for t in false_positive_list])
        true_negative_score = np.array([detected_negative_dico[t] for t in true_negative_list])

        if true_positive + false_negative == 0:
            recall = 1
        else:
            recall = true_positive / (true_positive + false_negative)

        # TODO
        recall = len(self._seen_attack) / len(self._attack)

        if total_positives == 0:
            precision = 1
        else:
            precision = true_positive / total_positives

        # TODO
        precision = len(self._seen_attack) / (false_positive + len(self._seen_attack))

#        print("total pos",total_positives, "total negative",total_negatives)
        if precision + recall != 0:
            fmeasure = 2*(precision * recall) / (precision + recall)
        else:
            fmeasure = float('nan')
        print("tp",true_positive, "tn",true_negative, "fp",false_positive,"fn", false_negative,"precision",precision,"recall",recall,"f-measure",fmeasure)

        if show_hist:
            plt.hist(true_positive_score, color='red', bins=100, histtype='step', log=True)
            plt.hist(false_negative_score, color='magenta', bins=100, histtype='step', log=True)
            plt.hist(false_positive_score, color='cyan', bins=100, histtype='step', log=True)
            plt.hist(true_negative_score, color='blue', bins=100, histtype='step', log=True)
            plt.title(self._id)
            plt.show()

        if show_time:
            for a in self._attack:
#            print(a[1]-a[0])
                plt.hlines(1, a[0], a[1], color='red')
            for t in detected_positive:
#            print(t)
                plt.vlines(t, 0.85, 0.9, color='red' if self.is_in_attack(t) else 'blue')
            plt.show()

        print("Cumulative detected : ",len(self._cumulative_seen_attack),"/",len(self._attack))

def predict(models, path_examples, data):
    try:
        # chargement des prédictions si possible
        (example_pos, example_neg) = joblib.load(path_examples)
        print("Predictions loaded")
    except:
        start = time.time()
        memory_size = models.get_memory_size()
        example_pos = {}
        example_neg = {}
#        memory = []

#        data = data[20000:30000]
#        for i in range(memory_size+1,1000):
        for i in range(memory_size+1,len(data)): # TODO
            if i % 100 == 0:
                print(i,"/",len(data))

#            if len(memory) == memory_size:
#                memory.pop(0)
#            memory.append(f[1:]) # hors timestamp
            d = data[i-1-memory_size:i]
#            print(data.shape, d.shape, d[0,0],d[0,1:])
#            print(d[0,0].shape, d[:,1:].shape)
            score = models.get_score(d[:,1:], d[0,0])
            if models.predict_thr(score, optimistic=False):
#                print("Attack detected at",d[0,0])
                if isinstance(score, dict):
                    example_pos[d[0,0]] = score[max(score,key=score.get)]
                else:
                    example_pos[d[0,0]] = score
            else:
                if isinstance(score, dict):
                    # on enregistre le plus haut score (ne sert qu'à l'affichage)
                    example_neg[d[0,0]] = score[max(score,key=score.get)]
                else:
                    example_neg[d[0,0]] = score

        end = time.time()
        print("Detection time:",(end-start),"s")
        joblib.dump((example_pos, example_neg), path_examples)
    return (example_pos, example_neg)

def predict_extractors(extractors, path_examples, folders_test):
    try:
        # chargement des prédictions si possible
        (example_pos, example_neg) = joblib.load(path_examples)
        print("Predictions loaded")
    except:
        start = time.time()
        example_pos = {}
        example_neg = {}
        i = 0

        paths = [os.path.join(directory,f) for directory in folders_test for f in sorted(os.listdir(directory))]
        for fname in paths:
            timestamp = int(os.path.split(fname)[1])
            data = read_file(fname, quant=True)
            if i % 100 == 0:
                print(i,"/",len(paths))
            i += 1

            score = extractors.get_score(data, timestamp)
            if extractors.predict_thr(score,optimistic=False):
                if isinstance(score, dict):
                    example_pos[timestamp] = score[max(score,key=score.get)]
                else:
                    example_pos[timestamp] = score
            else:
                if isinstance(score, dict):
                    # on enregistre le plus haut score (ne sert qu'à l'affichage)
                    example_neg[timestamp] = score[max(score,key=score.get)]
                else:
                    example_neg[timestamp] = score

        end = time.time()
        print("Detection time:",(end-start),"s")
        joblib.dump((example_pos, example_neg), path_examples)
    return (example_pos, example_neg)




# lecture config

config = Config()

with open("test_folders") as f:
    folders = f.readlines()
directories = [x.strip() for x in folders]

attack = np.loadtxt(os.path.join(config.get_config("section"), "logattack"), dtype='<U13')

# TODO : on ne garde les attaques que du 23 janvier
attack = np.array([a for a in attack if datetime.datetime.fromtimestamp(int(a[1])/1000).day == 23])

identifiers = np.unique(attack[:,0])
print("Attacks list:",identifiers)

#evaluators = [Evaluator(i, attack) for i in identifiers]
evaluators = []
evaluators.append(Evaluator(None, attack))
nb_features = config.get_config_eval("nb_features")
nb_features_macro = config.get_config_eval("nb_features_macro")
prefix = config.get_config("section")

# chargement du jeu de données de test micro

show_time = True
show_hist = False
use_micro = True
use_macro = True
use_autoenc = True

# modèle micro

if use_micro:
    models = MultiModels()
    try:
        models.load(os.path.join(prefix, "micro-OCSVM.joblib"))
        files = [os.path.join(prefix, "features-"+d.split("/")[-1]) for d in directories]
        print(files)
        data = np.concatenate([np.fromfile(f).reshape(-1, nb_features + 1) for f in files])
        print("data micro:",data.shape)
    except Exception as e:
        print("Loading failed:",e)
        use_micro = False

# modèle macro
if use_macro:
    models_macro = MultiModels()
    try:
        models_macro.load(os.path.join(prefix, "macro-HMM.joblib"))
        test_macro_filename = os.path.join(config.get_config("section"), "test_"+config.get_config("macro_features_stage_2"))
        data_macro = np.fromfile(test_macro_filename).reshape(-1, nb_features_macro + 1)
        print("data macro:",data_macro.shape)
    except Exception as e:
        print("Loading failed:",e)
        use_macro = False

# autoencoders
bands = config.get_config_eval('waterfall_frequency_bands')
dims = config.get_config_eval('autoenc_dimensions')
extractors = MultiExtractors()

if use_autoenc:
    for j in range(len(bands)):
        (i,s) = bands[j]
        m = CNN(i, s, dims[j], 0)
        extractors.load_model(m)

with open("train_folders") as f:
    folders_test = f.readlines()
folders_test = [x.strip() for x in folders]


# évaluation


path_examples = os.path.join(prefix, "results-OCSVM-micro.joblib")
path_examples_macro = os.path.join(prefix, "results-HMM-macro.joblib")
path_examples_extractors = os.path.join(prefix, "results-autoenc.joblib")

if use_micro:
    print("Prediction for micro…")
    (example_pos, example_neg) = predict(models, path_examples, data)
if use_macro:
    print("Prediction for macro…")
    (example_pos_macro, example_neg_macro) = predict(models_macro, path_examples_macro, data_macro)
if use_autoenc:
    print("Prediction for autoencoders…")
    (example_pos_extractors, example_neg_extractors) = predict_extractors(extractors, path_examples_extractors, folders_test)

for e in evaluators:
    print("***",e._id)
    if use_micro:
        print("Results micro: ",end='')
        e.evaluate(example_pos, example_neg)
    if use_macro:
        print("Results macro: ",end='')
        e.evaluate(example_pos_macro, example_neg_macro)
#    print("Results micro and macro")
#    e.evaluate(list(set(example_pos+example_pos_macro)),
#               list(set(example_neg+example_neg_macro)))
    if use_autoenc:
        print("Results autoencoders: ",end='')
        e.evaluate(example_pos_extractors, example_neg_extractors)

