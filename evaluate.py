#!/usr/bin/env python3
import sys
from models import MultiModels, MultiExtractors
from preprocess import *
import numpy as np
from config import Config
import os
from autoencodercnn import CNN
import time
from sklearn.externals import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import datetime
import itertools

def get_derivative(scores):
    out = {}
    keys = list(scores.keys())
    for i in range(len(scores)-1):
        score_old = scores[keys[i]]
        score_new = scores[keys[i+1]]
        if isinstance(score_old, dict):
            tmp = {}
            for k in score_old:
                tmp[k] = score_new.get(k) - score_old.get(k)
            out[keys[i+1]] = tmp
        else:
            out[keys[i+1]] = score_new - score_old
    return out

class Evaluator:

    """
        Recall (= true positive rate): attaques correctement trouvées / toutes les vraies attaques. Proportion d'attaques qui ont été détectées (si on prédit toujours une attaque, le recall vaut 1)
        Precision: nb attaques correctement trouvées / nb attaques détectées. Proportion de détection pertinentes (si on prédit toujours une attaque, la précision sera proche de 0)
        F-score = 2 * (precision * rappel) / (precision + rappel)
    """

    def __init__(self, all_attack, identifier=None):
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
            # self._attack = np.array(all_attack[:,:2].astype(np.float))
        else:
            self._attack = np.array(all_attack[all_attack[:,0] == identifier][:,1:].astype(np.integer))
            # self._attack = np.array(all_attack[all_attack[:,2] == identifier][:,:2].astype(np.float))

        # self._attack += 1530576000
        # self._attack -= 7251 # TODO décalage
        # self._attack *= 1000
        # self._attack = self._attack.astype(np.integer)

        # if self._id == "All":
        #     f = open("logattack_fixed","w+")
        #     for i in range(len(self._attack)):
        #         a = self._attack[i]
        #         f.write(all_attack[i, 2]+" "+str(a[0])+" "+str(a[1])+"\r\n")
        #     f.close()
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

    def roc(self, detected_positive_dico, scores, models, typestr):
        pass # TODO

    def evaluate(self, detected_positive_dico, scores, models, typestr):
        """
            Prediction : shape (-1,2)
            column 0 : timestamp
            column 1 : true iff detection
        """
        self._seen_attack = []

        detected_positive = np.array([k for k in detected_positive_dico])
        # detected_negative = np.array([k for k in detected_negative_dico])
        total_positives = len(detected_positive)
        true_positive_list = detected_positive[list(map(self.is_in_attack_detected, detected_positive))]
        false_positive_list = detected_positive[[t not in true_positive_list for t in detected_positive]]
        true_positive = len(true_positive_list)
        false_positive = total_positives - true_positive

        # total_negatives = len(detected_negative)
        # false_negative_list = detected_negative[list(map(self.is_in_attack, detected_negative))]
        # true_negative_list = detected_negative[[t not in false_negative_list for t in detected_negative]]
        # false_negative = len(false_negative_list)
        # true_negative = total_negatives - false_negative

        print("Detected : ",len(self._seen_attack),"/",len(self._attack))

        true_positive_score = np.array([detected_positive_dico[t] for t in true_positive_list])
        # false_negative_score = np.array([detected_negative_dico[t] for t in false_negative_list])
        false_positive_score = np.array([detected_positive_dico[t] for t in false_positive_list])
        # true_negative_score = np.array([detected_negative_dico[t] for t in true_negative_list])

        # if true_positive + false_negative == 0:
        #     recall = 1
        # else:
        #     recall = true_positive / len(self._attack)

        recall = len(self._seen_attack) / len(self._attack)

        # if total_positives == 0:
        #     precision = 1
        # else:
        #     precision = true_positive / total_positives
        if false_positive + len(self._seen_attack) == 0:
            precision = 1
        else:
            precision = len(self._seen_attack) / (false_positive + len(self._seen_attack))

#        print("total pos",total_positives, "total negative",total_negatives)
        if precision + recall != 0:
            fmeasure = 2*(precision * recall) / (precision + recall)
        else:
            fmeasure = float('nan')
        print("tp",true_positive, "fp",false_positive,"precision",precision,"recall",recall,"f-measure",fmeasure)
        # print("tp",true_positive, "tn",true_negative, "fp",false_positive,"fn", false_negative,"precision",precision,"recall",recall,"f-measure",fmeasure)

        if show_hist:
            plt.hist(true_positive_score, color='red', bins=100, histtype='step', log=True)
            # plt.hist(false_negative_score, color='magenta', bins=100, histtype='step', log=True)
            plt.hist(false_positive_score, color='cyan', bins=100, histtype='step', log=True)
            # plt.hist(true_negative_score, color='blue', bins=100, histtype='step', log=True)
            plt.title(self._id)
            plt.show()

        if show_time and self._id=="All":
            x = list(scores.keys())
            if isinstance(models, MultiModels):
                nbcols = len(models._models)
            else:
                nbcols = 1
            if nbcols > 1:
                fig, ax = plt.subplots(nrows=1, ncols=nbcols)
                ax = np.expand_dims(ax, 1)
            else:
                fig, ax = plt.subplots(nrows=1, ncols=1)
                ax = np.expand_dims(ax, 1)
                ax = np.expand_dims(ax, 2)
            hfmt = mdates.DateFormatter('%d/%m %H:%M:%S')
            for row in ax:
                for col in row:
                    col.xaxis.set_major_formatter(hfmt)
                    plt.setp(col.get_xticklabels(), rotation=15)
            for a in self._attack:
                for row in ax:
                    for col in row:
                        col.hlines(0,
                           mdates.date2num(datetime.datetime.fromtimestamp(a[0]/1000)),
                           mdates.date2num(datetime.datetime.fromtimestamp(a[1]/1000)),
                           color='red')

            for t in detected_positive:
                # TODO
                for row in ax:
                    for col in row:
                        col.hlines(0.01,
                           mdates.date2num(datetime.datetime.fromtimestamp(t/1000)),
                           mdates.date2num(datetime.datetime.fromtimestamp(t/1000+10)),
                           color='blue')



            # for t in detected_positive:
            #     plt.vlines(t, 0.85, 0.9, color='red' if self.is_in_attack(t) else 'blue')
            if isinstance(models, MultiModels):
                if typestr == "autoenc":
                    labels = ["400-500","800-900","2400-2500"]
                elif typestr == "micro":
                    labels = [f.__name__ for (f,_) in models._models]

                i = 0
                for row in ax:
                    for col in row:
                        if i < nbcols:
                            # col.hlines( # TODO
                            val = [scores[k].get(i) for k in x]
                            x, val = zip(*sorted(zip(x, val)))
                            x_dates = mdates.date2num([datetime.datetime.fromtimestamp(t/1000) for t in x])
                            col.plot(x_dates, val, alpha=0.7, label=labels[i])
                            i += 1
                            col.legend(loc='upper left')

            else:
                val = list(scores.values())
                x, val = zip(*sorted(zip(x, val)))
                plt.plot(x, val)
            plt.show()

        print("Cumulative detected : ",len(self._cumulative_seen_attack),"/",len(self._attack))
        return (true_positive, false_positive)

def predict(models, scores, threshold):
    start = time.time()
    memory_size = models.get_memory_size()
    example_pos = {}
    example_neg = {}
#        memory = []

#        data = data[20000:30000]
#        for i in range(memory_size+1,1000):
    for i in range(memory_size+1,len(data)): # TODO
        # if i % 100 == 0:
            # print(i,"/",len(data))

#            if len(memory) == memory_size:
#                memory.pop(0)
#            memory.append(f[1:]) # hors timestamp
        d = data[i-1-memory_size:i]
#            print(data.shape, d.shape, d[0,0],d[0,1:])
#            print(d[0,0].shape, d[:,1:].shape)
        score = scores[d[0,0]]
        if models.predict_thr(score, optimistic=False, threshold=threshold):
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
    return (example_pos, example_neg)

def scores_micro_macro(models, path_examples, data):
    try:
        # chargement des prédictions si possible
        scores = joblib.load(path_examples)
        print("Scores loaded")
    except:
        start = time.time()
        memory_size = models.get_memory_size()
        scores = {}
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
            scores[d[0,0]] = models.get_score(d[:,1:], d[0,0])

        end = time.time()
        print("Scoring time:",(end-start),"s")
        joblib.dump(scores, path_examples)
    return scores

def score_extractors(extractors, path_examples, folders_test):
    try:
        # chargement des prédictions si possible
#        (example_pos, example_neg) = joblib.load(path_examples)
        scores = joblib.load(path_examples)
        print("Scores loaded")
    except:
        scores = {}
        start = time.time()
        i = 0

        paths = [os.path.join(directory,f) for directory in folders_test for f in sorted(os.listdir(directory))]
        for fname in paths:
            timestamp = int(os.path.split(fname)[1])
            data = read_file(fname, quant=True)
            if i % 100 == 0:
                print(i,"/",len(paths))
            i += 1
            scores[timestamp] = extractors.get_score(data, timestamp)
        end = time.time()
        print("Scoring time:",(end-start),"s")
        joblib.dump(scores, path_examples)
    return scores

def predict_extractors(extractors, scores, threshold_autoencoder):
    start = time.time()
    consecutive = 0
    example_pos = {}
    example_neg = {}

    for timestamp in scores:
        score = scores[timestamp]
        if extractors.predict_thr(score,optimistic=False,threshold=threshold_autoencoder):
            consecutive += 1
        else:
            # if consecutive > 0:
                # print(consecutive)
            consecutive = 0
        if consecutive > 0:
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
    print("Positive:",len(example_pos))
    return (example_pos, example_neg)

use_micro = False
use_macro = False
use_autoenc = False
name_attack = None

i = 1
while i < len(sys.argv):
    if sys.argv[i] == "-a":
        i += 1
        if not name_attack:
            name_attack = []
        name_attack.append(sys.argv[i])
    elif sys.argv[i] == "-autoenc":
        use_autoenc = True
    elif sys.argv[i] == "-micro":
        use_micro = True
    elif sys.argv[i] == "-macro":
        use_macro = True
    else:
        print("Erreur:",sys.argv[i])
        exit()
    i += 1

if not use_autoenc and not use_micro and not use_macro:
    print("Aucun détecteur ! Utilisez -micro, -macro ou -autoenc")
    exit()

# lecture config

config = Config()

with open("test_folders") as f:
    folders = f.readlines()
directories = [x.strip() for x in folders]

attack = np.loadtxt(os.path.join(config.get_config("section"), "logattack"), dtype='<U13')

# TODO : on ne garde les attaques que du 23 janvier
# attack = np.array([a for a in attack if datetime.datetime.fromtimestamp(int(a[1])/1000).day == 23])
print(attack)
identifiers = np.unique(attack[:,0])
print("Attacks list:",identifiers)

evaluators = [Evaluator(attack, i) for i in identifiers]
# evaluators = []
if not name_attack:
    evaluators.append(Evaluator(attack))
else:
    for n in name_attack:
        evaluators.append(Evaluator(attack, n))
nb_features = sum(config.get_config_eval("features_number"))
nb_features_macro = config.get_config_eval("nb_features_macro")
prefix = config.get_config("section")
# threshold_autoencoder = config.get_config_eval("threshold_autoencoder")
threshold_autoencoder = [1,0.11,1]
threshold_macro = config.get_config_eval("threshold_macro")
threshold_micro = config.get_config_eval("threshold_micro")
# chargement du jeu de données de test micro

show_time = True
show_hist = False
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
        m = CNN(j)
        extractors.load_model(m)

with open("train_folders") as f:
    folders_test = f.readlines()
folders_test = [x.strip() for x in folders]


# évaluation

path_examples = os.path.join(prefix, "results-OCSVM-micro.joblib")
path_examples_macro = os.path.join(prefix, "results-HMM-macro.joblib")
path_examples_extractors = os.path.join(prefix, config.get_config("autoenc_filename")+"-results-autoenc.joblib")

if use_micro:
    print("Prediction for micro…")
    scores_micro = scores_micro_macro(models, path_examples, data)
    (example_pos, example_neg) = predict(models, scores_micro, threshold_micro)
if use_macro:
    print("Prediction for macro…")
    scores_macro = scores_micro_macro(models_macro, path_examples_macro, data_macro)
    (example_pos_macro, example_neg_macro) = predict(models_macro, scores_ācro, threshold_macro)
if use_autoenc:
    print("Prediction for autoencoders…")
    scores_ex = score_extractors(extractors, path_examples_extractors, folders_test)
    # scores_ex = get_derivative(scores_ex)
    (example_pos_extractors, example_neg_extractors) = predict_extractors(extractors, scores_ex, threshold_autoencoder)

for e in evaluators:
    print("***",e._id)
    if use_micro:
        print("Results micro: ",end='')
        e.evaluate(example_pos, scores_micro, models,"micro")
    if use_macro:
        print("Results macro: ",end='')
        e.evaluate(example_pos_macro, scores_macro, models_macro,"macro")
#    print("Results micro and macro")
#    e.evaluate(list(set(example_pos+example_pos_macro)),
#               list(set(example_neg+example_neg_macro)))
    if use_autoenc:
        print("Results autoencoders: ",end='')
        e.evaluate(example_pos_extractors, scores_ex, extractors,"autoenc")
