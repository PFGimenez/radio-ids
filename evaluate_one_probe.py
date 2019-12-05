#!/usr/bin/env python3
import sys
from models import MultiModels, MultiExtractors
import multimodels
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
import random
from enum import Enum
import evaluate

# use_micro = False
# use_macro = False
use_autoenc = False
use_autoenc_macro = False
name_attack = None
train = False
mini = False
predict_freq = False
use_cumul = False
show_time = True # TODO
show_hist = False
ROC = None
i = 1
while i < len(sys.argv):
    if sys.argv[i] == "-a":
        i += 1
        if not name_attack:
            name_attack = []
        name_attack.append(sys.argv[i])
    elif sys.argv[i] == "-autoenc":
        use_autoenc = True
    elif sys.argv[i] == "-no-time":
        show_time = False
    elif sys.argv[i] == "-autoenc-macro":
        use_autoenc_macro = True
    elif sys.argv[i] == "-cumul":
        use_cumul = True
    # elif sys.argv[i] == "-micro":
        # use_micro = True
    # elif sys.argv[i] == "-macro":
        # use_macro = True
    elif sys.argv[i] == "-train":
        train = True
    elif sys.argv[i] == "-ROC":
        ROC = True
    elif sys.argv[i] == "-mini":
        mini = True
    elif sys.argv[i] == "-freq":
        predict_freq = True
    else:
        print("Erreur:",sys.argv[i])
        exit()
    i += 1

if not use_autoenc and not use_autoenc_macro:
    print("Aucun détecteur ! Utilisez -autoenc ou -autoenc-macro")
    exit()

# lecture config

config = Config()
prefix_result_train = "train-"
if train:
    prefix_result = "train-"
    folder_file = config.get_config("train_folders")
elif mini:
    prefix_result = "mini-"
    folder_file = "mini_folders"
else:
    prefix_result = ""
    folder_file = config.get_config("test_folders")
with open(folder_file) as f:
    folders = f.readlines()

directories = [x.strip() for x in folders]

colors,attack,attack_freq = evaluate.load_attacks()




print(attack)
scores_ex = None
# evaluators = [Evaluator(attack, i) for i in identifiers]
evaluators = []
# if not name_attack:
# evaluators = [Evaluator(attack, i) for i in identifiers]
evaluators.append(evaluate.Evaluator(attack, attack_freq))
# else:
    # for n in name_attack:
        # evaluators.append(Evaluator(attack, n))
# nb_features = sum(config.get_config_eval("features_number"))
# nb_features_macro = config.get_config_eval("nb_features_macro")
prefix = config.get_config("section")
threshold_autoencoder_number = config.get_config_eval("threshold_autoencoder")
# threshold_macro = config.get_config_eval("threshold_macro")
# threshold_micro = config.get_config_eval("threshold_micro")
# chargement du jeu de données de test micro

# modèle micro

# if use_micro:
#     models = MultiModels()
#     try:
#         models.load(os.path.join(prefix, "micro-OCSVM.joblib"))
#         files = [os.path.join(prefix, "features-"+d.split("/")[-1]) for d in directories]
#         print(files)
#         data = np.concatenate([np.fromfile(f).reshape(-1, nb_features + 1) for f in files])
#         print("data micro:",data.shape)
#     except Exception as e:
#         print("Loading failed:",e)
#         use_micro = False

# # modèle macro
# if use_macro:
#     models_macro = MultiModels()
#     try:
#         models_macro.load(os.path.join(prefix, "macro-HMM.joblib"))
#         test_macro_filename = os.path.join(config.get_config("section"), "test_"+config.get_config("macro_features_stage_2"))
#         data_macro = np.fromfile(test_macro_filename).reshape(-1, nb_features_macro + 1)
#         print("data macro:",data_macro.shape)
#     except Exception as e:
#         print("Loading failed:",e)
#         use_macro = False

# autoencoders
bands = config.get_config_eval('waterfall_frequency_bands')
dims = config.get_config_eval('autoenc_dimensions')
# if ROC:
    # cumulated_threshold = [ROC,ROC,ROC]
    # print("ROC value: ",ROC)
# else:
cumulated_threshold = config.get_config_eval('cumul_threshold')
extractors = MultiExtractors()
for j in range(len(bands)):
    (i,s) = bands[j]
    m = CNN(j)
    # if predict_freq:
    extractors.load_model(m)
    # else:
        # extractors.add_model(m) # dummy is enough for most cases
extractors.set_dummy(False)

# évaluation
if use_cumul:
    path_detection_intervals = os.path.join(prefix, prefix_result+"results-detection-intervals-cumul"+str(cumulated_threshold)+".joblib")
    path_frequencies = os.path.join(prefix, prefix_result+"results-frequencies-cumul"+str(cumulated_threshold)+".joblib")
else:
    path_detection_intervals = os.path.join(prefix, prefix_result+"results-detection-intervals.joblib")
    path_frequencies = os.path.join(prefix, prefix_result+"results-frequencies.joblib")

# les scores
path_examples_extractors = os.path.join(prefix, prefix_result+config.get_config("autoenc_filename")+"-results-autoenc.joblib")
path_examples_train_extractors = os.path.join(prefix, prefix_result_train+config.get_config("autoenc_filename")+"-results-autoenc.joblib")

# if use_micro:
#     print("Prediction for micro…")
#     scores_micro = scores_micro_macro(models, path_examples, data)
#     (example_pos, example_neg) = predict(models, scores_micro, threshold_micro)
# if use_macro:
#     print("Prediction for macro…")
#     scores_macro = scores_micro_macro(models_macro, path_examples_macro, data_macro)
#     (example_pos_macro, example_neg_macro) = predict(models_macro, scores_macro, threshold_macro)

do_th = False
if show_time or ROC: # pour avoir un affichage correct
    do_th = True
if use_autoenc:

    threshold_autoencoder = {}
    periods = [multimodels.period_weekend_and_night, multimodels.period_day_not_weekend]
    # if not train: # décommenter pour un nouveau modèle
    if True:
        try:
            print(path_examples_train_extractors)
            scores_train = joblib.load(path_examples_train_extractors)
            if do_th:
                for p in periods:
                    print("Period",p.__name__)
                    thr = extractors.learn_threshold_from_scores(scores_train, period=p)
                    t = []
                    for l in thr:
                        # check the order of the keys
                        assert l == len(t)
                        t.append(thr.get(l)[threshold_autoencoder_number]+0.000001) # TODO
                    threshold_autoencoder[p] = t
                    # dictionnaire : {period : [seuil 400-500, seuil 800-900, seuil 2.4-2.5]}
            # cumulated_threshold = evaluate.get_cumul_threshold(extractors._models, scores_train, threshold_autoencoder)
            if ROC is None:
                cumulated_threshold=[1.3295,1.3295,1.3295]
        except Exception as e:
            print("Exception",e)
            print("No train score loaded")

            # threshold_autoencoder = {multimodels.period_always: [0.010, 0.008, 0.03]}
    if threshold_autoencoder == {}:
        threshold_autoencoder = {multimodels.period_always: [0.010, 0.008, 0.03]}
        print("Autoencoder thresholds:", threshold_autoencoder)
    # threshold_autoencoder = [0.010, 0.008, 0.03]
    # scores_ex = get_derivative(scores_ex)
    # scores_ex = moyenne_glissante(scores_ex)
    # scores_ex = passe_haut(scores_ex)

    try :
        # print(path_detection_intervals)
        example_pos_extractors = joblib.load(path_detection_intervals)
#        if ROC:
#            raise Exception
#        z=0
#        z=1/z
    except:
        if scores_ex == None:
            scores_ex = evaluate.load_scores(path_examples_extractors, extractors, bands, directories)
        if use_cumul:
            # print(cumulated_threshold)
            example_pos_extractors = evaluate.predict_extractors_cumul(extractors._models, scores_ex, threshold_autoencoder, cumulated_threshold)
        else:
            example_pos_extractors = evaluate.predict_extractors(extractors._models, scores_ex, threshold_autoencoder)
        joblib.dump(example_pos_extractors, path_detection_intervals)

    if predict_freq:
        print("Frequency localization")
        try:
            detected_freq = joblib.load(path_frequencies)
            # print(detected_freq)
        except:
            detected_freq = evaluate.predict_frequencies(example_pos_extractors, directories, extractors)
            joblib.dump(detected_freq, path_frequencies)

for e in evaluators:
    # print("***",e._id)
    # if use_micro:
    #     print("Results micro: ",end='')
    #     e.evaluate(example_pos, scores_micro, models,"micro", colors)
    # if use_macro:
    #     print("Results macro: ",end='')
    #     e.evaluate(example_pos_macro, scores_macro, models_macro,"macro", colors)
#    print("Results micro and macro")
#    e.evaluate(list(set(example_pos+example_pos_macro)),
#               list(set(example_neg+example_neg_macro)))
    if use_autoenc:
        # print("Results autoencoders: ",end='')
        if ROC:
            all_tpr = [[],[],[]]
            all_fpr = [[],[],[]]
            scores_ex = evaluate.load_scores(path_examples_extractors, extractors, bands, directories)
            for roc_val in [0,0.00001,0.001,0.01,0.1,1,10,100]:
            #[0,0.00001,0.001,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0, 5,10,30,100,1000,10000,100000,1000000]:
                cumulated_threshold=[roc_val,roc_val,roc_val]
                print("ROC:",roc_val)
                print("Seuil:",threshold_autoencoder)
                example_pos_extractors = evaluate.predict_extractors_cumul(extractors._models, scores_ex, threshold_autoencoder, cumulated_threshold)
                out_roc = e.evaluate(example_pos_extractors)
                for i in range(3):
                    all_fpr[i].append(out_roc[i][0])
                    all_tpr[i].append(out_roc[i][1])
                print((all_fpr,all_tpr))
        else:
                e.evaluate(example_pos_extractors)
        if predict_freq and not train:
            e.evaluate_freq(detected_freq)
        if show_time:
            if scores_ex == None:
                scores_ex = evaluate.load_scores(path_examples_extractors, extractors, bands, directories)
            e.print_score(example_pos_extractors, scores_ex, extractors,"autoenc", threshold_autoencoder, colors)
