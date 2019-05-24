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

class DetectorState(Enum):
    NOT_DETECTING = 0
    DETECTING = 1
    TRIGGERED = 2
    RESTING = 3

def load_attacks():
    attack = np.loadtxt("logattack", dtype='<U13')

# TODO : pour ne garder que les attaques d'un certain jour
    attack = np.array([a for a in attack if datetime.datetime.fromtimestamp(int(a[2])/1000).day != 4])
# il y a des attaques en trop le 4 avril
    print(attack)
    identifiers = np.unique(attack[:,0])

    colors = {}
    all_colors = {'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'fuchsia', 'grey', 'chocolate', 'lawngreen', 'salmon', 'indianred', 'turquoise', 'royalblue', 'lime', 'teal', 'orange'}
    for i in identifiers:
        colors[i] = all_colors.pop()
        print(i,"is",colors[i])

    print("Attacks list:",identifiers)

    attack_plot = {"scan433": 0, "strong433": 0, "scan868": 1, "dosHackRF45": 0, "dosHackRF89": 1, "strong868": 1, "tvDOS": 0, "bruijnSequenc": 0}
    attack_freq_type = {"scan433": [432,434], "strong433": [432,434], "scan868":[867,869], "strong868":[867,869], "tvDOS":[485,499], "bruijnSequenc":[432,434]}

# we add the plot number of the attack
    attack_tmp = []
    attack_freq = {}
    for a in attack:
        plot_nb = attack_plot.get(a[0])
        if plot_nb == None: # par défaut: 2400-2500
            plot_nb = 2
        attack_tmp.append([a[0], a[1], a[2], plot_nb])
        f = attack_freq_type.get(a[0])
        if f == None:
            f = [2540,2560] # TODO
        attack_freq[(int(a[1]),int(a[2]))] = f
    attack = np.array(attack_tmp)

    # if name_attack:
    #     a = []
    #     for n in name_attack:
    #         a.append(attack[attack[:,0] == n])
    #     attack = np.concatenate((a))

    return colors, attack, attack_freq

def merge(dico):
    """
    merge simplement les intersections
    """
    l = list(dico.keys())
    out = {}
    merged = []
    for i in range(len(l)):
        if i not in merged:
            current = l[i]
            merged_number = 1
            cumulated_freq = dico[current][0]
            number = dico[current][1]
            for j in range(i+1,len(l)):
                # on vérifie qu'il s'agit bien de la même band
                if number == dico[l[j]][1] and intersect(current,l[j]):
                    merged_number += 1
                    cumulated_freq += dico[l[j]][0]
                    merged.append(j)
                    current = (min(current[0], l[j][0]), max(current[1], l[j][1]))
            out[current] = [cumulated_freq / merged_number, number]
    return out

def merge_all(number, bestProbe, probe1, probe2):
    """
    merge les attaques des 3 sondes pour la bande "number"
    si 1 sonde voit une attaque sur la bande sur laquelle elle est sensible on la valide
    sinon, il faut au moins deux sondes pour valider l'attaque
    équation: attaques = bestProbe union (probe1 inter probe2)
    """
    out = {k:v for (k,v) in bestProbe.items() if v[1] == number}
    # print(len(out))
    for i in probe1:
        if probe1[i][1] == number:
            for j in probe2:
                if probe2[j][1] == number:
                    interval = intersect(i,j)
                    if interval:
                        # print("Add:",interval)
                        out[interval] = [(probe1[i][0] + probe2[j][0]) / 2, number]
    # print(len(out))
    # out = merge(out)
    # print(len(out))
    return out



def get_derivative(scores):
    out = {}
    keys = sorted(list(scores.keys()))
    for i in range(len(scores)-1):
        score_old = scores[keys[i]]
        score_new = scores[keys[i+1]]
        if isinstance(score_old, dict):
            tmp = {}
            for k in score_old:
                tmp[k] = score_new.get(k) - score_old.get(k)
            out[keys[i+1]] = tmp
        else:
            assert False
            out[keys[i+1]] = score_new - score_old
    return out

def passe_haut(scores):
    alpha = 0.999
    out = {}
    keys = sorted(list(scores.keys()))
    for i in range(len(scores)):
        out[keys[i]] = {}

    out[keys[0]] = scores[keys[0]]

    # for each model
    for k in scores[keys[0]]:
        y = scores[keys[0]][k]
        out[keys[0]][k] = y
        for i in range(1, len(keys)):
            x_old = scores[keys[i-1]][k]
            x_new = scores[keys[i]][k]
            y = alpha * (y + x_new - x_old)
            out[keys[i]][k] = y

    return out

def intersect(interval1, interval2):
    start = max(interval1[0], interval2[0])
    end = min(interval1[1], interval2[1])
    if start < end:
        return (start, end)
    return None

def moyenne_glissante(scores):
    out = {}
    keys = sorted(list(scores.keys()))
    print(keys)
    # keys : timestamps
    n = 1000
    m = {}
    number = {}
    for k in scores[keys[0]]:
        # k : model number
        s = 0
        number[k] = int(n/2)
        for j in range(0,n/2):
            s += scores[keys[j]].get(k)
        m[k] = s

    for i in range(n/2,len(scores)-1):
        # if i % 1000 == 0:
            # print(i)
        tmp = {}
        for k in scores[keys[i]]:
            m[k] += scores[keys[i]].get(k)
            m[k] -= scores[keys[i-n]].get(k)
            tmp[k] = m[k]
        out[keys[i]] = tmp
    return out

class Evaluator:

    """
        Recall (= true positive rate): attaques correctement trouvées / toutes les vraies attaques. Proportion d'attaques qui ont été détectées (si on prédit toujours une attaque, le recall vaut 1)
        Precision: nb attaques correctement trouvées / nb attaques détectées. Proportion de détection pertinentes (si on prédit toujours une attaque, la précision sera proche de 0)
        F-score = 2 * (precision * rappel) / (precision + rappel)
    """

    def __init__(self, all_attack, attack_freq, identifier=None):
        """
            format all_attack : shape (-1,3)
            column 0 : attack start timestamp
            column 1 : attack end itemstamp
            column 2 : attack identifier
        """
        # self._id = identifier
        # if self._id == None:
        self._id = "All"
        # print(all_attack)
        self.attack_freq = attack_freq
        self._all_attack = all_attack
        self._attack = np.array(all_attack[:,1:].astype(np.integer))
        # else:
            # self._attack = np.array(all_attack[all_attack[:,0] == identifier][:,1:].astype(np.integer))

        # if self._id == "All":
        #     f = open("logattack_fixed","w+")
        #     for i in range(len(self._attack)):
        #         a = self._attack[i]
        #         f.write(all_attack[i, 2]+" "+str(a[0])+" "+str(a[1])+"\r\n")
        #     f.close()
        self._seen_attack = []
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
                return True
        return False

    def roc(self, detected_positive_dico, scores, models, typestr):
        pass # TODO

    def evaluate_freq(self, detected_freq):

        identifiers = np.unique(self._all_attack[:,0])
        for ident in identifiers:
            errors = []
            ok = 0
            nb = 0
            attack_ident = self._all_attack[self._all_attack[:,0] == ident]
            # print(attack_ident)
            for [_, a1, a2, a_band] in attack_ident:
                a1 = int(a1)
                a2 = int(a2)
                a_band = int(a_band)
                # print(a1,a2,a_band)
                for d in self.true_positive_dates:
                    # we check only the true positive detection
                    (f,d_band) = detected_freq[d]
                    # print(d,f,d_band)
                    if a_band == d_band and intersect((a1,a2),d) != None:
                        (f1,f2) = self.attack_freq[(a1,a2)]
                        if f >= f1 and f <= f2:
                            ok += 1
                        errors.append(abs(f - (f1+f2)/2))
                        nb += 1
            print("    Results for",ident)
            if nb > 0:
                # print(ok, nb)
                print("Proportion within 1MHz:",ok/nb)
                print("Mean error:",np.mean(errors))
                # print(errors)
                # print(np.percentile(errors, 50))
                # print(np.percentile(errors, 80))
                # print(np.percentile(errors, 90))
                print("95% of errors below:", np.percentile(errors, 95))
                # print(np.percentile(errors, 99))
            else:
                print("No attack detected")

    def evaluate(self, detected_positive_dico):
        self._seen_attack = []
        detected_positive = np.array([k for (k,_) in detected_positive_dico])
        total_positives = len(detected_positive)
        true_positive_list = detected_positive[list(map(self.is_in_attack_detected, detected_positive))]
        false_positive_list = detected_positive[[t not in true_positive_list for t in detected_positive]]
        true_positive = len(true_positive_list)
        false_positive = total_positives - true_positive
        self.true_positive_dates = [(a,b) for (a,b) in detected_positive_dico if a in true_positive_list]
        # print(len(true_positive_list))
        # print(len(self.true_positive_dico))
        recall = {}
        print("Detected : ",len(self._seen_attack),"/",len(self._attack))

        # true_positive_score = np.array([detected_positive_dico[t] for t in true_positive_list])
        # false_positive_score = np.array([detected_positive_dico[t] for t in false_positive_list])

        new_method = True
        if new_method:
            l_i = 0
            l_a = 0

            identifiers = np.unique(self._all_attack[:,0])
            for ident in identifiers:
                attack_id = self._all_attack[self._all_attack[:,0] == ident]
                # attack_id = np.array(attack_id[:,1:].astype(np.integer))
                # attack_id = [(a,b) for [a,b,_] in attack_id]
                atk_number = len(attack_id)
                atk_detected = 0
                l_i_tmp = 0
                l_a_tmp = 0
                # for (a1,a2) in attack_id:
                for [_, a1, a2, a_band] in attack_id:
                    a1 = int(a1)
                    a2 = int(a2)
                    a_band = int(a_band)

                    l_a_tmp += a2 - a1

                    detected = False
                    for (d1,d2) in detected_positive_dico:
                        d_band = detected_positive_dico[(d1,d2)][0]
                        if a_band == d_band:
                            i = intersect((d1,d2), (a1,a2))
                            if i:
                                detected = True
                                l_i_tmp += i[1] - i[0]
                    if detected:
                        atk_detected += 1
                l_i += l_i_tmp
                l_a += l_a_tmp

                print("Recall for",ident,":",l_i_tmp / l_a_tmp)
                recall[ident] = l_i_tmp / l_a_tmp


                print("Detected for",ident,":",atk_detected,"/",atk_number,"=",atk_detected/atk_number)

            l_d = 0
            for (d1,d2) in detected_positive_dico:
                l_d += d2 - d1

            p = l_i / l_d
            r = l_i / l_a
            f = 2*p*r/(p+r)
            print("Precision",p,"Recall",r,"f-measure",f)

            precision = p

        else:
            if false_positive + len(self._seen_attack) == 0:
                precision = 1
            else:
                # precision = len(self._seen_attack) / (false_positive + len(self._seen_attack))
                precision = len(self._seen_attack) / (false_positive + len(self._seen_attack))

            identifiers = np.unique(self._all_attack[:,0])
            for ident in identifiers:
                attack_id = self._all_attack[self._all_attack[:,0] == ident]
                r = 0
                for i in attack_id:
                    if int(i[1]) in self._seen_attack:
                        r += 1
                r = r / len(attack_id)
                print("Recall for "+ident+": "+str(r))
                recall[ident] = r

        fmeasure = {}
        for ident in identifiers:
            if precision + recall[ident] != 0:
                fmeasure[ident] = 2*(precision * recall[ident]) / (precision + recall[ident])
            else:
                fmeasure[ident] = float('nan')
            print("f-measure for "+ident+": "+str(fmeasure[ident]))


        print("Overall precision:",precision)
        print("tp",true_positive, "fp",false_positive,"precision",precision,"recall",recall,"f-measure",fmeasure)

        print("Mean f-measure:", sum(fmeasure.values()) / len(fmeasure))

        # if show_hist:
        #     plt.hist(true_positive_score, color='red', bins=100, histtype='step', log=True)
        #     # plt.hist(false_negative_score, color='magenta', bins=100, histtype='step', log=True)
        #     plt.hist(false_positive_score, color='cyan', bins=100, histtype='step', log=True)
        #     # plt.hist(true_negative_score, color='blue', bins=100, histtype='step', log=True)
        #     plt.title(self._id)
        #     plt.show()
    def print_score(self, detected_positive_dico, scores, models, typestr, thr=None, colors=None):
        # print(thr)
        x = sorted(list(scores.keys()))
        threshold = []
        for i in range(3):
            t = {}
            for d in x:
                for p in thr:
                    if p(d):
                        t[d] = thr[p][i]
                        break
            threshold.append(t)

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
        i = 0
        for row in ax:
            for col in row:
                col.xaxis.set_major_formatter(hfmt)
                plt.setp(col.get_xticklabels(), rotation=15)
                # color = {multimodels.period_weekend_and_night: "green", multimodels.period_day_not_weekend: "blue", multimodels.period_always: "magenta"}
                # for p in thr:
                    # col.axhline(thr[p][i],0,1,color=color[p])
                # i += 1

        for a in self._attack:
            l = self._all_attack[self._all_attack[:,1] == str(a[0])]
            ident = l[0,0]
            nb = 0
            for row in ax:
                for col in row:
                    if a[2] == nb:
                        col.hlines(0,
                        mdates.date2num(datetime.datetime.fromtimestamp(a[0]/1000)),
                        mdates.date2num(datetime.datetime.fromtimestamp(a[1]/1000)),
                        color=colors[ident])
                    nb += 1

        for (t1, t2) in detected_positive_dico:
            i = 0
            for row in ax:
                for col in row:
                    if i in detected_positive_dico[(t1,t2)]:
                        col.hlines(-0.0001,
                        mdates.date2num(datetime.datetime.fromtimestamp(t1/1000)),
                        mdates.date2num(datetime.datetime.fromtimestamp(t2/1000)),
                        color='green')
                    i += 1



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
                        val = [scores[k].get(i) for k in x]
                        valTh = [threshold[i][k] for k in x]
                        x, val = zip(*sorted(zip(x, val)))
                        x_dates = mdates.date2num([datetime.datetime.fromtimestamp(t/1000) for t in x])
                        col.plot(x_dates, val, alpha=0.7, label=labels[i])
                        col.plot(x_dates, valTh, alpha=0.7, color="magenta")
                        i += 1
                        col.legend(loc='upper left')

        else:
            val = list(scores.values())
            x, val = zip(*sorted(zip(x, val)))
            plt.plot(x, val)
        plt.show()

def score_extractors(extractors, path_examples, folders_test):
    scores = {}
    start = time.time()
    i = 0
    # print(os.listdir(folders_test[0]))
    paths = [os.path.join(directory,f) for directory in folders_test for f in sorted(os.listdir(directory))]
    for fname in paths:
        timestamp = int(os.path.split(fname)[1])
        data = read_file(fname, quant=True)
        if i % 100 == 0:
            print(i,"/",len(paths))
        i += 1
        scores = {**scores, **extractors.get_score(data, timestamp)}
    end = time.time()
    print("Scoring time:",(end-start),"s")
    joblib.dump(scores, path_examples)
    return scores

def predict_extractors_cumul(models, scores, all_t):
    start = time.time()
    example_pos = {}
    timestamps = sorted(scores.keys())

    for (_,m) in models:
        state = DetectorState.NOT_DETECTING
        cumulated_threshold = 0.7
        resting_duration = 0
        cumul = 0
        # consecutive = 0
        previous_timestamp = None
        discontinuity_timestamp = None
        for timestamp in timestamps:
            found = False
            for p in all_t:
                if p(timestamp):
                    found = True
                    threshold_autoencoder = all_t[p]
                    low_threshold_autoencoder = threshold_autoencoder
                    # low_threshold_autoencoder = [0.8 * t for t in threshold_autoencoder]
            assert found, multimodels.process_unix_time(timestamp)
            score = scores[timestamp].get(m._number)


            if state == DetectorState.NOT_DETECTING and m.predict_thr(score,threshold=threshold_autoencoder[m._number]):
            # if state == DetectorState.NOT_DETECTING and extractors.predict_thr(score,optimistic=False,threshold=threshold_autoencoder):
                state = DetectorState.DETECTING
                cumul = 0
                previous_timestamp = timestamp

            elif (state == DetectorState.DETECTING or state == DetectorState.TRIGGERED) and not m.predict_thr(score,threshold=low_threshold_autoencoder[m._number]):
                if state == DetectorState.TRIGGERED:
                    # End of the attack
                    if timestamp - discontinuity_timestamp > 3600000: # discontinuité dans les données
                        timestamp = discontinuity_timestamp
                    example_pos[(previous_timestamp, timestamp)] = [m._number]
                previous_timestamp = timestamp
                state = DetectorState.RESTING

            elif state == DetectorState.DETECTING:
                if cumul > cumulated_threshold:
                    state = DetectorState.TRIGGERED

            elif state == DetectorState.RESTING:
                # if extractors.predict_thr(score,optimistic=False,threshold=threshold_autoencoder):
                if m.predict_thr(score,threshold=low_threshold_autoencoder[m._number]):
                    # consecutive = 0
                    previous_timestamp = timestamp
                # else:
                    # consecutive += 1
                if timestamp - previous_timestamp > resting_duration:
                # if consecutive > resting_duration:
                    state = DetectorState.NOT_DETECTING
                    # consecutive = 0

            if state == DetectorState.DETECTING:
                cumul += abs(score - threshold_autoencoder[m._number])
                if timestamp - previous_timestamp > 3600000: # discontinuité dans les données
                    state = DetectorState.NOT_DETECTING

            discontinuity_timestamp = timestamp

    end = time.time()
    print("Detection time:",(end-start),"s")
    print("Positive:",len(example_pos))
    return example_pos

def predict_extractors(models, scores, all_t):
    start = time.time()
    example_pos = {}
    timestamps = sorted(scores.keys())

    for (_,m) in models:
        state = DetectorState.NOT_DETECTING
        detection_duration = 20000
        resting_duration = 0
        # consecutive = 0
        previous_timestamp = None
        discontinuity_timestamp = None
        for timestamp in timestamps:
            found = False
            for p in all_t:
                if p(timestamp):
                    found = True
                    threshold_autoencoder = all_t[p]
                    low_threshold_autoencoder = threshold_autoencoder
                    # low_threshold_autoencoder = [0.8 * t for t in threshold_autoencoder]
            assert found, multimodels.process_unix_time(timestamp)
            score = scores[timestamp].get(m._number)


            if state == DetectorState.NOT_DETECTING and m.predict_thr(score,threshold=threshold_autoencoder[m._number]):
            # if state == DetectorState.NOT_DETECTING and extractors.predict_thr(score,optimistic=False,threshold=threshold_autoencoder):
                state = DetectorState.DETECTING
                previous_timestamp = timestamp

            elif (state == DetectorState.DETECTING or state == DetectorState.TRIGGERED) and not m.predict_thr(score,threshold=low_threshold_autoencoder[m._number]):
                if state == DetectorState.TRIGGERED:
                    # End of the attack
                    if timestamp - discontinuity_timestamp > 3600000: # discontinuité dans les données
                        timestamp = discontinuity_timestamp
                    example_pos[(previous_timestamp, timestamp)] = [m._number]
                previous_timestamp = timestamp
                state = DetectorState.RESTING

            elif state == DetectorState.DETECTING:
                # consecutive += 1
                # if consecutive > detection_duration:

                if timestamp - previous_timestamp > 3600000: # discontinuité dans les données
                    state = DetectorState.NOT_DETECTING

                elif timestamp - previous_timestamp > detection_duration:
                    # attack detected !
                    # example_pos[timestamp] = extractors.get_predictor(score,optimistic=False,threshold=threshold_autoencoder)
                    # previous = example_pos.get(timestamp)
                    # if previous == None:
                    # else:
                        # example_pos[timestamp].append(m._number)
                    # consecutive = 0
                    state = DetectorState.TRIGGERED

            elif state == DetectorState.RESTING:
                # if extractors.predict_thr(score,optimistic=False,threshold=threshold_autoencoder):
                if m.predict_thr(score,threshold=low_threshold_autoencoder[m._number]):
                    # consecutive = 0
                    previous_timestamp = timestamp
                # else:
                    # consecutive += 1
                if timestamp - previous_timestamp > resting_duration:
                # if consecutive > resting_duration:
                    state = DetectorState.NOT_DETECTING
                    # consecutive = 0
            discontinuity_timestamp = timestamp

    end = time.time()
    print("Detection time:",(end-start),"s")
    print("Positive:",len(example_pos))
    return example_pos

def get_snr(example_pos, folders_list, median):
    snr = {}
    for (t1, t2) in example_pos:
        l = []
        freq = frequency_to_index(example_pos[(t1,t2)][0])
        nb = example_pos[(t1,t2)][1]
        for folders in folders_list:
            m = median[folders][nb]
            w = read_files_from_timestamp(t1, t2, folders_list[folders])[:,freq-2,freq+2]
            l.append((np.mean(w)-m, np.max(w)-m, np.std(w)))
        l.append(nb)
        snr[(t1,t2)]=l
    return snr

def predict_frequencies(example_pos, folders, extractors):
    frequencies = {}
    for (t1, t2) in example_pos:
        nb = example_pos[(t1,t2)][0]
        # print(nb)
        waterfalls = read_files_from_timestamp(t1, t2, folders)
        # print(waterfalls.shape)
        # print(nb)
        # print("S",waterfalls.shape)
        waterfalls[:,0:nb*1000] = 0
        waterfalls[:,(nb+1)*1000:3000] = 0
        # print("S après",waterfalls.shape)
        (weights, data) = extractors.get_frequencies(waterfalls, number=nb)
        # print(data, weights)
        if len(data) >= 2:
            median = weighted_median(data, weights)
            # print("Median:",median)
            median += nb*1000
        elif len(data) == 1:
            median = data[0] + nb*1000
        else:
            median = nb*1000+500 # absence of signal
            # print(f)
        f = index_to_frequency(median)
        frequencies[(t1, t2)] = (f,nb)

    return frequencies

def load_scores(path_examples_extractors, extractors, bands):
    try:
        print("Loading scores…")
        # chargement des prédictions si possible
#        (example_pos, example_neg) = joblib.load(path_examples)
        s = joblib.load(path_examples_extractors)
        print("Scores loaded")
        return s
    except Exception as e:
        print("Scores not found:",e)
        print("Prediction for autoencoders…")
        # extractors = MultiExtractors()
        for j in range(len(bands)):
            (i,s) = bands[j]
            m = CNN(j)
            extractors.load_model(m)
        return score_extractors(extractors, path_examples_extractors, directories)

