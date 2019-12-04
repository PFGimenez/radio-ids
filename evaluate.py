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
    attack = np.loadtxt("logattack", dtype='<U13', usecols=(0,1,2))

# TODO : pour ne garder que les attaques d'un certain jour
    attack = np.array([a for a in attack if datetime.datetime.fromtimestamp(int(a[2])/1000).day != 4])
# il y a des attaques en trop le 4 avril
    print(attack)
    identifiers = np.unique(attack[:,0])

    colors = {}
    all_colors = {'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'fuchsia', 'grey', 'chocolate', 'lawngreen', 'salmon', 'indianred', 'turquoise', 'royalblue', 'lime', 'teal', 'orange', 'ivory', 'olive', 'pink'}
    for i in identifiers:
        colors[i] = all_colors.pop()
        print(i,"is",colors[i])

    print("Attacks list:",identifiers)

    attack_plot = {"scan433": 0, "anomaly467":0, "strong433": 0, "scan868": 1, "dosHackRF45": 0, "dosHackRF89": 1, "strong868": 1, "tvDOS": 0, "bruijnSequenc": 0, "old-scan433": 0, "old-strong433": 0, "anomaly": 0, "bruijnHarmo": 1}
    attack_freq_type = {"scan433": [433,433], "strong433": [433,433], "scan868":[868,868], "strong868":[868,868], "tvDOS":[485,499], "bruijnSequenc":[433.8,433.8], "bleScan": [2400,2480], "btScan": [2400,2480], "dosHackRF45": [400,500], "dosHackRF89": [800, 900], "floodZigbee": [2475,2485], "injectESB": [2400, 2500], "old-scan433": [433,433], "old-strong433": [433,433], "anomaly467": [467,467], "wifiDeauth": [2451,2473], "wifiRogueAP": [2461,2483], "wifiScan": [2400,2500], "zigbeeScan": [2400,2480], "anomaly":[462,462],
                        "bruijnHarmo": [867.7,867.7]}

    for (n,d1,d2) in attack:
        if int(d1) >= int(d2):
            print("ERREUR!!! Date à l'envers!",n,d1,d2)
        for(n2,d3,d4) in attack:
            if d3 != d1 and attack_plot.get(n)==attack_plot.get(n2) :
                 if intersect((int(d1),int(d2)),(int(d3),int(d4))) is not None:
                    print("ERREUR!!! Intersection attaque!",n,d1,d2,n2,d3,d4)



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
            # merged_number = (l[i][1]-l[i][0])
            # cumulated_freq = dico[current][0]*(l[i][1]-l[i][0])
            number = dico[current][1]
            for j in range(i+1,len(l)):
                # on vérifie qu'il s'agit bien de la même band
                if number == dico[l[j]][1] and intersect(current,l[j]):
                    # merged_number += (l[j][1]-l[j][0])
                    merged_number += 1
                    # cumulated_freq += dico[l[j]][0]*(l[j][1]-l[j][0])
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
        output={}
        identifiers = np.unique(self._all_attack[:,0])
        all_error = []
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
                for d in detected_freq:
                    (t1,t2)=d
                    i = intersect((a1,a2),d)
                # for d in self.true_positive_dates:
                    # we check only the true positive detection
                    (f,d_band) = detected_freq[d]
                    # print(d,f,d_band)
                    # if a_band == d_band and intersect((a1,a2),d) != None:
                    # if ident=="scan433":
                        # print(f,f1,f2,a1,a2,t1,t2)
                    if a_band == d_band and i is not None and (i[1] - i[0]) / (t2-t1) > 0.5 and (i[1] - i[0]) / (a2 - a1) > 0.5:
                        (f1,f2) = self.attack_freq[(a1,a2)]
                        if f >= f1 and f <= f2:
                            ok += 1
                            e = 0
                        else:
                            e = min(abs(f1-f),abs(f2-f))
                        if e<=1:
                            output[d]=(f,d_band)
                        if e >= 3:
                            print(ident,e,"error frequency: ",t1,(t1,t2))
                        errors.append(e)
                        nb += 1
            print("    Results for",ident)
            if nb > 0:
                # print(ok, nb)
                print("Proportion in the right band:",ok/nb)
                print("Mean error:",np.mean(errors))
                print("Median error:",np.median(errors))
                print("Number:",len(errors))
                # print(errors)
                # print(np.percentile(errors, 50))
                # print(np.percentile(errors, 80))
                # print(np.percentile(errors, 90))
                print("95% of errors below:", np.percentile(errors, 95))
                # print(np.percentile(errors, 99))
            else:
                print("No attack detected")
            all_error += errors
        print("Mean frequency error:",np.mean(all_error))
        return output

    def evaluate(self, detected_positive_dico):
        self._seen_attack = []
        detected_positive = np.array([k for (k,_) in detected_positive_dico])
        total_positives = len(detected_positive)
        true_positive_list = detected_positive[list(map(self.is_in_attack_detected, detected_positive))]
        false_positive_list = detected_positive[[t not in true_positive_list for t in detected_positive]]
        true_positive = len(true_positive_list)
        false_positive = total_positives - true_positive
        self.true_positive_dates = [(a,b) for (a,b) in detected_positive_dico if a in true_positive_list]
        # print(len(self.true_positive_dates),len(detected_positive_dico))
        # print(len(true_positive_list))
        # print(len(self.true_positive_dico))
        recall = {}
        print("\nDetected : ",len(self._seen_attack),"/",len(self._attack))

        # true_positive_score = np.array([detected_positive_dico[t] for t in true_positive_list])
        # false_positive_score = np.array([detected_positive_dico[t] for t in false_positive_list])

        # detection that correspond to at least one attack
        useful_detection=[{},{},{}]
        partially_useful_detection=[{},{},{}]
        all_detection=[{},{},{}]
        total_time = 772722500
        new_method = True
        if new_method:
            l_i = [0,0,0]
            l_a = [0,0,0]

            identifiers = np.unique(self._all_attack[:,0])
            for ident in identifiers:
                attack_id = self._all_attack[self._all_attack[:,0] == ident]
                # attack_id = np.array(attack_id[:,1:].astype(np.integer))
                # attack_id = [(a,b) for [a,b,_] in attack_id]
                atk_number = len(attack_id)
                atk_detected = 0
                l_i_tmp = [0,0,0]
                l_a_tmp = [0,0,0]
                # for (a1,a2) in attack_id:
                for [_, a1, a2, a_band] in attack_id:
                    a1 = int(a1)
                    a2 = int(a2)
                    a_band = int(a_band)
                    l_a_tmp[a_band] += a2 - a1

                    detected = False
                    for (d1,d2) in detected_positive_dico:
                        d_band = detected_positive_dico[(d1,d2)][0]
                        all_detection[d_band][(d1,d2)]=True
                        if a_band == d_band:
                            i = intersect((d1-3750,d2-3750), (a1,a2))
                            if i:
                                useful_detection[d_band][(d1,d2)]=True
                                if (i[1] - i[0])/(d2-d1) > .5:
#                                    print(i,(d1,d2),(i[1] - i[0])/(d2-d1))
                                    partially_useful_detection[d_band][(d1,d2)]=True
                                detected = True
                                l_i_tmp[a_band] += i[1] - i[0]
                    if detected:
                        atk_detected += 1
                    elif a_band < 2:
                        print(ident,"not detected: ",a1,(a1,a2))
                for i in range(3):
                    l_i[i] += l_i_tmp[i]
                    l_a[i] += l_a_tmp[i]

                print("*** Detected for",ident,":",atk_detected,"/",atk_number,"=",atk_detected/atk_number)
                print("True positive:",sum(l_i_tmp),"ms")
                print("False negative:",sum(l_a_tmp)-sum(l_i_tmp),"ms")
                print("Recall for",ident,":",sum(l_i_tmp) / sum(l_a_tmp))
                recall[ident] = sum(l_i_tmp) / sum(l_a_tmp)


#                print("Number:",atk_detected)

            print("\n=== Result by band")
            l_d = [0,0,0]
            for (d1,d2) in detected_positive_dico:
                l_d[detected_positive_dico[(d1,d2)][0]] += d2 - d1

            for i in range(3):
                print("Useful detection number on band",str(i),":",len(useful_detection[i]),"/",len(all_detection[i]))

                if i<2:
                    k = sorted(list(all_detection[i].keys()))
                    for (d1,d2) in k:
                        if (d1,d2) not in useful_detection[i].keys():
                            print("False positive: ",(d2-d1),(d1,d2))
#                    if (d1,d2) not in partially_useful_detection[i].keys():
#                        print("Partial false positive: ",(d2-d1),(d1,d2))

                if l_d[i] > 0:
                    tp=l_i[i]
                    fp=l_d[i]-l_i[i]
                    fn=l_a[i]-l_i[i]
                    tn=total_time-fp-tp-fn
                    print("True positive on band",i,":",tp,"ms")
                    print("False positive on band",i,":",fp,"ms")
                    print("True negative on band",i,":",tn,"ms")
                    print("False negative on band",i,":",fn,"ms")
                    p = l_i[i] / l_d[i]
                    print("Detected time on band",i,":",l_d[i],"ms")
                    print("Precision for band",i,": ",p)
                    print("Recall on band",i,":",tp/(tp+fn),"ms")
                    print("Specificity for band",i,": ",tn/(tn+fp))
                    print("FPR for band",i,": ",1-tn/(tn+fp))
                    print("Accuracy for band",i,":",(tp+tn)/total_time)
                if l_a[i] > 0:
                    r = l_i[i] / l_a[i]
                    #print("Total attack time on band "+str(i)+": "+str(l_a[i]))
                    #print("Recall for band "+str(i)+": "+str(r))
                    if p+r > 0:
                        print("f-measure for band "+str(i)+": "+str(2*p*r/(p+r)))
                print("")

            print("=== Global results")
            print("Total detected time:",str(sum(l_d)),"ms")
            if sum(l_a) > 0 and sum(l_d) > 0 and sum(l_i) > 0:
                p = sum(l_i) / sum(l_d)
                r = sum(l_i) / sum(l_a)
                f = 2*p*r/(p+r)
                print("Global true positive: ",sum(l_i),"ms")
                print("Global false positive: ",sum(l_d)-sum(l_i),"ms")
                print("Global true negative: ",total_time-(sum(l_d)-sum(l_i)),"ms")
                print("Global false negative: ",sum(l_a)-sum(l_i),"ms")
                print("Precision",p,"Recall",r,"f-measure",f)
            else:
                p = 1
                r = 0
                f = 0
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
                        col.hlines(0.01,
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
                        # col.plot(x_dates, val, "bo-", alpha=0.7, label=labels[i])
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
        del data
    end = time.time()
    print("Scoring time:",(end-start),"s")
    joblib.dump(scores, path_examples)
    return scores

def get_cumul_threshold(models, scores, all_t):
    t=[[],[],[]]
    cumul = get_cumul(models, scores, all_t)
    for c in cumul:
        band=cumul[c][0][0]
        t[band].append(cumul[c][1])
    # return [np.percentile(t[i],99.9) for i in range(3)]
    out = []
    for i in range(3):
        print("band",i)
        print("0",np.percentile(t[i], 99.990))
        print("1",np.percentile(t[i], 99.991))
        print("2",np.percentile(t[i], 99.992))
        print("3",np.percentile(t[i], 99.993))
        print("4",np.percentile(t[i], 99.994))
        print("5",np.percentile(t[i], 99.995))
        print("6",np.percentile(t[i], 99.996))
        print("7",np.percentile(t[i], 99.997))
        print("8",np.percentile(t[i], 99.998))
        print("9",np.percentile(t[i], 99.999))
        out.append(np.percentile(t[i], 99.995))
    return out

def predict_extractors_cumul(models, scores, all_t, cumulated_threshold):
    print("Prediction from cumulative scores...")
    out = {}
    cumul = get_cumul(models, scores, all_t)
    for c in cumul:
        if cumul[c][1]>cumulated_threshold[cumul[c][0][0]]:
            out[c]=cumul[c][0]
    return out

def get_cumul(models, scores, all_t):
    start = time.time()
    example_pos = {}
    timestamps = sorted(scores.keys())
    for (_,m) in models:
        state = DetectorState.NOT_DETECTING
        # cumulated_threshold = 0.7
        # resting_duration = 0
        cumul = 0
        somme = 0
        nbSomme = 0
        # consecutive = 0
        previous_timestamp = None
        discontinuity_timestamp = timestamps[0]
        for timestamp in timestamps:
            discontinuity = (timestamp - discontinuity_timestamp > 3600000)
            # print(timestamp, discontinuity_timestamp, discontinuity, state)
            found = False
            for p in all_t:
                if p(timestamp): # we get the threshold of this timestamp
                    found = True
                    threshold_autoencoder = all_t[p]
                    low_threshold_autoencoder = threshold_autoencoder
            assert found, multimodels.process_unix_time(timestamp)
            score = scores[timestamp].get(m._number)


            if state == DetectorState.NOT_DETECTING and m.predict_thr(score,threshold=threshold_autoencoder[m._number]):
                state = DetectorState.DETECTING
                cumul = 0
                somme = 0
                nbSomme = 0
                previous_timestamp = timestamp

            elif (state == DetectorState.DETECTING or state == DetectorState.TRIGGERED) and (not m.predict_thr(score,threshold=low_threshold_autoencoder[m._number]) or discontinuity):
                example_pos[(previous_timestamp, discontinuity_timestamp)] = ([m._number],cumul,somme/nbSomme)
                previous_timestamp = timestamp
                state = DetectorState.NOT_DETECTING

            # elif state == DetectorState.DETECTING:
                # if cumul > cumulated_threshold:
                    # state = DetectorState.TRIGGERED

            if state == DetectorState.DETECTING:
                cumul += abs(score - threshold_autoencoder[m._number])
                somme += score
                nbSomme +=1

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
    start = time.time()
    for (t1, t2) in example_pos:
        if t2-t1 > 3600000:
            print("Attack too long ! Only 10mn")
            tend = t1 + 600000
        else:
            tend = t2
        l = []
        freq = frequency_to_index(example_pos[(t1,t2)][0])
        nb = example_pos[(t1,t2)][1]
        for i in range(3):
            m = median[i][nb]
            w = read_files_from_timestamp(t1, tend, folders_list[i],quant=False)[:,freq-1:freq+1]
            l.append((np.mean(w)-m, np.median(w)-m, np.std(w)))
        l.append(nb)
        l.append(example_pos[(t1,t2)][0])
        snr[(t1,t2)]=l
        print(l)
    end = time.time()
    print("SNR extraction:",(end-start),"s")
    return snr

def predict_frequencies(example_pos, folders, extractors):
    frequencies = {}
    start = time.time()
    for (t1, t2) in example_pos:
        if t2-t1 > 3600000:
            print("Attack too long ! Only 10mn")
            tend = t1 + 600000
        else:
            tend = t2
        nb = example_pos[(t1,t2)][0]
        # print(nb)
        waterfalls = read_files_from_timestamp(t1, tend, folders)
        # print(waterfalls.shape)
        # print(nb)
        # print("S",waterfalls.shape)
        waterfalls[:,0:nb*1000] = 0
        waterfalls[:,(nb+1)*1000:3000] = 0
        # print("S après",waterfalls.shape)
        (weights, data) = extractors.get_frequencies(waterfalls, number=nb)
        # print(data, weights)

        # use the max
        if len(data) >= 1:
            wmax = None
            fout = None
            for i in range(len(data)):
                w = weights[i]
                if wmax is None or w > wmax:
                    wmax = w
                    fout = data[i]
            f = index_to_frequency(fout+nb*1000)
        else:
            print("No signal!")
            f = index_to_frequency(nb*1000+500) # absence of signal



        if False: # use the weighted median
            if len(data) >= 2:
                median = weighted_median(data, weights)
                # print("Median:",median)
                median += nb*1000
            elif len(data) == 1:
                print("Almost no signal!")
                median = data[0] + nb*1000
            else:
                print("No signal!")
                median = nb*1000+500 # absence of signal
                # print(f)
            f = index_to_frequency(median)
        frequencies[(t1, t2)] = (f,nb)

    end = time.time()
    print("Freq localization:",(end-start),"s")
    return frequencies

def load_scores(path_examples_extractors, extractors, bands, directories):
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
        extractors.set_dummy(False)
        return score_extractors(extractors, path_examples_extractors, directories)

