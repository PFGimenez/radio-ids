#!/usr/bin/env python3
from sklearn.externals import joblib
import csv
import math
import numpy as np
import evaluate

def extract_train(filename, median):
    with open(filename, newline="") as f:
        reader = csv.reader(f, delimiter=' ')
        next(reader) # skip first line
        m = {}
        for row in reader:
            m[(float(row[9]) - median[0],\
                float(row[10]) - median[1],\
               float(row[11]) - median[2])] = (float(row[15]),float(row[16]))
    return m

# distance manhattan, marche moins bien
# def distance(tp1, tp2):
    # return abs(tp1[0]-tp2[0]) + abs(tp1[1]-tp2[1]) + abs(tp1[2]-tp2[2])

def distance2(tp1, tp2):
    m1 = (tp1[0]+tp1[1]+tp1[2])/3
    m2 = (tp2[0]+tp2[1]+tp2[2])/3
    return math.sqrt((tp1[0]-m1-tp2[0]+m2)**2 + (tp1[1]-m1-tp2[1]+m2)**2 + (tp1[2]-m1-tp2[2]+m2)**2)

def distance(tp1, tp2):
    return math.sqrt((tp1[0]-tp2[0])**2 + (tp1[1]-tp2[1])**2 + (tp1[2]-tp2[2])**2)

def distance_pos(pos1, pos2):
    return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

def find_knn(k, m, tp):
    knn = []
    for neighbor in m:
        added = False
        if len(knn) < k:
            knn.append(neighbor)
            added = True
        elif distance(tp, neighbor) < distance_max:
            knn[index_to_remove] = neighbor
            added = True

        if added: # update index_to_remove and distance_max
            distance_max = -math.inf
            for i in range(len(knn)):
                n = knn[i]
                if distance(n, tp) > distance_max:
                    index_to_remove = i
                    distance_max = distance(n, tp)
    x = 0
    y = 0
    somme = 0
    for n in knn:
        pos = m[n]
        # d = distance(n, tp) + 0.01
        d = 1
        somme += 1/d
        x += pos[0]/d
        y += pos[1]/d
    x /= somme
    y /= somme
    return knn, (x,y)

def find_nn(m, tp):
    nn = None
    distance_min = math.inf
    for tp2 in m:
        if distance(tp, tp2) < distance_min:
            nn = tp2
            distance_min = distance(tp, tp2)
    assert nn != None
    return nn,m[nn]

def load_atk(filename):
    attack = np.loadtxt(filename, dtype='<U13', usecols=(0,1,2,4,5))
    print(attack)
    identifiers = np.unique(attack[:,0])
    print(identifiers)
    return attack


median = [[-68,-64,-65], [-68,-64,-65], [-68,-64,-64]]

train = []
train.append(extract_train("features_400-500", median[0]))
train.append(extract_train("features_800-900", median[1]))
train.append(extract_train("features_2400-2500", median[2]))

snr = joblib.load("snr.joblib")

atk = load_atk("logattack_position_no2425")

attack_plot = {"scan433": 0, "strong433": 0, "scan868": 1, "dosHackRF45": 0, "dosHackRF89": 1, "strong868": 1, "tvDOS": 0, "bruijnSequenc": 0, "old-scan433": 0, "old-strong433": 0, "anomaly": 0}

# print(find_knn(5, train[0], (18,9,0)))
# print(find_nn(train[0], (18,9,0)))
# exit()
result = {}
k = 5
for t in snr:
    d = snr[t]
    nb = d[3]
    tp = (d[0][1], d[1][1], d[2][1])

    if not math.isnan(tp[0]) and not math.isnan(tp[1]) and not math.isnan(tp[2]):
        _,nn = find_knn(k,train[nb], tp)
        # print(nn[1])
        for a in atk:
            nbTh = attack_plot.get(a[0])
            if nbTh == None:
                nbTh = 2
            if nb == nbTh and evaluate.intersect((int(a[1]),int(a[2])), t):
                index = (a[0],a[3],a[4])
                if not result.get(index):
                    s = []
                else:
                    s = result[index]
                s.append(distance_pos(nn, (int(a[3]), int(a[4]))))
                result[index] = s

s2 = []
for i in result:
    s = result[i]
    print(i, np.median(s)*.6)
    s2.append(np.median(s)*.6)
print(np.median(s2))
print(np.std(s2))
