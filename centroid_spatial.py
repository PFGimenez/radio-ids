#!/usr/bin/env python3
from sklearn.externals import joblib
import csv
import math
import numpy as np
import evaluate
from pylab import *
import sys

def distance(tp1, tp2):
    return math.sqrt((tp1[0]-tp2[0])**2 + (tp1[1]-tp2[1])**2 + (tp1[2]-tp2[2])**2)

def distance_pos(pos1, pos2):
    return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

def load_atk(filename):
    attack = np.loadtxt(filename, dtype='<U13', usecols=(0,1,2,4,5))
    attack = np.array([a for a in attack if a[3]!="0"])
    # print(attack)
    identifiers = np.unique(attack[:,0])
    # print(identifiers)
    return attack

smin = float(sys.argv[1]) if len(sys.argv)>1 else -65
snr = joblib.load("snr11.joblib")
print(len(snr))
atk = load_atk("logattack")

attack_plot = {"scan433": 0, "anomaly467":0, "strong433": 0, "scan868": 1, "dosHackRF45": 0, "dosHackRF89": 1, "strong868": 1, "tvDOS": 0, "bruijnSequenc": 0, "old-scan433": 0, "old-strong433": 0, "anomaly": 0, "bruijnHarmo": 1}

attack_freq_type = {"scan433": [433,433], "strong433": [433,433], "scan868":[868,868], "strong868":[868,868], "tvDOS":[485,499], "bruijnSequenc":[433.8,433.8], "bleScan": [2400,2480], "btScan": [2400,2480], "dosHackRF45": [400,500], "dosHackRF89": [800, 900], "floodZigbee": [2475,2485], "injectESB": [2400, 2500], "old-scan433": [433,433], "old-strong433": [433,433], "anomaly467": [467,467], "wifiDeauth": [2451,2473], "wifiRogueAP": [2461,2483], "wifiScan": [2400,2500], "zigbeeScan": [2400,2480], "anomaly":[462,462],
                        "bruijnHarmo": [867.7,867.7]}


result = {}
k = 6

probes_pos = [(8,11),(1,5),(12.5,3)]
all_pos=[]
for t in snr:
    d = snr[t]
    nb = d[3] # sur quelle plage de fréquence ?
    f = d[4] # sur quelle fréquence estimée ?
    tp = (d[0][1], d[1][1], d[2][1]) # le triplet de SNR
    (t1,t2) = t # dates estimées de l'attaque
    t1 -= 3750
    t2 -= 3750
    # if t2-t1 > 200000:
        # continue
    # print(train_by_pos)
    w = []
    for i in range(3):
        w.append((tp[i]-smin)**2)

    posx=sum([w[i]*probes_pos[i][0] for i in range(3)])/sum(w)
    posy=sum([w[i]*probes_pos[i][1] for i in range(3)])/sum(w)
    pos=(posx,posy)
    all_pos.append(pos)

    if not math.isnan(tp[0]) and not math.isnan(tp[1]) and not math.isnan(tp[2]):

        # on recherche l'attaque qui correspond
        for a in atk:
            f1,f2 = attack_freq_type[a[0]]
            # if f < f1-10 or f > f2+10:
                # continue
            # print(f1,f2,a[2])
            nbTh = attack_plot.get(a[0])
            # hack : on n'évalue pas sur la bande 2.4-2.5 GHz
            if nbTh == None:
                continue
                # nbTh = 2
            i = evaluate.intersect((int(a[1]),int(a[2])), t)
            if i == None: # on vérifie qu'il y a une intersection non-négligeable et qu'il s'agit de la bonne bande
                continue

            if nb == nbTh and (i[1] - i[0]) / (t2-t1) > 0.5 and (i[1] - i[0]) / (int(a[2]) - int(a[1])) > 0.5:
                index = (a[0],a[3],a[4])
                # print(f1,f2,f)

                if not result.get(index):
                    s = []
                else:
                    s = result[index]

                s.append(distance_pos(pos, (int(a[3]), int(a[4]))))
                plot([pos[0],int(a[3])], [pos[1], int(a[4])], 'o-',color="blue")
                # if a[0] == "old-strong433":
                    # print(tp,info,nn,int(a[3]),int(a[4]))
                result[index] = s
                break

s2 = []

for i in result:
    s = result[i]
    print(i, np.median(s)*.6,"m, #points:",len(s))
    # if(len(s)>100):
    s2.append(np.median(s)*.6)
print(len(s2))
print("Mean:",np.mean(s2),"m")
print("Median:",np.median(s2),"m")
print("Std:",np.std(s2))

if True:
    probes_pos_zip=list(zip(*probes_pos))
    all_pos_zip=list(zip(*all_pos))
    scatter(probes_pos_zip[0],probes_pos_zip[1], s=100 ,marker='o',color="red")
    scatter(all_pos_zip[0],all_pos_zip[1], s=100 ,marker='o')
    show()
