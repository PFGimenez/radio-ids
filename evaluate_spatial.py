#!/usr/bin/env python3
import sys
from sklearn.externals import joblib
import os
import evaluate

probes45 = None
probes89 = None
probes2425 = None

i = 1
while i < len(sys.argv):
    if sys.argv[i] == "-p45":
        i += 1
        probes45 = sys.argv[i]
    elif sys.argv[i] == "-p89":
        i += 1
        probes89 = sys.argv[i]
    elif sys.argv[i] == "-p2425":
        i += 1
        probes2425 = sys.argv[i]
    i += 1

path = "merged-intervals.joblib"
intervals = joblib.load(path)

median = {probes45: [-68,-68,-68], probes2425: [-64,-64,-64], probes89: [-65,-65,-64]}

with open("test_folders_rp1") as f:
    f1 = f.readlines()
    f1 = [x.strip() for x in f1]
with open("test_folders_rp2") as f:
    f2 = f.readlines()
    f2 = [x.strip() for x in f2]
with open("test_folders_rp3") as f:
    f3 = f.readlines()
    f3 = [x.strip() for x in f3]

folders = {probes45: f1, probes89: f3, probes2425: f2}

snr = evaluate.get_snr(intervals, folders, median)
print(snr)
joblib.dump(snr, "snr.joblib")

