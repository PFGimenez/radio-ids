#!/usr/bin/env python3
import sys
from sklearn.externals import joblib
import os
import evaluate

# i = 1
# while i < len(sys.argv):
#     if sys.argv[i] == "-rp1":
#         i += 1
#         probe1 = sys.argv[i]
#     elif sys.argv[i] == "-rp2":
#         i += 1
#         probe2 = sys.argv[i]
#     elif sys.argv[i] == "-rp3":
#         i += 1
#         probe3 = sys.argv[i]
#     i += 1

path = "merged-intervals-union.joblib"
intervals = joblib.load(path)

median = [[-68,-68,-68], [-64,-64,-64], [-65,-65,-64]]
median = [[0,0,0], [0,0,0], [0,0,0]]

with open("test_folders_rp1") as f:
    f1 = f.readlines()
    f1 = [x.strip() for x in f1]
with open("test_folders_rp2") as f:
    f2 = f.readlines()
    f2 = [x.strip() for x in f2]
with open("test_folders_rp3") as f:
    f3 = f.readlines()
    f3 = [x.strip() for x in f3]

folders = [f1, f2, f3]

snr = evaluate.get_snr(intervals, folders, median)
print(snr)
joblib.dump(snr, "snr11.joblib")

