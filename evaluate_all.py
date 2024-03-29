#!/usr/bin/env python3
import evaluate
import sys
from sklearn.externals import joblib
import os
import config as conf

# ce script suppose que "evaluate_one_probe" ait déjà été utilisé pour chaque sonde
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

print("Merge intervals…")

config = conf.Config()
cumulated_threshold = config.get_config_eval('cumul_threshold')
path = "results-frequencies-cumul"+str(cumulated_threshold)+".joblib"
# path = "results-detection-intervals-cumul.joblib"

intervals45 = {}
intervals89 = {}
intervals2425 = {}
if probes45:
    intervals45 = joblib.load(os.path.join(probes45, path))
    # print(intervals45)
    # exit()
if probes89:
    intervals89 = joblib.load(os.path.join(probes89, path))
if probes2425:
    intervals2425 = joblib.load(os.path.join(probes2425, path))

intervals45 = evaluate.merge(intervals45)
intervals89 = evaluate.merge(intervals89)
intervals2425 = evaluate.merge(intervals2425)

merged1 = evaluate.merge_all(0,intervals45,intervals89,intervals2425)
merged2 = evaluate.merge_all(1,intervals89,intervals45,intervals2425)
merged3 = evaluate.merge_all(2,intervals2425,intervals89,intervals45)
merged = evaluate.merge({**merged1, **merged2, **merged3})

path_output = "merged-intervals.joblib"
joblib.dump(merged, path_output)


merged_union = evaluate.merge({**intervals45, **intervals89, **intervals2425})
merged_union = evaluate.merge(merged_union)
path_output_union = "merged-intervals-union.joblib"
joblib.dump(merged_union, path_output_union)

merged_without_frequencies_union = {t:[v] for (t,(_,v)) in merged_union.items()}
# merged_without_frequencies = {t:[v] for (t,(_,v)) in merged.items()}
_,attack,attack_freq = evaluate.load_attacks()
e = evaluate.Evaluator(attack, attack_freq)
e.evaluate(merged_without_frequencies_union)
# e.evaluate(merged_without_frequencies)
merged_union = e.evaluate_freq(merged_union)
joblib.dump(merged_union, path_output_union)
