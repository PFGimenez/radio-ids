#!/usr/bin/env python3

import numpy as np
from sklearn.decomposition import PCA
from preprocess import *
from hmm import HMM
import os.path
from sklearn.model_selection import train_test_split

def get_event_list(data, timestamps, temporal_step, spectral_step, overlap):
    data = np.concatenate(data)

    step_x = round(temporal_step * (1 - overlap))
    x = math.floor((data.shape[0] - temporal_step) / step_x) + 1
    out_time = []
    for i in range(x):
        out_time.append(float(timestamps[int((i*step_x) / waterfall_length)]))
    data = decompose_raw(data, (temporal_step, spectral_step), overlap)

    # Utiliser le timestamp des fichiers est plus fiable que calculer le temps théorique d'un waterfall car, dans les faits, il n'y a pas exactement 4s entre deux waterfalls mais un peu moins
    # Pour mesurer le temps :
#    print(list(map(int.__sub__, out_time[1:], out_time[:-1])))
    out = np.array([np.amax(data, axis=(2,3)),
                    np.mean(data, axis=(2,3)),
                    np.std(data, axis=(2,3)),
                    np.median(data, axis=(2,3))])
    out = np.hstack(out)
#    print(out.shape)
    return (out,out_time)

def extract_macro(output, output_time, directory_list, overlap):
    if not os.path.isfile(output) or not os.path.isfile(output_time):
        with open(directory_list) as f:
            folders = f.readlines()
        folders = [x.strip() for x in folders]
        print(folders)
#        folders = ["data-test2"]
        data = []
        data_time = []
        for f in folders:
            all_data = read_directory_with_timestamps(f)
            l = len(all_data[0])
            steps_nb = 24
            pas = int(l / steps_nb)
            for i in range(steps_nb):
                d = all_data[0][pas*i:pas*(i+1)]
                t = all_data[1][pas*i:pas*(i+1)]
                events = get_event_list(d, t, round(temporal_duration / waterfall_duration * waterfall_length), int(1500 / nb_macro_band), overlap)
#                print(events)
                data.append(events[0])
                data_time.append(events[1])

        data = np.concatenate(np.array(data))
        data_time = np.concatenate(np.array(data_time))
        print("Features stage 1 shape:",data.shape)
        data.tofile(output)
        data_time.tofile(output_time)
    else:
        print(output,"already extracted!")

config = Config()
temporal_duration = config.get_config_eval("macro_feature_duration")
waterfall_duration = config.get_config_eval("waterfall_duration")
waterfall_length = config.get_config_eval("waterfall_dimensions")[0]
nb_macro_band = config.get_config_eval("nb_macro_band")

# extract_macro(
#     os.path.join(config.get_config("section"),"test_"+config.get_config("macro_features_stage_1")),
#     os.path.join(config.get_config("section"),"time_test_"+config.get_config("macro_features_stage_1")),
#     "test_folders",
#     config.get_config_eval("macro_window_overlap_testing"))

extract_macro(
    os.path.join(config.get_config("section"),"train_"+config.get_config("macro_features_stage_1")),
    os.path.join(config.get_config("section"),"time_train_"+config.get_config("macro_features_stage_1")),
    "train_folders",
    config.get_config_eval("macro_window_overlap_training"))

