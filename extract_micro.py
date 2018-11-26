from preprocess import *
from autoencodercnn import CNN
import sys
import os
from config import Config

def extract_micro(autoenc, directories, window_overlap, prefix, nb_features):
    for d in directories:
        print("Extracting features from",d)
        filenames = get_files_names([d])

        out = []
        for i in range(len(filenames)-1):
            out.append(
                autoenc.extract_features(
                    autoenc.decompose(
                        np.concatenate(read_files([filenames[i], filenames[i+1]])),
                        window_overlap),
                    int(os.path.split(filenames[i])[1])))
        out = np.array(out).reshape(-1, nb_features+1)
        print(out.shape)
        exit()
        # take only the last part of the directory
        d2 = d.split("/")[-1]
        out.tofile(os.path.join(prefix,"features-"+d2))

config = Config()
prefix = config.get_config("section")
nb_features = config.get_config_eval("nb_features")

autoenc = CNN()
autoenc.load()

with open("train_folders") as f:
    folders = f.readlines()
directories = [x.strip() for x in folders]

extract_micro(autoenc, directories, config.get_config_eval("window_overlap_training"), prefix, nb_features)

with open("test_folders") as f:
    folders = f.readlines()
directories = [x.strip() for x in folders]

extract_micro(autoenc, directories, config.get_config_eval("window_overlap_testing"), prefix, nb_features)


