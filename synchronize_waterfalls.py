#!/usr/bin/env python3
# synchronise les différentes flux de waterfalls

from config import Config
import os
import numpy as np
from preprocess import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

config = Config()
prefix = "raspi-merged"
waterfall_duration = config.get_config_eval("waterfall_duration")
delta_t = int(waterfall_duration * 1.5) # un peu de marge pour la recherche
waterfall_dimensions_time = config.get_config_eval("waterfall_dimensions")[0]
temporal_step = waterfall_duration / waterfall_dimensions_time

with open("raspi_folders") as f:
    folders = f.readlines()
directories = [x.strip() for x in folders]
days = os.listdir(directories[0])
#waterfalls = [[[os.path.join(d,day,f) for f in sorted(os.listdir(os.path.join(d,day)))] for day in days] for d in directories]
waterfalls_int = [[[int(f) for f in sorted(os.listdir(os.path.join(d,day)))] for day in days] for d in directories]
nb_rpi = len(directories)
nb_waterfall_per_day = int(((24*3600000)/waterfall_duration))
print(nb_waterfall_per_day)

# évaluation de la durée d'un waterfall
# summ = 0
# nb = 0
# for d in range(len(days)):
#     for r in range(nb_rpi):
#         l = len(waterfalls_int[r][d])
#         summ = summ + sum(np.array(waterfalls_int[r][d][1:]) - np.array(waterfalls_int[r][d][:l-1]))
#         nb += l-1
# print(summ/nb)

# waterfalls : index 0 = rpi, index 1 = jour, index 2 = fichier
for d in range(8,len(days)):
    # on crée le dossier de résultat s'il n'existe pas
    if not os.path.exists(os.path.join(prefix,days[d])):
        os.makedirs(os.path.join(prefix,days[d]))
    waterfall_date = max([waterfalls_int[r][d][0] for r in range(nb_rpi)])
    print("day",d,"/",len(days))
    for w in range(nb_waterfall_per_day):
#        print(waterfall_date)
        l = []
        error = False
        for r in range(nb_rpi):
#            print("RPI",r)
            files = [t for t in waterfalls_int[r][d] if t > waterfall_date - delta_t and t < waterfall_date + delta_t]
            if not files:
                error = True
            else:
#            print(files)
                initial_date = files[0]
#            print(files)
                files = [os.path.join(directories[r],days[d],str(f)) for f in files]
#            print(files)
                val = np.concatenate(read_files(files))
#            print(val.shape)
                begin = int((waterfall_date - initial_date) / temporal_step)
#            print(waterfall_date, initial_date, temporal_step)
#            print(begin)
                val = val[begin:begin + waterfall_dimensions_time]
                if val.shape[0] < waterfall_dimensions_time:
                    error = True
                else:
                    l.append(val)
        if error:
            print("Incomplete last waterfall")
        else:
            l = np.array(l)
            print(l.shape)
            scaler = StandardScaler()
            scaler.fit(l.reshape(-1,1))
            l = scaler.transform(l.reshape(-1,1)).reshape(3,50,1500)
            l_save = l
            print(l.shape)
            fig = plt.figure()
#            plt.imshow(np.concatenate(l,axis=1), cmap='hot', interpolation='nearest', aspect='auto')
            l = np.moveaxis(l,0,2)
            print(l.shape)
            pca = PCA(1, svd_solver="full")
            l = pca.fit_transform(l.reshape(-1,3))
            print(pca.components_)
            print(l.shape)
            l = l.reshape(50,1500)
#            plt.imshow(l_save[0], cmap='hot', interpolation='nearest', aspect='auto')
            plt.imshow(np.concatenate((np.concatenate(l_save,axis=1), l), axis=1), cmap='hot', interpolation='nearest', aspect='auto')
            plt.show()
            exit()
            l = np.max(l,axis=0)
#        print(l.shape)
            l.tofile(os.path.join(prefix,days[d],str(waterfall_date)))
        waterfall_date += waterfall_duration
