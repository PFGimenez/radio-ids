#!/usr/bin/env python3
# synchronise les différentes flux de waterfalls

from config import Config
import os

config = Config()
prefix = config.get_config("section")

with open("raspi_folders") as f:
    folders = f.readlines()
directories = [x.strip() for x in folders]
days = os.listdir(directories[0])
waterfalls = [[[os.path.join(d,day,f) for f in sorted(os.listdir(os.path.join(d,day)))] for day in days] for d in directories]
waterfalls_int = [[[f for f in sorted(os.listdir(os.path.join(d,day)))] for day in days] for d in directories]
nb_rpi = len(directories)

# waterfalls : index 0 = rpi, index 1 = jour, index 2 = fichier
for d in range(len(days)):
    initial_date = min([waterfalls_int[r][d][0] for r in range(nb_rpi)])
    print(initial_date)
