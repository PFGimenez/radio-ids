#!/usr/bin/env python3

import numpy as np
from preprocess import *
from hmm import HMM

try:
    data = np.fromfile("events_01_10_1mn").reshape(-1,15)
except:
    files = get_files_names(["/data/data/00.raw/raw/Adr_Expe_28-08_07-10/raspi1/"], "01_October")
    data = read_directory(files[0])
#    data = read_directory("data-test")
    data = get_event_list(data, 750, 100) # 1mn
    data.tofile("events_01_10_1mn")
