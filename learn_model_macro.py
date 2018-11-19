import numpy as np
from preprocess import *
import random
from hmm import HMM

np.random.seed()
random.seed()

try:
    data = np.fromfile("events_01_10_1mn").reshape(-1,15)
except:
    files = get_files_names(["/data/data/00.raw/raw/Adr_Expe_28-08_07-10/raspi1/"], "01_October")
    data = read_directory(files[0])
#    data = read_directory("data-test")
    data = get_event_list(data, 750, 100) # 1mn
    data.tofile("events_01_10_1mn")
detector = HMM(4, 0.1)
print(data.shape)
train_data = data[:1000,]
test_data = data[1000:,]
detector.learn(train_data)
predictions = test_prediction(test_data, detector)
print(np.max(predictions))
print(np.min(predictions))
print(np.mean(predictions))


#print(detector.predict_list(test_data))
