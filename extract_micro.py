from preprocess import *
from autoencodercnn import CNN
import sys
overlap = 0

directories = get_files_names(["/data/data/00.raw/raw/Adr_Expe_28-08_07-10/raspi1/"], "28_August")

shape = (16,1472)

autoenc = CNN(shape, 0.8, -150, 0)
try:
    autoenc.load("test-3cf6357.h5")
    print(directories)
    for d in directories:
        d2 = d.split("/")[-1]
        print(d2)
        print("Extracting features from ",d)
        filenames = get_files_names([d])
#        print(filenames)
        extracted = np.array([autoenc.extract_features(
            autoenc.decompose(read_file(filename)))
            for filename in filenames])
        extracted.tofile("features-"d2)
except Exception as e:
    print(e)
    raise e
