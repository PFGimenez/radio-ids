from preprocess import *
from autoencodercnn import CNN
import sys

directories = get_files_names(["/data/data/00.raw/raw/Adr_Expe_28-08_07-10/raspi1/"], "01_October")

with open("train_folders") as f:
    folders = f.readlines()
directories = [x.strip() for x in folders]

with open("test_folders") as f:
    folders = f.readlines()
directories = directories + [x.strip() for x in folders]

autoenc = CNN()
try:
    autoenc.load()
    print(directories)
    for d in directories:
        print("Extracting features from",d)
        filenames = get_files_names([d])

        out = np.array([autoenc.extract_features(autoenc.decompose(read_file(filename)), int(os.path.split(filename)[1])) for filename in filenames])

        # take only the last part of the directory
        d2 = d.split("/")[-1]
        out.tofile("features-"+d2)
except Exception as e:
    print(e)
    raise e
