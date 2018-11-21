from preprocess import *
from autoencodercnn import CNN
import sys
overlap = 0

directories = get_files_names(["/data/data/00.raw/raw/Adr_Expe_28-08_07-10/raspi1/"], "01_October")

original_shape = (50, 1500)
shape = (16,1472)

autoenc = CNN(original_shape, shape, 0.2, -150, 0)
try:
    autoenc.load("test-3cf6357.h5")
    print(directories)
    for d in directories:
        d2 = d.split("/")[-1]
        print(d2)
        print("Extracting features from",d)
        filenames = get_files_names([d])
#        print(filenames)

        out = np.array([autoenc.extract_features(autoenc.decompose(read_file(filename)), int(os.path.split(filename)[1])) for filename in filenames])
        print(out.shape)
        out.tofile("features-"+d2)
except Exception as e:
    print(e)
    raise e
