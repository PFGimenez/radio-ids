from preprocess import *
from autoencodercnn import CNN
import sys
import config

directories = get_files_names(["/data/data/00.raw/raw/Adr_Expe_28-08_07-10/raspi1/"], "01_October")

config = Config()

original_shape = config.get_config_eval('waterfall_dimensions')
autoenc_shape = config.get_config_eval('autoenc_dimensions')
window_overlap = config.get_config_eval('window_overlap')
min_value = config.get_config_eval('min_value')
max_value = config.get_config_eval('max_value')

autoenc = CNN()
try:
    autoenc.load()
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
