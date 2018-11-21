from svm import OCSVM
from multimodels import *
from preprocess import *

nb_features = 2944

files = get_files_names(["/data/data/00.raw/raw/Adr_Expe_28-08_07-10/raspi1/"], "01_October")
files = ["features-"+d.split("/")[-1] for d in files]
print(files)
data = np.concatenate([np.fromfile(f).reshape(-1, nb_features + 1) for f in files])

print(data.shape)

periods = [period_night, period_day]
models = MultiModels()

for p in periods:
    detector = OCSVM()
    data = extract_period(data, p)
    if data.shape[0] > 0:
        print("Learning for",p.__name__,"from",data.shape[0],"examples")
        detector.learn(data[:2000,1:]) # should not learn the timestamp
        models.add_model(detector, p)
    else:
        print("No data to learn period",p.__name__)

models.save("micro-ocsvm.joblib")
