"""
    Anomaly detector based on local outlier factor
"""

from anomalydetector import AnomalyDetector
from sklearn.externals import joblib
from sklearn.neighbors import LocalOutlierFactor

class LOF(AnomalyDetector):

    def __init__(self):
        self._model = LocalOutlierFactor(novelty=True)

    def preprocess(self, data):
        pass

    def learn(self, data):
        self._model.fit(data)

#    def predict_list(self, data):
#        self._model.predict(data[-1,:])

    def predict(self, data, obs):
        return self._model.predict(obs)

    def get_memory_size(self):
        return 1

    def save(self, filename):
        joblib.dump(self._model, filename)

    def load(self, filename):
        self._model = joblib.load(filename)
