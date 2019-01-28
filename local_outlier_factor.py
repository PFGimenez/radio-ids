"""
    Anomaly detector based on local outlier factor
"""

from anomalydetector import AnomalyDetector
from sklearn.externals import joblib
from sklearn.neighbors import LocalOutlierFactor

class LOF(AnomalyDetector):

    def __init__(self):
        self._model = LocalOutlierFactor(novelty=True)

    def learn(self, data):
        self._model.fit(data)

    def predict(self, data, obs):
        return self._model.predict(obs) == -1

    def get_score(self, data):
        return self._model.score_samples(data)

    def anomalies_have_high_score(self):
        return False

    def get_memory_size(self):
        return 1

    def save(self, filename):
        joblib.dump(self._model, filename)

    def load(self, filename):
        self._model = joblib.load(filename)
