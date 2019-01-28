"""
    Anomaly detector based on one-class SVM
"""

from anomalydetector import AnomalyDetector
from sklearn.externals import joblib
from sklearn.svm import OneClassSVM
import numpy as np

class OCSVM(AnomalyDetector):
    """
        One-class SVM
    """
    def __init__(self, kernel="rbf"):
        self._model = OneClassSVM(gamma='scale', kernel=kernel)
#        self._thresholds = None
        # TODO : tester gamma="auto"

    def learn(self, data):
        self._model.fit(data)

    def get_score(self, data):
        return self._model.decision_function(obs)

    def anomalies_have_high_score(self):
        return True

    def predict(self, data, obs):
        return self._model.predict(obs) == -1

    def get_memory_size(self):
        return 1

    def save(self, filename):
        joblib.dump(self._model, filename)

    def load(self, filename):
        self._model = joblib.load(filename)
