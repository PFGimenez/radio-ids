"""
    Anomaly detector based on one-class SVM
"""

from anomalydetector import AnomalyDetector
from sklearn.externals import joblib
from sklearn.svm import OneClassSVM

class OCSVM(AnomalyDetector):
    """
        One-class SVM
    """
    def __init__(self, kernel="rbf"):
        self._model = OneClassSVM(gamma='scale', kernel=kernel)
        # TODO : tester gamma="auto"

    def preprocess(self, data):
        pass

    def learn(self, data):
        self._model.fit(data)

    def predict_list(self, data):
        self._model.predict(data[-1,:])

    def predict(self, data, obs):
        self._model.predict(obs)

    def save(self, filename):
        joblib.dump(self._model, 'ocsvm.joblib')

    def load(self, filename):
        self._model = joblib.load('ocsvm.joblib')
