import statsmodels.api as sm
import numpy as np
from anomalydetector import AnomalyDetector
import pickle
from preprocess import do_PCA

class Armax(AnomalyDetector):
    """
        ARMAX models
        Since ARMAX are univariate, learn one model for each feature
    """

    def __init__(self, order, distance, threshold):
        self._order = order
        self._armax = None
        self._distance = distance
        self._threshold = threshold

    def preprocess(self, data):
        return do_PCA(data, 0.95)

    def learn(self, data, exo=None):
        # First, we de-mean the data
        # In fact, it's useless if the data come from a PCA
#        scaler = StandardScaler(with_std=False)
#        scaler.fit(data)
#        data = scaler.transform(data)
        self._armax = [sm.tsa.ARMA(data[:, i], self._order, exo).fit()
                       for i in range(data.shape[1])]
#        print(self._armax.summary())

    def predict(self, data, obs):
        # If there is not enough previous observations
        if len(data) < max(order[0], order[1]):
            return False # by default, not an anomaly
        prediction = [self._armax.predict(data[:, i]) for i in range(data.shape[1])]
        return self._distance(prediction, obs) > self._threshold
