import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import numpy as np
from anomalydetector import AnomalyDetector

class Armax(AnomalyDetector):
    """
        ARMAX models
        Since ARMAX are univariate, learn one model for each feature
    """

    def __init__(self, order):
        self._order = order
        self._armax = None

    def learn(self, data, exo=None):
        # First, we de-mean the data
        scaler = StandardScaler(with_std = False)
        scaler.fit(data)
        data = scaler.transform(data)

        self._armax = [sm.tsa.ARMA(data[:,i], self._order, exo).fit()
                       for i in range(data.shape[1])]
#        print(self._armax.summary())

    def predict(self, data, obs):
        # If there is not enough previous observations
        if len(data) < order[0]:
            return False # by default, not an anomaly
        pass # TODO

