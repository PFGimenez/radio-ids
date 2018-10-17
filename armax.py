import statsmodels.api as sm
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
        print(data[:,1])
        print(data.shape)
        self._armax = [sm.tsa.ARMA(data[:,i], self._order, exo).fit()
                       for i in range(data.shape[1])]
        print(self._armax.summary())

    def predict(self, data):
        pass # TODO

