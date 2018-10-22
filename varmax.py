import statsmodels.api as sm
#from sklearn.preprocessing import StandardScaler
import numpy as np
from anomalydetector import AnomalyDetector

class Varmax(AnomalyDetector):
    """
        VARMAX model
    """
    def __init__(self, order):
        self._order = order
        self._varmax = None

    def learn(self, data, exo=None):
        # First, we de-mean the data
#        scaler = StandardScaler(with_std=False)
#        scaler.fit(data)
#        data = scaler.transform(data)

        self._varmax = sm.tsa.VARMAX(data, exo, self._order, enforce_stationarity=False).fit()
        print(self._varmax.summary())

    def predict(self, data, obs):
#        self._varmax.predict()
        pass # TODO
