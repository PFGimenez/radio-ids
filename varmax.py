import statsmodels.api as sm
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
        self._varmax = sm.tsa.VARMAX(data, exo, self._order).fit()

    def predict(self, data):
        pass # TODO
