import statsmodels.api as sm
import numpy as np
from anomalydetector import *

class Varmax(AnomalyDetector):
    """
        VARMAX model
    """
    def __init__(self, order):
        self._order = order

    def learn(self, data, exo):
        self._varmax = sm.tsa.VARMAX(data, exo, self._order).fit()

    def predict(self, data):
        pass # TODO
