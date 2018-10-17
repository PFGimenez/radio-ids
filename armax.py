import statsmodels.api as sm
import numpy as np
from anomalydetector import *

class Armax(AnomalyDetector):
    """
        ARMAX model
    """

    def __init__(self):
        self._armax = None

    def learn(self, data, exo):
        self._armax = sm.tsa.ARMA(data, order=(3, 3), exog=exo).fit()

    def predict(self, data):
        pass # TODO
