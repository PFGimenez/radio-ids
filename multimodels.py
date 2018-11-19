from enum import Enum

"""
    Multiple models
"""

class JourSemaine(Enum):
    LUNDI = 1
    MARDI = 2
    MECREDI = 3
    JEUDI = 4
    VENDREDI = 5
    SAMEDI = 6
    DIMANCHE = 7

def periode_week_end(day, time):
    return day == SAMEDI or day == DIMANCHE

def periode_semaine(day, time):
    return not periode_week_end(day)

def periode_journee(day, time):
    return time > 7*3600 and time < 19*3600

def periode_nuit(day, time):
    return time < 9*3600 or time > 18*3600

class MultiModels():

    def __init__(self):
        self._models = []

    def add_model(self, model, fun):
        self._models.append((fun, model))

    def predict(self, data, day, time):
        """
            No anomaly if at least one model says there isn't
        """
        for (f,m) in self._models:
            if f(day, time) and not m.predict_list(data):
                return False
        return True

    def save(self, filename):
        joblib.dump(self._models, filename)

    def load(self, filename):
        self._models = joblib.load(filename)
