from sklearn.externals import joblib
from enum import Enum
import datetime

"""
    Multiple models
"""

class JourSemaine(Enum):
    LUNDI = 0
    MARDI = 1
    MECREDI = 2
    JEUDI = 3
    VENDREDI = 4
    SAMEDI = 5
    DIMANCHE = 6

def process_unix_time(time):
    date = datetime.datetime.fromtimestamp(time/1000)
    return (JourSemaine(date.weekday()), ((date.hour * 24) + date.minute) * 60 + date.second)

def period_weekend(time):
    (day, time) = process_unix_time(time)
    return day == JourSemaine.SAMEDI or day == JourSemaine.DIMANCHE

def period_week(time):
    (day, time) = process_unix_time(time)
    return not period_weekend(day)

def period_day(time):
    (day, time) = process_unix_time(time)
    return time > 7*3600 and time < 19*3600

def period_night(time):
    (day, time) = process_unix_time(time)
    return time < 9*3600 or time > 18*3600

def extract_period(data, fun):
    return data[list(map(fun, data[:,0]))]

class MultiModels():

    def __init__(self):
        self._models = []

    def add_model(self, model, fun):
        self._models.append((fun, model))

    def predict(self, data, epoch):
        """
            No anomaly if at least one model says there isn't
        """
        for (f,m) in self._models:
            if f(epoch) and not m.predict_list(data):
                return False
        return True

    def save(self, filename):
        joblib.dump(self._models, filename)

    def load(self, filename):
        self._models = joblib.load(filename)

    def get_memory_size(self):
        return max([m.get_memory_size() for (_,m) in self._models])
