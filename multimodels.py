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

class Period():
    def __init__(self, deb=0, fin=24*3600, days=None):
        self._deb = deb
        self._fin = fin
        self._days = days

    def is_in_period(time):
        (day, time) = process_unix_time(time)
        return (self._days == None or day in self._days) and time >= self._deb and time <= self._fin

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

class MultiModels(AnomalyDetector):

    def __init__(self):
        self._models = []

    def add_model(self, model, fun):
        self._models.append((fun, model))

    def predict(self, data, epoch):
        """
            No anomaly if at least one model says there isn't
        """
        for (f,m) in self._models:
            if f(epoch) and not m.predict(data):
                return False
        return True


    def get_score(self, data, epoch):
        """
            Optimistic score
        """
        s = []
        # get the score for each model enable at this date
        for (f,m) in self._models:
            if f(epoch):
                s.append(m_get_score(data)
        if self.anomalies_have_high_score():
            return min(s)
        else:
            return max(s)

    def anomalies_have_high_score(self):
        return self._models[0][1].anomalies_have_high_score()

    def save(self, filename):
        joblib.dump(self._models, filename)

    def load(self, filename):
        self._models = joblib.load(filename)

    def get_memory_size(self):
        return max([m.get_memory_size() for (_,m) in self._models])
