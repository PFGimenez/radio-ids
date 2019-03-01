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
    return (JourSemaine(date.weekday()), ((date.hour * 60) + date.minute) * 60 + date.second)

class Period():
    def __init__(self, deb=0, fin=24*3600, days=None):
        self._deb = deb
        self._fin = fin
        self._days = days

    def is_in_period(time):
        (day, time) = process_unix_time(time)
        return (self._days == None or day in self._days) and time >= self._deb and time <= self._fin

def period_always(time):
    return True

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


