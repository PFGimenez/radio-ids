"""
    Hidden Markov Model learning
"""

import numpy as np
from hmmlearn import hmm
from anomalydetector import AnomalyDetector
from preprocess import do_PCA
from sklearn.externals import joblib

class HMM(AnomalyDetector):
    """
        Hidden markov chain
    """
    def __init__(self, nb_states, threshold):
        self._nb_states = nb_states
        self._model = None

    def preprocess(self, data):
        return do_PCA(data, 0.95)

    def learn(self, data, exo=None):
#x = np.random.random(1000).reshape(-1,5)
#print(x)
        self._model = hmm.GaussianHMM(self._nb_states, "full", verbose=True)
        self._model.fit(data)
        print(self._model.decode(data))
        print(self._model.covars_)

    def predict(self, data, obs):
        return self._model.score(np.concatenate((data,np.expand_dims(obs, axis=0)))) - self._model.score(data)
        # evaluation : P(X) / P(X[:-1])

    def save(self, filename):
        joblib.dump(self._model, "hmm.pkl")

    def load(self, filename):
        self._model = joblib.load("filename.pkl")
