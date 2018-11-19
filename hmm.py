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

    def learn(self, data):
        self._model = hmm.GaussianHMM(self._nb_states, "full", verbose=True)
#        self._model = hmm.GMMHMM(self._nb_states, n_mix=5, covariance_type="full", verbose=True)
        self._model.fit(data)
#        print(self._model.decode(data))
#        print(self._model.covars_)

    def predict_list(self, data):
        if len(data) == 1:
            return self._model.score(data)
        return self._model.score(data) - self._model.score(data[0:len(data)-1,:])

    def predict(self, data, obs):
        if len(data) == 1:
            return self._model.score(data)
        return self._model.score(np.concatenate((data,np.expand_dims(obs, axis=0)))) - self._model.score(data)
        # evaluation : P(X) / P(X[:-1])

    def save(self, filename):
        joblib.dump(self._model, filename)

    def load(self, filename):
        self._model = joblib.load(filename)
