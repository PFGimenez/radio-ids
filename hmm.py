"""
    Hidden Markov Model learning
"""

import numpy as np
from hmmlearn import hmm
from anomalydetector import AnomalyDetector
from preprocess import do_PCA

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
        self._model = hmm.GaussianHMM(self._nb_states, "full")
        self._model.fit(data)

    def predict(self, data, obs):
        # TODO
        z = self._model.predict(data)
        print(z)

    def save(self, filename):
        pass

    def load(self, filename):
        pass
# evaluation : model.score(X) / model.score(X[:-1])
