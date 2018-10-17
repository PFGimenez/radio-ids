"""
    Hidden Markov Model learning
"""

import numpy as np
from hmmlearn import hmm
from anomalydetector import AnomalyDetector

class HMM(AnomalyDetector):
    """
        Hidden markov chain
    """
    def __init__(self, nb_states):
        self._nb_states = nb_states
        self._model = None

    def learn(self, data, exo=None):
#x = np.random.random(1000).reshape(-1,5)
#print(x)
        self._model = hmm.GaussianHMM(self._nb_states, "full")
        self._model.fit(data)

    def predict(self, data, obs):
        # TODO
        z = self._model.predict(data)
        print(z)

# evaluation : model.score(X) / model.score(X[:-1])
