"""
    Hidden Markov Model learning
"""

import numpy as np
from hmmlearn import hmm
from anomalydetector import *

class HMM(AnomalyDetector):
    """
        Hidden markov chain
    """
    def __init__(self, nb_states):
        self._nb_states = 5

    def learn(self, data, exo):
#x = np.random.random(1000).reshape(-1,5)
#print(x)
        model = hmm.GaussianHMM(_nb_states, "full")
        model.fit(data)

    def predict(self, data):
        # TODO
        z = model.predict(data)
        print(z)

# evaluation : model.score(X) / model.score(X[:-1])

