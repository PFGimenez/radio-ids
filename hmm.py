"""
    Hidden Markov Model learning
"""

from preprocess import *
import numpy as np
from hmmlearn import hmm
from anomalydetector import AnomalyDetector
from preprocess import do_PCA
from sklearn.externals import joblib

class HMM(AnomalyDetector):
    """
        Hidden markov chain
    """
    def __init__(self, nb_states):
        self._nb_states = nb_states
        self._model = None
        self._threshold = None
        self._mem_size = 100

    def learn(self, data):
        self._model = hmm.GaussianHMM(self._nb_states, "full", verbose=True)
#        self._model = hmm.GMMHMM(self._nb_states, n_mix=5, covariance_type="full", verbose=True)
        self._model.fit(data)
        print("Model learnt")

#         print("Threshold estimation…")

#         predictions = []
#         memory = []
#         i = 0
#         memory_size = self.get_memory_size()

#         for f in data:
#             if i % 100 == 0:
#                 print(i,"/",len(data))
#             i += 1
#             print(f)
#             if len(memory) == memory_size:
#                 memory.pop(0)

#             memory.append(f[1:])

#             if len(memory) != memory_size:
#                 continue

#             self.predict(np.array(memory), f[0])
#             predictions.append(self._model.score(np.concatenate((memory,np.expand_dims(f[0], axis=0)))) - self._model.score(memory))


#         print("max:",np.max(predictions))
#         p = np.percentile(predictions, 1)
#         print("1% quantile:",p)
#         print("min",np.min(predictions))
#         print("mean",np.mean(predictions))
#         self._threshold = p


    def anomalies_have_high_score(self):
        # Ce sont des log-probabilités. Les anomalies sont rares dont avec une faible proba
        return False


    def get_score(self, data):
        # TODO : peut-être que ce n'est pas nécessaire
        if len(data) < self._mem_size:
            return None
        return self._model.score(data[len(data) - self._mem_size:])

#    def predict_list(self, data):
#        if len(data) == 1:
#            return self._model.score(data) < self._threshold
#        return self._model.score(data) - self._model.score(data[0:len(data)-1,:]) < self._threshold

#    def predict(self, data, obs):
#        if len(data) == 1:
#            return self._model.score(data)
#        return self._model.score(np.concatenate((data,np.expand_dims(obs, axis=0)))) - self._model.score(data)
        # evaluation : P(X) / P(X[:-1])

    def save(self, filename):
        joblib.dump(self._model, filename)

    def load(self, filename):
        self._model = joblib.load(filename)

    def get_memory_size(self):
        return self._mem_size
