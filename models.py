from preprocess import show_histo
from sklearn.externals import joblib
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from hmmlearn import hmm
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class AnomalyDetector(ABC):
    """
    Abstract class for all anomaly detector algorithms
    """

    @abstractmethod
    def learn(self, data):
        """
            Learn from data
        """
        pass

    def predict_thr(self, data, epoch, nbThreshold = 1):
        return self.get_score(data, epoch) < self._thresholds[nbThreshold]

    def histo_score(self, data):
        s = [get_score(d) for d in data]
        show_histo(s)

    @abstractmethod
    def get_memory_size(self):
        pass

    @abstractmethod
    def get_score(self, data, epoch=None):
        """ Return the score of data """
        pass

    @abstractmethod
    def anomalies_have_high_score(self):
        pass

    def learn_threshold(self, data):
#        print(data.shape)
        memory_size = self.get_memory_size()
#        i = memory_size
#        print(memory_size, i-1-memory_size, i)
        predictions = np.array([self.get_score(data[i-1-memory_size:i,:]) for i in range(memory_size+1,len(data))])
        # on retire les None
        predictions = [x for x in predictions if x is not None]
        r = [0,1,2,3,4,5,7,10] if self.anomalies_have_high_score() else [100,99,98,97,96,95,93,90]
        self._thresholds = [np.percentile(predictions, p) for p in r]
        print(self._thresholds)
        plt.hist(predictions, log=False, bins=100)
        plt.show()

    @abstractmethod
    def save(self, filename):
        pass

    @abstractmethod
    def load(self, filename):
        pass

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

    def anomalies_have_high_score(self):
        # Ce sont des log-probabilités. Les anomalies sont rares dont avec une faible proba
        return False


    def get_score(self, data, epoch=None):
        # TODO : peut-être que ce n'est pas nécessaire
        if len(data) < self._mem_size:
            return None
        # TODO formule
#        return self._model.score(np.concatenate((data,np.expand_dims(obs, axis=0)))) - self._model.score(data)
        # evaluation : P(X) / P(X[:-1])
        return self._model.score(data[len(data) - self._mem_size:])

    def save(self, filename):
        joblib.dump(self._model, filename)

    def load(self, filename):
        self._model = joblib.load(filename)

    def get_memory_size(self):
        return self._mem_size

class OCSVM(AnomalyDetector):
    """
        Anomaly detector based on one-class SVM
    """
    def __init__(self, kernel="rbf"):
        self._model = OneClassSVM(gamma='scale', kernel=kernel)
#        self._thresholds = None
        # TODO : tester gamma="auto"

    def learn(self, data):
        self._model.fit(data)

    def get_score(self, data, epoch=None):
        assert len(data) == 1, "len(data) = "+str(len(data))
        return self._model.decision_function(data)

    def anomalies_have_high_score(self):
        return True

    def predict(self, data, obs):
        return self._model.predict(obs) == -1

    def get_memory_size(self):
        return 0

    def save(self, filename):
        joblib.dump(self._model, filename)

    def load(self, filename):
        self._model = joblib.load(filename)

class LOF(AnomalyDetector):
    """
        Anomaly detector based on local outlier factor
    """

    def __init__(self):
        self._model = LocalOutlierFactor(novelty=True)

    def learn(self, data):
        self._model.fit(data)

    def predict(self, data, obs):
        return self._model.predict(obs) == -1

    def get_score(self, data, epoch=None):
        return self._model.score_samples(data)

    def anomalies_have_high_score(self):
        return False

    def get_memory_size(self):
        return 0

    def save(self, filename):
        joblib.dump(self._model, filename)

    def load(self, filename):
        self._model = joblib.load(filename)


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

    def learn(self, data):
        pass
    # TODO : pas de timestamp
#        for (f,m) in self._models:
#            data_m = extract_period(data, f)
#            m.learn(data_m)

    def get_score(self, data, epoch=None):
        """
            Optimistic score
        """
        if epoch == None:
            raise ValueError("Impossible None")
        s = []
        # get the score for each model enable at this date
        for (f,m) in self._models:
            if f(epoch):
                s.append(m.get_score(data))
        if self.anomalies_have_high_score():
            return min(s)
        else:
            return max(s)

    def learn_threshold(self, data):
        for (f,m) in self._models:
            data_m = extract_period(data, f)
            m.learn_threshold(data_m)

    def anomalies_have_high_score(self):
        return self._models[0][1].anomalies_have_high_score()

    def save(self, filename):
        joblib.dump(self._models, filename)

    def load(self, filename):
        self._models = joblib.load(filename)

    def get_memory_size(self):
        return max([m.get_memory_size() for (_,m) in self._models])
