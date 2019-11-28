from preprocess import show_histo, read_files, quantify
from sklearn.externals import joblib
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from hmmlearn import hmm
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from multimodels import period_always, extract_period, process_unix_time
import os.path
from config import Config
import math
import preprocess
from heapq import nlargest, nsmallest
import datetime

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


    def predict_thr_from_data(self, data, epoch, nbThreshold = 1, optimistic=True):
        """
            optimistic is useless with a single detector
        """
        return predict_thr(self.get_score(data, epoch))

    def predict_thr(self, score, nbThreshold = 1, threshold=None, optimistic=True):
        """
            optimistic is useless with a single detector
        """
        if not threshold:
            self._thresholds = [1,1,1,1,1] # TODO VIRER !!!
            threshold = self._thresholds[nbThreshold]
        if self.anomalies_have_high_score():
            return score > threshold
        return score < thresholds

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

    def get_worse_score(self, data, number):
        """
        return the worse score among a dataset
        """
        memory_size = self.get_memory_size()
        predictions = [self.get_score(data[i-1-memory_size:i,1:]) for i in range(memory_size+1,len(data))]
        if self.anomalies_have_high_score():
            score = nlargest(number, predictions)
        else:
            score = nsmallest(number, predictions)
        print("Max score:",max(score))
        return np.array([data[predictions.index(s)] for s in score]), max(score)

    def learn_threshold(self, data):
#        print(data.shape)
        memory_size = self.get_memory_size()
#        i = memory_size
#        print(memory_size, i-1-memory_size, i)
        predictions = np.array([self.get_score(data[i-1-memory_size:i,:]) for i in range(memory_size+1,len(data))])
        print(max(predictions))
        # on retire les None
        predictions = np.array([x for x in predictions if x is not None])
        self._learn_threshold_from_scores(predictions)

    def _learn_threshold_from_scores(self, predictions):
        r = [100,99.5,98,97,96,95,93,90] if self.anomalies_have_high_score() else [0,1,2,3,4,5,7,10]
        self._thresholds = [np.percentile(predictions, p) for p in r]
        print("Thresholds:",self._thresholds)
        # plt.hist(predictions, log=False, bins=100)
        # plt.show()

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
        self._mem_size = 5

    def learn(self, data):
        self._model = hmm.GaussianHMM(self._nb_states, "full", verbose=True)
#        self._model = hmm.GMMHMM(self._nb_states, n_mix=5, covariance_type="full", verbose=True)
        self._model.fit(data)
        print("Model learnt")

    def anomalies_have_high_score(self):
        # Ce sont des log-probabilités. Les anomalies sont rares dont avec une faible proba
        return False


    def get_score(self, data, epoch=None):
        return self._model.score(data) - self._model.score(data[:-1])
        # evaluation : P(X) / P(X[:-1])

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
            print(m)
            if f(epoch) and not m.predict_thr_from_data(data, epoch):
                return False
        return True

    def learn(self, data):
        """
            The learning must be done by each model directly
        """
        assert False

    def get_score(self, data, epoch=None):
        if epoch == None:
            raise ValueError("Epoch is missing !")
        s = {}
        # get the score for each model enable at this date
        for (f,m) in self._models:
            if f(epoch):
                s[m._number] = m.get_score(data)
        return s
#        if self.anomalies_have_high_score():
#            return min(s)
#        else:
#            return max(s)

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
        for i in range(len(self._models)):
            self._models[i][1]._number = i

    def get_memory_size(self):
        return max([m.get_memory_size() for (_,m) in self._models])

    def predict_thr(self, score, nbThreshold = 1, threshold=None, optimistic=True):
        """
            Optimistic detection (no detection if at least one model sees no detection)
        """
        p = []
        i = 0
        for (_,m) in self._models:
            if score.get(m._number):
                if isinstance(score.get(m._number), list):
                    s = max(score.get(m._number))
                else:
                    s = score.get(m._number)
                p.append(m.predict_thr(s, nbThreshold, threshold=threshold[i], optimistic=optimistic))
            i += 1
        if optimistic:
            return all(p) # detection if all detectors detect
        else:
            return any(p) # detection if one detector detects

    def get_predictor(self, score, nbThreshold = 1, threshold=None, optimistic=True):
        p = []
        i = 0
        for (_,m) in self._models:
            if score.get(m._number):
                if isinstance(score.get(m._number), list):
                    s = max(score.get(m._number))
                else:
                    s = score.get(m._number)
                if m.predict_thr(s, nbThreshold, threshold=threshold[i], optimistic=optimistic):
                    p.append(m._number)
            i += 1
        return p

class MultiExtractors(MultiModels):

    def predict(self, data, epoch):
        """ If a least one mini-waterfall throws a detection
        """
        data = self._models[0][1].decompose_test(data)
        for d in data:
            if super().predict(d, epoch):
                return True
        return False

    def decompose_test(self, data):
        return self._models[0][1].decompose_test(data)

    def __init__(self):
        super().__init__()
        self._config = Config()
        overlap_test = self._config.get_config_eval('extractors_window_overlap_testing')
        waterfall_duration = self._config.get_config_eval("waterfall_duration")
        autoenc_dimensions_t = self._config.get_config_eval("autoenc_dimensions")[0][0]
        waterfall_dimensions_t = self._config.get_config_eval("waterfall_dimensions")[0]
        self._delta_t = int(waterfall_duration * autoenc_dimensions_t / waterfall_dimensions_t * (1 - overlap_test))
        self._is_dummy = True

    def add_model(self, model):
        """
            models:
            index 0: debut spectre
            index 1: fin spectre
            index 2: model
        """
        super().add_model(model, period_always)

    def save_all(self, suffix=""):
        """
        Sauvegarde spéciale (chaque autoencodeur a sa sauvegarde)
        """
        for (f,m) in self._models:
            m.save(str(m._i)+"-"+str(m._s)+"-")

    def load_model(self, model):
        if self._is_dummy:
            model.load(str(model._i)+"-"+str(model._s)+"-")
            self.add_model(model)

    def set_dummy(self, dummy):
        self._is_dummy = dummy

    def extract_features(self, data, initial_timestamp, overlap):
        waterfall_duration = self._config.get_config_eval('waterfall_duration')
        original_shape = self._config.get_config_eval('waterfall_dimensions')
        size_x = self._config.get_config_eval('big_sweep_temporal_dimension')
        step_x = round(size_x * (1 - overlap))
        x = math.floor((original_shape[0] - size_x) / step_x) + 1
        delta_timestamp = np.array(range(2*x)) * step_x * waterfall_duration / original_shape[0] + size_x / 2 * waterfall_duration / original_shape[0]
        delta_timestamp = delta_timestamp[delta_timestamp < waterfall_duration]
        delta_timestamp = delta_timestamp.reshape(delta_timestamp.shape[0], 1)
        delta_timestamp += initial_timestamp

        data = preprocess.decompose(data, (size_x, original_shape[1]), overlap)
        # data est trop grand (2 waterfall), donc il faut en garder le bon nombre
        out = np.array([m.extract_features(data[0:len(delta_timestamp),:,m._i:m._s,:]) for _,m in self._models])
        out = out.reshape(len(delta_timestamp),-1)
        out = np.concatenate((delta_timestamp, out), axis=1)
        return out

    def reconstruct(self, data):
        out = [m.reconstruct(data[:,m._i:m._s]) for _,m in self._models]
#        print("after reconstruct",np.array(out).shape)
        out = np.dstack(out)
#        print(out.shape)
        return out

    def diff_reconstruct(self, data):
        original = [data[:,m._i:m._s] for _,m in self._models]
        original = np.dstack(original)
        out = [m.reconstruct(data[:,m._i:m._s]) for _,m in self._models]
        out = np.dstack(out)
        return original - out

    def learn_threshold_from_scores(self, scores, period=None):
        thr = {}
        for (_,m) in self._models:
            r = [100,99,98,97,96,95,93,90] if m.anomalies_have_high_score() else [0,1,2,3,4,5,7,10]
            score_model = []
            for s in scores:
                if period and period(s):
                    val = scores.get(s).get(m._number)
                    if val != None:
                        score_model.append(val)
            thr[m._number] = [np.percentile(score_model, p) for p in r]
            print("Model",m._number,":",thr[m._number])
        return thr

    def learn_threshold(self, fnames):
        scores = {m : [] for (_,m) in self._models}
        # un jour à la fois
        for flist in fnames:
            print("Learn from",flist[0],len(flist))
            data = read_files(flist,quant=True)
#            quantify(data)
            for (_,m) in self._models:
                scores[m].append([m.get_score_vector(d) for d in data])
        for (_,m) in self._models:
            m._learn_threshold_from_scores(np.array(scores[m]).flatten())

    def get_frequencies(self, data, number=None):
        if number == None:
            s = []
            for (_,m) in self._models:
                s.append(m.get_diff_vector(data))
        else:
            s = [self._models[number][1].get_diff_vector(data)]
        s = np.array(s)
        # print(s.shape)
        s = np.dstack(s)
        # print(s.shape)
        s = np.sum(s, axis=1)
        # print(s.shape)
        nz = np.nonzero(s)
        index = list(nz[1])
        data = s[nz]
        return (data, index)

    def get_score(self, data, epoch=None):
        s = []
        for (_,m) in self._models:
            s.append(m.get_score_vector(data))
        scores = {}
        for i in range(len(s[0])):
            l = {}
            j = 0
            for (_,m) in self._models:
                l[m._number] = s[j][i]
                j += 1
            scores[epoch + i*self._delta_t] = l
        return scores

class FeatureExtractor(ABC):

    @abstractmethod
    def learn_extractor(self, filenames, inf, sup):
        pass

    @abstractmethod
    def reconstruct(self, data):
        """
            Only for the autoencoder
        """
        pass

    @abstractmethod
    def extract_features(self, data):
        pass

    @abstractmethod
    def save(self, prefix):
        pass

    @abstractmethod
    def load(self, prefix):
        pass
