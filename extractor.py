from config import Config
from abc import ABC, abstractmethod
from preprocess import *
import os.path

class MultiExtractors:

    def __init__(self):
        self._config = Config()
        self._models = []
        self._threshold = self._config.get_config_eval("autoenc_detection_threshold")

    def add_model(self, model, inf, sup):
        """
            models:
            index 0: debut spectre
            index 1: fin spectre
            index 2: model
        """
        self._models.append((inf, sup, model))

    def save(self):
        for m in self._models:
            m[2].save(str(m[0])+"-"+str(m[1])+"-")

    def load(self, i, s, model):
        model.load(str(i)+"-"+str(s)+"-")
        self._models.append((i, s, model))

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

        data = decompose(data, (size_x, original_shape[1]), overlap)
        # data est trop grand (2 waterfall), donc il faut en garder le bon nombre
        out = np.array([m[2].extract_features(data[0:len(delta_timestamp),:,m[0]:m[1],:]) for m in self._models])
        out = out.reshape(len(delta_timestamp),-1)
        out = np.concatenate((delta_timestamp, out), axis=1)
        return out

    def reconstruct(self, data):
        out = [m[2].reconstruct(data[:,m[0]:m[1]]) for m in self._models]
#        print("after reconstruct",np.array(out).shape)
        out = np.dstack(out)
#        print(out.shape)
        return out

    def rmse(self, data, m):
        out = np.array(m[2].squared_diff(data[:,m[0]:m[1]]))
        out = np.mean(out, axis=(0,2,3))
        out = np.sqrt(out)
        return out


    def predict(self, data):
        """
            Il y a une anomalie si la RMSE dépasse un certain seuil
        """
        for m in self._models:
            if any(self.rmse(data, m) > m._thresholds[1]):
                return True
        return False


    # def rmse_from_folders(self, fnames):
    #     data = np.array([self.rmse(read_file(f)) for f in fnames])
    #     data = np.concatenate(data)
    #     return data

    def learn_thresholds(self, fnames):
        # TODO : un jour à la fois
        data = np.array([read_file(f) for f in fnames])
        data = np.concatenate(data)
        for m in self._models:
            m[2].learn_threshold(data, m[0], m[1])

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
