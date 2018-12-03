from config import Config
from abc import ABC, abstractmethod
from preprocess import *

class MultiExtractors:

    def __init__(self):
        self._config = Config()
        self._models = []

    def add_model(self, model, inf, sup):
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
        delta_timestamp = np.array(range(2*x)) * step_x * waterfall_duration / original_shape[0] + size_x / 2 * waterfall_duration / original_shape[0] + initial_timestamp
        delta_timestamp = delta_timestamp[delta_timestamp < waterfall_duration]
        delta_timestamp = delta_timestamp.reshape(delta_timestamp.shape[0], 1)

        print("avant",data.shape)
        data = decompose(data, (size_x, original_shape[1]), overlap)
        print("après",data.shape)
        # data est trop grand (2 waterfall), donc il faut en garder le bon nombre
        out = [m[2].extract_features(data[m[0]:m[1],:])[0:len(delta_timestamp),:] for m in self._models]
        out = np.concatenate((delta_timestamp, out), axis=1)
        return out

    def reconstruct(self, data):
        out = [m[2].reconstruct(data[m[0]:m[1],:]) for m in self._models]
        return np.concatenate(out, axis=1)


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
