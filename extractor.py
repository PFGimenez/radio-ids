from config import Config
from abc import ABC, abstractmethod

class MultiExtractors:

    def __init__(self):
        self._config = Config()
        self._models = []
        original_shape = self._config.get_config_eval('waterfall_dimensions')
        self._size_x = self._config.get_config_eval('big_sweep_temporal_dimension')
        step_x = round(self._size_x * (1 - self._overlap))
        x = math.floor((original_shape[0] - self._size_x) / step_x) + 1
        self._delta_timestamp = np.array(range(2*x)) * step_x * waterfall_duration / original_shape[0] + self._size_x / 2 * waterfall_duration / original_shape[0]
        self._delta_timestamp = self._delta_timestamp[self._delta_timestamp < waterfall_duration]
        self._delta_timestamp = self._delta_timestamp.reshape(self._delta_timestamp.shape[0], 1)

    def add_model(self, model, inf, sup):
        self._models.append((inf, sup, model))

    def extract_features(self, data):
        out = [m[2].extract_features(data[m[0]:m[1],:]) for m in self._models]
        return np.concatenate(out)

    def save(self):
        for m in models:
            m[2].save(str(m[0])+"-"+str(m[1])+"-")

    def load(self, i, s, model):
        self._models.append((i, s, model.load(str(i)+"-"+str(s)+"-")))

    def extract_features(self, data, initial_timestamp):
        out = [m[2].extract_features(data) for m in self._models]
#        data = self._crop_samples(data)
#        data = self._add_samples(data)
#        out = self._coder.predict(data)
#        out = out.reshape(data.shape[0], -1)
#        out = out[0:len(self._delta_timestamp),:]
        out = np.concatenate((self._delta_timestamp + initial_timestamp, out), axis=1)
        return out

class FeatureExtractor(ABC):

    @abstractmethod
    def learn_extractor(self, filenames, inf, sup):
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
