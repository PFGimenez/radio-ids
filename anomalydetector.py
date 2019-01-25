from abc import ABC, abstractmethod
import numpy as np

class AnomalyDetector(ABC):
    """
    Abstract class for all anomaly detector algorithms
    """

    # @abstractmethod
    # def preprocess(self, data):
    #     """
    #         Preprocess the train and test test
    #     """
    #     pass

    @abstractmethod
    def learn(self, data):
        """
            Learn from data
        """
        pass


    @abstractmethod
    def predict(self, data):
        pass

#    @abstractmethod
#    def predict(self, data, obs):
#        """
#            Predict whether there is an anomaly or not
#            data: previous observations
#            obs: observation to evaluate
#            Returns true iff the observation is an anomaly
#        """
#        pass

    def predict_list(self, data):
        return self.predict(data[:-1,:], data[-1,:].reshape(1,-1))

    @abstractmethod
    def get_memory_size(self):
        pass

    @abstractmethod
    def get_score(self, data):
        """ Return the score of data """
        pass

    @abstractmethod
    def anomalies_have_high_score(self):
        pass

    def learn_threshold(self, data):
        predictions = np.array([self.get_score(data[:i]) for i in range(1,len(data))])
        # on retire les None
        predictions = [x for x in predictions if x is not None]
        r = [0,1,2,3,4,5,7,10] if self.anomalies_have_high_score() else [100,99,98,97,96,95,93,90]
        self._thresholds = [np.percentile(predictions, p) for p in r]

    @abstractmethod
    def save(self, filename):
        pass

    @abstractmethod
    def load(self, filename):
        pass
