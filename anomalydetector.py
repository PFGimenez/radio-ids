from abc import ABC, abstractmethod

class AnomalyDetector(ABC):
    """
    Abstract class for all anomaly detector algorithms
    """

    @abstractmethod
    def preprocess(self, data):
        """
            Preprocess the train and test test
        """
        pass

    @abstractmethod
    def learn(self, data, exo=None):
        """
            Learn from data
        """
        pass

    @abstractmethod
    def predict(self, data, obs):
        """
            Predict whether there is an anomaly or not
            data: previous observations
            obs: observation to evaluate
            Returns true iff the observation is an anomaly
        """
        pass

    @abstractmethod
    def save(self, filename):
        pass

    @abstractmethod
    def load(self, filename):
        pass
