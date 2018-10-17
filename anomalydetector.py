from abc import ABC, abstractmethod

class AnomalyDetector(ABC):
    """
    Abstract class for all anomaly detector algorithms
    """

    @abstractmethod
    def learn(self, data, exo=None):
        """
            Learn from data
        """
        pass

    @abstractmethod
    def predict(self, data):
        """
            Predict whether there is an anomaly or not
        """
        pass
