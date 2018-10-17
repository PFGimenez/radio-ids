from abc import ABC, abstractmethod

class AnomalyDetector(ABC):
    """
    Abstract class for all anomaly detector algorithms
    """

    @abstractmethod
    def learn(self, data, exo=None):
        pass

    @abstractmethod
    def predict(self, data):
        pass
