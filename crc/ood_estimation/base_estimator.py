from abc import ABC, abstractmethod


class OODEstimator(ABC):
    def __init__(self, seed, task):
        self.seed = seed
        self.task = task

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_ood):
        raise NotImplementedError
