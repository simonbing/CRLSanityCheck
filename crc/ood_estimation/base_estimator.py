from abc import ABC, abstractmethod


class OODEstimator(ABC):
    def __init__(self, seed, task, data_root):
        self.seed = seed
        self.task = task
        self.data_root = data_root

        self.train_frac = 0.8  # hardcoding for now

    @abstractmethod
    def train(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_ood):
        raise NotImplementedError
