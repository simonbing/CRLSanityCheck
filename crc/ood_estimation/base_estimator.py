from abc import ABC, abstractmethod


class OODEstimator(ABC):
    def __init__(self, seed, image_data, task, data_root, results_root):
        self.seed = seed
        self.image_data = image_data
        self.task = task
        self.data_root = data_root
        self.results_root = results_root

        self.train_frac = 0.8  # hardcoding for now

    @abstractmethod
    def train(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_ood):
        raise NotImplementedError
