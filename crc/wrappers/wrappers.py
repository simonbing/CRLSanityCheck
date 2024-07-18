from abc import ABC, abstractmethod
import os

import numpy as np


class TrainModel(ABC):
    def __init__(self, dataset, experiment, model, seed, batch_size, epochs,
                 lat_dim, run_name,
                 root_dir='/Users/Simon/Documents/PhD/Projects/CausalRepresentationChambers/results'):
        self.dataset = dataset
        self.experiment = experiment
        self.model = model
        self.run = run_name

        # This is where train/test data is saved
        self.model_dir = os.path.join(root_dir, self.dataset, self.experiment,
                                      self.model)
        self.train_dir = os.path.join(self.model_dir, self.run, 'train')
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        # Shared training hyperparameters
        self.seed = seed
        self.batch_size = batch_size
        self.epochs = epochs
        self.lat_dim = lat_dim

    @abstractmethod
    def train(self):
        raise NotImplementedError


class EvalModel(ABC):
    def __init__(self):
        pass
