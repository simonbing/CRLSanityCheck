from abc import ABC, abstractmethod
import os

import torch

from crc.utils import get_device


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
    """
    This abstract base class serves as a wrapper for evaluating different models
    on multiple metrics.
    """
    def __init__(self, trained_model_path):
        """
        Args:
            trained_model_path: (str) Path to trained pytorch model.
        """
        self.device = get_device()
        self.trained_model = torch.load(trained_model_path)
        self.trained_model = self.trained_model.to(self.device)

    @abstractmethod
    def get_encodings(self, dataset_test):
        """
        Returns the learned latent encoding of the input data.

        Args:
            dataset_test: (torch.Dataset) Contains test data, including ground truth

        Returns:
            z: (np.array [n, latent_dims]) Ground truth latents
            z_hat: (np.array [n, latent_dims]) The learned latent encodings
        """
        raise NotImplementedError
