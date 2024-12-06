"""
Abstract base class for CRL methods.
"""
from abc import ABC, abstractmethod
import torch

from crc.utils import get_device


class CRLMethod(ABC):
    def __init__(self, seed, dataset, task, data_root, d, batch_size, epochs, lr):
        self.seed = seed

        self.dataset = dataset
        self.task = task
        self.data_root = data_root

        self.d = d

        # Hardcoded for all experiments
        self.train_size = 0.8
        self.val_size = 0.1
        self.test_size = 0.1

        self.device = get_device()
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.val_step = 10

    @abstractmethod
    def _build_model(self):
        raise NotImplementedError

    @abstractmethod
    def _get_dataset(self):
        raise NotImplementedError

    def _get_optimizer(self):
        """Must be called after defining self.model!"""
        return torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

    def train(self):



        pass
