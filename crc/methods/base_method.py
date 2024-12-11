"""
Abstract base class for CRL methods.
"""
import copy

from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from torch.nn.utils import clip_grad_norm_
import wandb

from crc.utils import get_device, train_val_test_split


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

        self.scheduler = None

    @abstractmethod
    def _build_model(self):
        raise NotImplementedError

    @abstractmethod
    def _get_dataset(self):
        raise NotImplementedError

    def _get_optimizer(self):
        """Must be called after defining self.model!"""
        return torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

    @abstractmethod
    def train_step(self, data):
        raise NotImplementedError

    def train(self):
        train_idxs, val_idxs, test_idxs = train_val_test_split(
            np.arange(len(self.dataset)),
            train_size=self.train_size,
            val_size=self.val_size,
            random_state=self.seed)

        train_dataset = Subset(self.dataset, train_idxs)
        val_dataset = Subset(self.dataset, val_idxs)

        train_dataloader = DataLoader(train_dataset, shuffle=True,
                                      batch_size=self.batch_size)
        val_dataloader = DataLoader(val_dataset, shuffle=False,
                                    batch_size=self.batch_size)

        best_val_loss = np.inf
        best_model = copy.deepcopy(self.model)

        for epoch in range(self.epochs):
            self.model.train()
            # Training
            for data in train_dataloader:
                # Zero gradients
                self.optimizer.zero_grad()

                total_loss, loss_dict = self.train_step(data)  # dict

                # Log losses
                wandb.log({f'{key}_train': value for key, value in loss_dict.items()})

                # Apply gradients
                total_loss.backward()
                # TODO: check if we always need gradient clipping
                # clip_grad_norm_(self.model.parameters(), max_norm=2.0,
                #                 norm_type=2)
                self.optimizer.step()

            # Validation
            if (epoch+1) % self.val_step == 0 or epoch == (self.epochs-1):
                self.model.eval()
                val_loss_values = []
                with torch.no_grad():
                    for data in val_dataloader:
                        total_loss, loss_dict = self.train_step(data)
                        val_loss_values.append(total_loss.item())
                        # Log losses
                        wandb.log({f'{key}_val': value for key, value in loss_dict.items()})

                if np.mean(val_loss_values) <= best_val_loss:
                    best_val_loss = np.mean(val_loss_values)
                    best_model = copy.deepcopy(self.model)

                # Optional lr scheduling
                if self.scheduler is not None:
                    self.scheduler.step(np.mean(val_loss_values))  # TODO this only works for the plateu scheduler!

        return best_model
