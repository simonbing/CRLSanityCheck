"""
Abstract base class for CRL methods.
"""
import copy
import random

from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from torch.nn.utils import clip_grad_norm_
import wandb

from crc.utils import get_device


class CRLMethod(ABC):
    def __init__(self, seed, dataset, task, data_root, d, batch_size, epochs,
                 lr):
        self.seed = seed

        # Set all seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.dataset = dataset
        self.task = task
        self.data_root = data_root

        self.d = d

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
    def get_dataset(self):
        raise NotImplementedError

    def _get_optimizer(self):
        """Must be called after defining self.model!"""
        return torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

    @abstractmethod
    def train_step(self, data):
        raise NotImplementedError

    def train(self, train_dataset, val_dataset):
        self.model = self.model.to(self.device)

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

    @abstractmethod
    def encode_step(self, data):
        raise NotImplementedError

    def get_encodings(self, test_dataset):
        self.model = self.model.to(self.device)
        self.model.eval()

        # Set eval mode to true in dataset
        test_dataset.dataset.eval = True

        train_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=2000)

        z_gt_list = []
        z_hat_list = []
        for data in train_dataloader:
            z_gt_batch = data[-1]  # return ground truth z last

            z_hat_batch = self.encode_step(data)

            z_gt_list.append(z_gt_batch)
            z_hat_list.append(z_hat_batch)

        z_gt = torch.cat(z_gt_list).cpu().detach().numpy()
        z_hat = torch.cat(z_hat_list).cpu().detach().numpy()

        return z_gt, z_hat

