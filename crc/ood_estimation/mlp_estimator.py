import copy
import os

import numpy as np
from skimage import io
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import AdamW
import wandb

from crc.ood_estimation.base_estimator import OODEstimator
from crc.utils.torch_utils import get_device


class MLPOODEstimator(OODEstimator):
    def __init__(self, seed, task, data_root, epochs, batch_size, learning_rate,
                 val_freq=5):
        super().__init__(seed, task, data_root)

        # Build model
        self.device = get_device()
        self.model = self._build_model()

        # Training params
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.val_freq = val_freq

        self.optim = self._get_optim()

    def _build_model(self):
        """
        Conv NN adapted from P. Lippe.
        """
        return MLPModule(in_channels=3, n_conv_layers=2, h_dim=64)

    def _get_optim(self):
        return AdamW(self.model.parameters(), lr=self.learning_rate)

    def train(self, X, y):
        # Build dataset and dataloader from training data
        dataset = ChamberImageDataset(X, y, self.data_root)

        train_idxs, val_idxs = train_test_split(range(len(dataset)),
                                                train_size=self.train_frac,
                                                shuffle=True,
                                                random_state=self.seed)

        train_dataset = Subset(dataset, train_idxs)
        val_dataset = Subset(dataset, val_idxs)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                shuffle=False)

        loss_fn = nn.MSELoss()
        self.model = self.model.to(self.device)

        best_val_loss = np.inf

        for i in range(self.epochs):
            # Training
            self.model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                y_hat_batch = self.model(X_batch)

                loss = loss_fn(y_hat_batch, y_batch)
                wandb.log({'train_loss': loss.item()})

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            # Eval
            if (i % self.val_freq) == 0:
                self.model.eval()
                epoch_val_loss = 0
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    y_hat_batch = self.model(X_batch)

                    val_loss = loss_fn(y_hat_batch, y_batch)

                    epoch_val_loss += val_loss
                wandb.log({'val_loss': epoch_val_loss.item()/len(val_loader)})
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    self.best_model = copy.deepcopy(self.model)

    def predict(self, X_ood):
        dataset = ChamberImageDataset(X_ood, None, self.data_root, mode='test')

        ood_loader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=False)

        self.best_model = self.best_model.to(self.device)
        self.best_model.eval()

        y_hat_list = []
        for X_batch in ood_loader:
            X_batch = X_batch.to(self.device)
            y_hat_batch = self.best_model(X_batch)
            y_hat_list.append(y_hat_batch.detach().cpu().numpy())

        y_hat = np.concatenate(y_hat_list)

        return y_hat


class MLPModule(nn.Module):
    def __init__(self, in_channels, n_conv_layers, h_dim, z_dim=5):
        super().__init__()

        self.in_channels = in_channels
        self.n_conv_layers = n_conv_layers
        self.h_dim = h_dim
        self.z_dim = z_dim

        NormLayer = lambda d: nn.GroupNorm(num_groups=8, num_channels=d)

        conv_layers = [
            nn.Sequential(
                nn.Conv2d(self.in_channels if i_layer == 0 else self.h_dim,
                          self.h_dim,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          bias=False),
                NormLayer(self.h_dim),
                nn.SiLU(),
                nn.Conv2d(self.h_dim, self.h_dim, kernel_size=3, stride=1, padding=1,
                          bias=False),
                NormLayer(self.h_dim),
                nn.SiLU()
            ) for i_layer in range(self.n_conv_layers)
        ]

        self.network = nn.Sequential(
            *conv_layers,
            nn.Flatten(),
            nn.Linear(16 * 16 * self.h_dim, 16 * self.h_dim),
            nn.LayerNorm(16 * self.h_dim),
            nn.SiLU(),
            nn.Linear(16 * self.h_dim, self.z_dim),
            nn.LayerNorm(self.z_dim),
            nn.SiLU()
        )

        self.lin_head = nn.Linear(self.z_dim, 1)

    def forward(self, x):
        z_hat = self.network(x)
        y_hat = self.lin_head(z_hat)

        return y_hat


class ChamberImageDataset(Dataset):
    def __init__(self, X, y, data_root, mode='train'):
        super().__init__()

        self.X = X
        self.y = y

        self.data_root = data_root

        self.mode = mode

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        img_path = os.path.join(self.data_root, self.X['image_file'].iloc[item])

        img_sample = io.imread(img_path)
        # Normalize
        img_sample = img_sample / 255.0

        if self.mode == 'train':
            return torch.as_tensor(img_sample.transpose((2, 0, 1)),
                                   dtype=torch.float32), \
                torch.as_tensor(self.y[item], dtype=torch.float32)
        else:
            return torch.as_tensor(img_sample.transpose((2, 0, 1)),
                                   dtype=torch.float32)
