import logging
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import torch
from torch.utils.data import DataLoader
import wandb

from crc.ood_estimation.base_estimator import OODEstimator
from crc.ood_estimation.datasets import EmbeddingDataset, PCLEmbeddingDataset
from crc.baselines import TrainPCL, TrainCMVAE, TrainContrastCRL, TrainRGBBaseline
from crc.utils import get_device


class CRLOODEstimator(OODEstimator):
    def __init__(self, seed, image_data, task, dataset, data_root, results_root, crl_model, lat_dim,
                 batch_size,
                 epochs, run_name,
                 overwrite_data=False):
        super().__init__(seed, image_data, task, data_root, results_root)
        self.dataset = dataset
        self.lat_dim = lat_dim
        self.batch_size = batch_size
        self.epochs = epochs

        self.crl_model = crl_model

        self.run_name = run_name
        self.overwrite_data = overwrite_data

        # Get CRL trainer
        trainer = self._get_trainer()
        self.trainer = trainer(data_root=self.data_root,
                               dataset=self.dataset,
                               image_data=self.image_data,
                               task=self.task,
                               overwrite_data=self.overwrite_data,
                               model=self.crl_model,
                               run_name=self.run_name,
                               seed=self.seed,
                               batch_size=self.batch_size,
                               epochs=self.epochs,
                               lat_dim=self.lat_dim,
                               root_dir=self.results_root)
        self.device = get_device()

        # Linear head
        self.lin_model = LinearRegression()

    def _get_trainer(self):
        if self.crl_model == 'cmvae':
            return TrainCMVAE
        elif self.crl_model == 'contrast_crl':
            return TrainContrastCRL
        elif self.crl_model == 'pcl':
            return TrainPCL
        elif self.crl_model == 'rgb_baseline':
            return TrainRGBBaseline

    def _get_embed_dataset(self, X):
        if self.crl_model == 'pcl':
            dataset = PCLEmbeddingDataset(data=X, data_root=self.data_root)
        else:
            dataset = EmbeddingDataset(data=X, data_root=self.data_root)

        return dataset

    def train(self, X, y):
        self.trainer.train()

        # Get trained model
        trained_model_path = os.path.join(self.trainer.train_dir, 'best_model.pt')
        self.trained_model = torch.load(trained_model_path)

        # Get embeddings
        embed_dataset = self._get_embed_dataset(X)
        embed_dataloader = DataLoader(embed_dataset, batch_size=self.batch_size,
                                      shuffle=False)

        z_hat_list = []
        self.trained_model = self.trained_model.to(self.device)
        self.trained_model.eval()

        for x_batch in embed_dataloader:
            x_batch = x_batch.to(self.device)

            z_hat_batch = self.trained_model.get_z(x_batch)
            z_hat_list.append(z_hat_batch.detach().cpu().numpy())

        z_hat = np.concatenate(z_hat_list)

        z_hat_train, z_hat_test, y_train, y_test = train_test_split(z_hat, y,
                                                                    train_size=self.train_frac,
                                                                    shuffle=True,
                                                                    random_state=self.seed)

        # Train linear regression with embedding and labels
        # Discarding first sample because of PCL dataloader quirk
        self.lin_model.fit(z_hat_train[1:, :], y_train[1:, :])

        y_hat_test = self.lin_model.predict(z_hat_test)
        mse_id = np.mean((y_hat_test - y_test) ** 2)
        logging.info(f'ID mse: {mse_id}')
        wandb.run.summary['mse_id'] = mse_id

    def predict(self, X_ood):
        # Get embeddings
        embed_dataset = self._get_embed_dataset(X_ood)
        embed_dataloader = DataLoader(embed_dataset, batch_size=self.batch_size,
                                      shuffle=False)

        z_hat_ood_list = []
        self.trained_model = self.trained_model.to(self.device)
        self.trained_model.eval()

        for x_batch in embed_dataloader:
            x_batch = x_batch.to(self.device)

            z_hat_batch = self.trained_model.get_z(x_batch)
            z_hat_ood_list.append(z_hat_batch.detach().cpu().numpy())

        z_hat_ood = np.concatenate(z_hat_ood_list)

        y_hat = self.lin_model.predict(z_hat_ood)

        return y_hat
