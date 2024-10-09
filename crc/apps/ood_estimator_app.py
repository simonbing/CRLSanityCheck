import random

import numpy as np
import torch
import wandb

from crc.ood_estimation import get_ood_task_data, OLSOODEstimator, \
    LassoOODEstimator, MLPOODEstimator, CRLOODEstimator


class OODEstimatorApplication(object):
    def __init__(self, seed, estimation_model, dataset, task, data_root, results_root, lat_dim, epochs,
                 batch_size, learning_rate, run_name):
        self.seed = seed
        self.dataset = dataset
        self.task = task  # encodes which environments to train and test on
        self.data_root = data_root
        self.results_root = results_root

        # NN training params
        self.lat_dim = lat_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.run_name = run_name

        self.estimator = self._get_estimator(estimation_model)

    def _get_estimator(self, estimation_model):
        if estimation_model == 'ols':
            return OLSOODEstimator(seed=self.seed, task=self.task,
                                   data_root=self.data_root)
        elif estimation_model == 'lasso':
            return LassoOODEstimator(seed=self.seed, task=self.task,
                                     data_root=self.data_root)
        elif estimation_model == 'mlp':
            return MLPOODEstimator(seed=self.seed,
                                   task=self.task,
                                   data_root=self.data_root,
                                   epochs=self.epochs,
                                   batch_size=self.batch_size,
                                   learning_rate=self.learning_rate)
        elif estimation_model in ['pcl', 'cmvae', 'contrast_crl']:
            return CRLOODEstimator(seed=self.seed,
                                   dataset=self.dataset,
                                   task=self.task,
                                   data_root=self.data_root,
                                   results_root=self.results_root,
                                   lat_dim=self.lat_dim,
                                   epochs=self.epochs,
                                   batch_size=self.batch_size,
                                   crl_model=estimation_model,
                                   run_name=self.run_name)

    def run(self):
        # Set all seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Get train/test data
        X_df_train, X_df_test, y_train, y_test = \
            get_ood_task_data(task=self.task,
                              data_root=self.data_root)

        # Train estimator on training environments
        self.estimator.train(X_df_train, y_train)

        # Get estimates on held out test environment
        y_hat = self.estimator.predict(X_df_test)

        mse_ood = np.mean((y_test - y_hat) ** 2)

        wandb.run.summary['mse_ood'] = mse_ood
        a = 0

        # Compute held out test metrics
