import random

import numpy as np
import torch

from crc.ood_estimation import OLSOODEstimator, get_chamber_data


class OODEstimatorApplication(object):
    def __init__(self, seed, estimation_model, task, data_root):
        self.seed = seed
        self.task = task  # encodes which environments to train and test on
        self.data_root = data_root
        self.estimator = self._get_estimator(estimation_model)

    def _get_estimator(self, estimation_model):
        if estimation_model == 'ols':
            return OLSOODEstimator(seed=self.seed, task=self.task,
                                   data_root=self.data_root)

    def run(self):
        # Set all seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Get train/test data
        X_df_train, X_df_test, y_train, y_test = \
            get_chamber_data(task=self.task,
                             data_root=self.data_root)

        # Train estimator on training environments
        self.estimator.train(X_df_train, y_train)

        # Get estimates on held out test environment
        y_hat = self.estimator.predict(X_df_test)

        mse_ood = np.mean((y_test - y_hat) ** 2)
        a = 0

        # Compute held out test metrics
